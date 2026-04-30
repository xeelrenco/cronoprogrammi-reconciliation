import argparse
import json
import math
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

import pandas as pd

from timeline_reconciliation_common import (
    CONFIG_FILE,
    CRONOPROGRAMMI_DIR,
    OUTPUT_DIR,
    build_task_text,
    chat_json,
    connect_motherduck,
    extract_project_code,
    load_task_with_wbs,
    parse_config_txt,
    remove_prefix,
    serialize_date_value,
)


DEFAULT_SAMPLE_SEED = 42
DEFAULT_PROGRESS_EVERY = 25
CREATED_BY = "1_classify_timeline_tasks.py"
PROMPT_VERSION = "timeline_task_classification_v1"
BATCH_IDS_FILE = Path(__file__).resolve().parent / ".timeline_classify_last_batch_ids.json"
BATCH_MANIFEST_FILE = Path(__file__).resolve().parent / ".timeline_classify_last_batch_manifest.json"
BATCH_ENDPOINT = "/v1/chat/completions"
OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES = 209_715_200
DEFAULT_BATCH_TARGET_BYTES = 120_000_000
DEFAULT_BATCH_POLL_INTERVAL = 60
DEFAULT_ADAPTIVE_BATCH_INITIAL_LIMIT = 800
DEFAULT_ADAPTIVE_BATCH_MIN_LIMIT = 100
DEFAULT_ADAPTIVE_BATCH_MAX_LIMIT = 1200
DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR = 0.5
DEFAULT_ADAPTIVE_BATCH_GROWTH_FACTOR = 1.25


def compute_proportional_quotas(counts, total_limit):
    non_empty = {k: v for k, v in counts.items() if v > 0}
    if not non_empty:
        return {k: 0 for k in counts}
    total_rows = sum(non_empty.values())
    raw = {k: (total_limit * v / total_rows) for k, v in non_empty.items()}
    quotas = {k: int(math.floor(x)) for k, x in raw.items()}
    if total_limit >= len(non_empty):
        for k in non_empty:
            if quotas[k] == 0:
                quotas[k] = 1
    remaining = max(0, total_limit - sum(quotas.values()))
    remainders = sorted(((k, raw[k] - math.floor(raw[k])) for k in non_empty), key=lambda x: x[1], reverse=True)
    i = 0
    while remaining > 0 and remainders:
        k = remainders[i % len(remainders)][0]
        if quotas[k] < non_empty[k]:
            quotas[k] += 1
            remaining -= 1
        i += 1
        if i > 10 * len(remainders) and remaining > 0:
            for kk in [name for name, count in non_empty.items() if quotas[name] < count]:
                if remaining <= 0:
                    break
                quotas[kk] += 1
                remaining -= 1
    out = {k: 0 for k in counts}
    out.update({k: min(v, non_empty[k]) for k, v in quotas.items()})
    return out


def build_or_load_sample_map(tasks_by_file, sample_limit, sample_file, random_seed=DEFAULT_SAMPLE_SEED):
    counts = {name: len(df) for name, df in tasks_by_file.items()}
    expected_quotas = compute_proportional_quotas(counts, sample_limit)
    if sample_file.exists():
        sample_df = pd.read_csv(sample_file, dtype={"cronoprogramma": str, "task_row_id": int})
        valid = len(sample_df) == sample_limit
        valid = valid and set(sample_df["cronoprogramma"].unique()).issubset(set(tasks_by_file.keys()))
        if valid:
            existing_counts = sample_df["cronoprogramma"].value_counts().to_dict()
            valid = all(existing_counts.get(name, 0) == quota for name, quota in expected_quotas.items())
        if valid:
            sample_map = {}
            for name, task in tasks_by_file.items():
                ids = set(sample_df[sample_df["cronoprogramma"] == name]["task_row_id"].tolist())
                sample_map[name] = task[task["task_row_id"].isin(ids)].copy() if ids else task.iloc[0:0].copy()
            return sample_map, sample_df, False

    rows = []
    sample_map = {}
    for i, (name, task) in enumerate(tasks_by_file.items()):
        q = expected_quotas.get(name, 0)
        if q <= 0:
            sample_map[name] = task.iloc[0:0].copy()
            continue
        sampled = task.sample(n=q, random_state=random_seed + i).copy()
        sample_map[name] = sampled
        for _, r in sampled.iterrows():
            rows.append(
                {
                    "cronoprogramma": name,
                    "task_row_id": int(r["task_row_id"]),
                    "task_code": r.get("task_code"),
                    "task_name": r.get("task_name"),
                    "wbs_name": r.get("wbs_name"),
                }
            )
    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(sample_file, index=False, encoding="utf-8")
    return sample_map, sample_df, True


def classify_task(task_name, wbs_name, cfg):
    system, user = build_classifier_prompts(task_name, wbs_name)
    fallback = {"task_class": "OTHER", "confidence": "LOW", "reason_short": "LLM unavailable or invalid response"}
    try:
        parsed = chat_json(cfg, system, user, timeout=45)
    except Exception:
        return fallback
    return _parse_classification_result(parsed)


def build_classifier_prompts(task_name, wbs_name):
    system = """
You classify Primavera P6 schedule tasks.

Return ONLY valid JSON.

Classify each task into exactly one class:

ENG_DOC:
The task represents progress, issue, review, approval, revision, delivery, or update
of engineering documents/deliverables that can be linked to MDR/RACI documents.

OTHER:
The task is not direct MDR/RACI document progress. This includes procurement/material phases,
construction, installation, testing, commissioning, meetings, milestones, site activities,
manufacturing, delivery, and generic project activities.

Important:
- If the activity refers to procurement/material purchasing phases, classify OTHER.
- Use wbs_name only as supporting context, not as the only reason.
- If uncertain, use OTHER with LOW or MEDIUM confidence.

JSON schema:
{
  "task_class": "ENG_DOC" | "OTHER",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reason_short": "brief reason in English"
}
"""
    user = {
        "task_name": task_name,
        "task_name_clean": remove_prefix(task_name),
        "wbs_name": wbs_name,
    }
    return system, user


def _llm_base_url(cfg):
    return cfg.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _llm_headers(cfg, content_type="application/json"):
    api_key = cfg.get("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_API_KEY mancante in config.txt")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": content_type}


def _http_post_json(cfg, url_path, payload, timeout=120):
    req = urllib.request.Request(
        f"{_llm_base_url(cfg)}{url_path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=_llm_headers(cfg),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(cfg, url_path, timeout=120):
    req = urllib.request.Request(
        f"{_llm_base_url(cfg)}{url_path}",
        headers=_llm_headers(cfg),
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(cfg, url_path, timeout=120):
    req = urllib.request.Request(
        f"{_llm_base_url(cfg)}{url_path}",
        headers={"Authorization": _llm_headers(cfg)["Authorization"]},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _upload_batch_input_file(cfg, file_path):
    boundary = f"----boundary{uuid.uuid4().hex}"
    filename = Path(file_path).name
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    lines = []
    lines.append(f"--{boundary}\r\n")
    lines.append('Content-Disposition: form-data; name="purpose"\r\n\r\n')
    lines.append("batch\r\n")
    lines.append(f"--{boundary}\r\n")
    lines.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n')
    lines.append("Content-Type: application/jsonl\r\n\r\n")
    body_prefix = "".join(lines).encode("utf-8")
    body_suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = body_prefix + file_bytes + body_suffix
    req = urllib.request.Request(
        f"{_llm_base_url(cfg)}/files",
        data=body,
        headers=_llm_headers(cfg, content_type=f"multipart/form-data; boundary={boundary}"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_classification_result(parsed: Dict[str, Any]) -> Dict[str, str]:
    fallback = {"task_class": "OTHER", "confidence": "LOW", "reason_short": "LLM unavailable or invalid response"}
    if not isinstance(parsed, dict):
        return fallback
    label = str(parsed.get("task_class", "OTHER")).upper().strip()
    if label not in ("ENG_DOC", "OTHER"):
        return fallback
    confidence = str(parsed.get("confidence", "LOW")).upper().strip()
    if confidence not in ("HIGH", "MEDIUM", "LOW"):
        confidence = "LOW"
    return {"task_class": label, "confidence": confidence, "reason_short": str(parsed.get("reason_short", ""))[:300]}


def _build_batch_line(task_ref: Dict[str, Any], cfg: Dict[str, str]) -> str:
    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    system, user = build_classifier_prompts(task_ref.get("task_name", ""), task_ref.get("wbs_name", ""))
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    return json.dumps(
        {
            "custom_id": task_ref["custom_id"],
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": body,
        },
        ensure_ascii=False,
    )


def _extract_batch_text(row: Dict[str, Any]) -> Optional[str]:
    response = row.get("response") or {}
    body = response.get("body") if isinstance(response, dict) else {}
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(item.get("text", ""))
        return "\n".join([x for x in out if x]).strip() or None
    return None


def _flatten_tasks_for_batch(tasks_by_file, sample_map=None):
    task_refs = []
    for timeline, task in tasks_by_file.items():
        selected = sample_map.get(timeline, task.iloc[0:0].copy()) if sample_map else task
        if selected.empty:
            continue
        for _, row in selected.iterrows():
            task_row_id = int(row["task_row_id"])
            task_refs.append(
                {
                    "timeline_name": timeline,
                    "task_row_id": task_row_id,
                    "task_code": row.get("task_code"),
                    "task_name": str(row.get("task_name", "")),
                    "wbs_name": str(row.get("wbs_name", "")),
                    "task_start_date": serialize_date_value(row.get("task_start_date")),
                    "task_finish_date": serialize_date_value(row.get("task_finish_date")),
                    "task_actual_start_date": serialize_date_value(row.get("task_actual_start_date")),
                    "task_actual_finish_date": serialize_date_value(row.get("task_actual_finish_date")),
                    "task_date_fields_json": row.get("task_date_fields_json"),
                    "custom_id": f"{timeline}::{task_row_id}",
                }
            )
    return task_refs


def run_batch_submit(task_refs: List[Dict[str, Any]], cfg: Dict[str, str], target_max_bytes: int) -> List[str]:
    if target_max_bytes <= 0 or target_max_bytes > OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES:
        raise ValueError(
            f"--batch-max-bytes deve essere > 0 e <= {OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES}"
        )
    if not task_refs:
        return []
    batch_ids: List[str] = []
    chunks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_ids: List[str] = []
    current_bytes = 0
    for ref in task_refs:
        line = _build_batch_line(ref, cfg)
        line_bytes = len((line + "\n").encode("utf-8"))
        if line_bytes > target_max_bytes:
            print(f"Skip {ref['custom_id']}: request troppo grande ({line_bytes} bytes)")
            continue
        if current_lines and current_bytes + line_bytes > target_max_bytes:
            chunks.append({"lines": current_lines, "custom_ids": current_ids, "size_bytes": current_bytes})
            current_lines = []
            current_ids = []
            current_bytes = 0
        current_lines.append(line)
        current_ids.append(ref["custom_id"])
        current_bytes += line_bytes
    if current_lines:
        chunks.append({"lines": current_lines, "custom_ids": current_ids, "size_bytes": current_bytes})

    for i, chunk in enumerate(chunks, 1):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for line in chunk["lines"]:
                f.write(line + "\n")
            tmp_path = f.name
        try:
            upload = _upload_batch_input_file(cfg, tmp_path)
            batch = _http_post_json(
                cfg,
                "/batches",
                {"input_file_id": upload["id"], "endpoint": BATCH_ENDPOINT, "completion_window": "24h"},
            )
            batch_id = str(batch["id"])
            batch_ids.append(batch_id)
            print(f"Submitted chunk {i}/{len(chunks)} -> batch_id={batch_id}, tasks={len(chunk['custom_ids'])}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    BATCH_IDS_FILE.write_text(json.dumps(batch_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    BATCH_MANIFEST_FILE.write_text(
        json.dumps(
            {
                "created_at": int(time.time()),
                "prompt_version": PROMPT_VERSION,
                "model": cfg.get("LLM_MODEL", "gpt-4o-mini"),
                "batch_ids": batch_ids,
                "task_refs": task_refs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return batch_ids


def _wait_batch_completed(cfg: Dict[str, str], batch_id: str, poll_interval_sec: int):
    while True:
        batch = _http_get_json(cfg, f"/batches/{batch_id}")
        status = str(batch.get("status", ""))
        if status == "completed":
            return batch
        if status in ("failed", "cancelled", "expired"):
            return batch
        print(f"Batch {batch_id} status={status}, attendo {poll_interval_sec}s...")
        time.sleep(poll_interval_sec)


def run_batch_collect(cfg: Dict[str, str], poll_interval_sec: int, fill_missing_fallback: bool = True):
    if not BATCH_MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest batch non trovato: {BATCH_MANIFEST_FILE}")
    manifest = json.loads(BATCH_MANIFEST_FILE.read_text(encoding="utf-8"))
    batch_ids = manifest.get("batch_ids") or []
    task_refs = manifest.get("task_refs") or []
    if not batch_ids:
        print("Nessun batch id nel manifest.")
        return None

    ref_by_custom_id = {str(r["custom_id"]): r for r in task_refs}
    result_by_custom_id: Dict[str, Dict[str, str]] = {}
    failed_batch_ids: List[str] = []
    for batch_id in batch_ids:
        batch = _wait_batch_completed(cfg, str(batch_id), poll_interval_sec)
        status = str(batch.get("status", ""))
        if status != "completed":
            print(f"Batch {batch_id} non completato: status={status}")
            failed_batch_ids.append(str(batch_id))
            continue
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            print(f"Batch {batch_id} completato ma output_file_id assente.")
            continue
        raw = _http_get_bytes(cfg, f"/files/{output_file_id}/content").decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = str(row.get("custom_id") or "")
            if not custom_id:
                continue
            text = _extract_batch_text(row)
            if not text:
                result_by_custom_id[custom_id] = {
                    "task_class": "OTHER",
                    "confidence": "LOW",
                    "reason_short": "Batch response missing content",
                }
                continue
            try:
                parsed = json.loads(text.strip().strip("`"))
            except Exception:
                parsed = {}
            result_by_custom_id[custom_id] = _parse_classification_result(parsed)

    if fill_missing_fallback:
        missing = [cid for cid in ref_by_custom_id if cid not in result_by_custom_id]
        for cid in missing:
            result_by_custom_id[cid] = {
                "task_class": "OTHER",
                "confidence": "LOW",
                "reason_short": "No batch result found",
            }
    return manifest, result_by_custom_id, failed_batch_ids


def persist_classification_outputs(
    manifest: Dict[str, Any],
    result_by_custom_id: Dict[str, Dict[str, str]],
    cfg: Dict[str, str],
    conn=None,
    db_name: str = "my_db",
):
    task_refs = manifest.get("task_refs") or []
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for ref in task_refs:
        custom_id = str(ref.get("custom_id"))
        timeline = str(ref.get("timeline_name"))
        cls = result_by_custom_id.get(
            custom_id,
            {"task_class": "OTHER", "confidence": "LOW", "reason_short": "No batch result found"},
        )
        grouped_rows.setdefault(timeline, []).append(
            {
                "task_row_id": int(ref.get("task_row_id")),
                "task_code": ref.get("task_code"),
                "task_name": ref.get("task_name"),
                "wbs_name": ref.get("wbs_name"),
                "task_start_date": ref.get("task_start_date"),
                "task_finish_date": ref.get("task_finish_date"),
                "task_actual_start_date": ref.get("task_actual_start_date"),
                "task_actual_finish_date": ref.get("task_actual_finish_date"),
                "task_date_fields_json": ref.get("task_date_fields_json"),
                "task_class": cls["task_class"],
                "classification_confidence": cls["confidence"],
                "classification_reason": cls["reason_short"],
            }
        )

    saved_rows = 0
    for timeline, rows in grouped_rows.items():
        classified = pd.DataFrame(rows).sort_values("task_row_id").reset_index(drop=True)
        out_path = OUTPUT_DIR / f"classification_{timeline}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            classified[
                [
                    "task_code",
                    "task_name",
                    "wbs_name",
                    "task_start_date",
                    "task_finish_date",
                    "task_actual_start_date",
                    "task_actual_finish_date",
                    "task_class",
                    "classification_confidence",
                    "classification_reason",
                ]
            ].to_excel(writer, sheet_name="Task_LLM_classification", index=False)
        print(f"Creato: {out_path}")
        if conn is not None:
            rows_df = build_staging_rows(classified, timeline, cfg)
            saved_rows += save_classified_tasks(conn, db_name, rows_df)
            print(f"[{timeline}] righe salvate in TimelineTasksClassified: {len(rows_df)}")
    return saved_rows


def run_batch_and_collect_adaptive(
    task_refs: List[Dict[str, Any]],
    cfg: Dict[str, str],
    conn,
    db_name: str,
    target_max_bytes: int,
    poll_interval_sec: int,
    initial_limit: int,
    min_limit: int,
    max_limit: int,
    backoff_factor: float,
    growth_factor: float,
    max_rounds: Optional[int],
):
    pending_by_id = {str(r["custom_id"]): r for r in task_refs}
    current_limit = max(min_limit, min(max_limit, int(initial_limit)))
    round_no = 0
    total_saved_rows = 0

    while pending_by_id:
        if max_rounds is not None and round_no >= max_rounds:
            print(f"Stop adaptive: raggiunto max-rounds={max_rounds}")
            break
        round_no += 1
        tranche = list(pending_by_id.values())[:current_limit]
        if not tranche:
            break
        print(
            f"[Adaptive round {round_no}] submit {len(tranche)} task "
            f"(limit={current_limit}, pending={len(pending_by_id)})"
        )
        batch_ids = run_batch_submit(task_refs=tranche, cfg=cfg, target_max_bytes=target_max_bytes)
        if not batch_ids:
            print(f"[Adaptive round {round_no}] nessun batch inviato, applico backoff.")
            next_limit = max(min_limit, int(max(1, current_limit) * float(backoff_factor)))
            if next_limit == current_limit and current_limit > min_limit:
                next_limit = current_limit - 1
            current_limit = max(min_limit, next_limit)
            continue

        collected = run_batch_collect(cfg=cfg, poll_interval_sec=poll_interval_sec, fill_missing_fallback=False)
        if not collected:
            print(f"[Adaptive round {round_no}] collect vuoto, applico backoff.")
            next_limit = max(min_limit, int(max(1, current_limit) * float(backoff_factor)))
            if next_limit == current_limit and current_limit > min_limit:
                next_limit = current_limit - 1
            current_limit = max(min_limit, next_limit)
            continue

        manifest, result_by_custom_id, failed_batch_ids = collected
        if failed_batch_ids:
            print(
                f"[Adaptive round {round_no}] batch non completati ({len(failed_batch_ids)}): "
                f"{', '.join(failed_batch_ids)}"
            )

        # Salva solo i risultati realmente presenti nel collect.
        successful_custom_ids = [
            str(ref.get("custom_id"))
            for ref in (manifest.get("task_refs") or [])
            if str(ref.get("custom_id")) in result_by_custom_id
        ]
        if successful_custom_ids:
            filtered_manifest = {
                "task_refs": [
                    ref for ref in (manifest.get("task_refs") or []) if str(ref.get("custom_id")) in successful_custom_ids
                ]
            }
            filtered_results = {cid: result_by_custom_id[cid] for cid in successful_custom_ids}
            total_saved_rows += persist_classification_outputs(
                manifest=filtered_manifest,
                result_by_custom_id=filtered_results,
                cfg=cfg,
                conn=conn,
                db_name=db_name,
            )
            for cid in successful_custom_ids:
                pending_by_id.pop(cid, None)

        if failed_batch_ids:
            next_limit = max(min_limit, int(max(1, current_limit) * float(backoff_factor)))
            if next_limit == current_limit and current_limit > min_limit:
                next_limit = current_limit - 1
            current_limit = max(min_limit, next_limit)
            print(f"[Adaptive round {round_no}] backoff -> next limit={current_limit}")
        else:
            next_limit = min(max_limit, int(max(1, current_limit) * float(growth_factor)))
            if next_limit == current_limit and current_limit < max_limit:
                next_limit = current_limit + 1
            current_limit = min(max_limit, next_limit)
            print(
                f"[Adaptive round {round_no}] completed, pending={len(pending_by_id)}, "
                f"next limit={current_limit}"
            )

    return total_saved_rows, len(pending_by_id)


def classify_tasks(task, cfg, file_label="", progress_every=DEFAULT_PROGRESS_EVERY):
    out = task.copy()
    results = []
    started = time.time()
    total = len(out)
    for idx, (_, row) in enumerate(out.iterrows(), 1):
        results.append(classify_task(str(row.get("task_name", "")), str(row.get("wbs_name", "")), cfg))
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[{file_label}] classified {idx}/{total} tasks (elapsed {round(time.time() - started, 1)}s)")
    out["task_class"] = [x["task_class"] for x in results]
    out["classification_confidence"] = [x["confidence"] for x in results]
    out["classification_reason"] = [x["reason_short"] for x in results]
    return out


def build_staging_rows(task, timeline_name, cfg):
    return pd.DataFrame(
        {
            "TimelineName": timeline_name,
            "ProjectCode": extract_project_code(timeline_name),
            "TaskRowId": task["task_row_id"],
            "TaskCode": task.get("task_code"),
            "TaskName": task.get("task_name"),
            "WbsName": task.get("wbs_name"),
            "TaskStartDate": task.get("task_start_date"),
            "TaskFinishDate": task.get("task_finish_date"),
            "TaskActualStartDate": task.get("task_actual_start_date"),
            "TaskActualFinishDate": task.get("task_actual_finish_date"),
            "TaskDateFieldsJson": task.get("task_date_fields_json"),
            "TaskText": task.apply(lambda r: build_task_text(r.get("task_name"), r.get("wbs_name"), r.get("task_class")), axis=1),
            "TaskClass": task["task_class"],
            "TaskClassConfidence": task["classification_confidence"],
            "TaskClassReason": task["classification_reason"],
            "ClassifierModel": cfg.get("LLM_MODEL", "gpt-4o-mini"),
            "ClassifierPromptVersion": PROMPT_VERSION,
            "CreatedBy": CREATED_BY,
        }
    )


def save_classified_tasks(conn, db_name, rows):
    conn.register("classified_rows", rows)
    try:
        conn.execute(
            f"""
            INSERT INTO {db_name}.timeline_reconciliation.TimelineTasksClassified (
                TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                TaskStartDate, TaskFinishDate, TaskActualStartDate, TaskActualFinishDate,
                TaskDateFieldsJson, TaskText,
                TaskClass, TaskClassConfidence, TaskClassReason, ClassifierModel,
                ClassifierPromptVersion, CreatedBy
            )
            SELECT
                TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                TaskStartDate, TaskFinishDate, TaskActualStartDate, TaskActualFinishDate,
                TaskDateFieldsJson, TaskText,
                TaskClass, TaskClassConfidence, TaskClassReason, ClassifierModel,
                ClassifierPromptVersion, CreatedBy
            FROM classified_rows
            """
        )
    finally:
        conn.unregister("classified_rows")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="1 classify Primavera timeline tasks into ENG_DOC or OTHER")
    parser.add_argument("--limit", type=int, default=0, help="Totale task da classificare. 0 = nessun limite.")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--save-db", action="store_true", help="Salva su DB anche quando si usa --limit.")
    parser.add_argument("--batch-submit", action="store_true", help="Invia richieste in OpenAI Batch API e termina.")
    parser.add_argument("--batch-collect", action="store_true", help="Colleziona output dei batch inviati e salva output/DB.")
    parser.add_argument(
        "--batch-and-collect",
        action="store_true",
        help="Loop adattivo: submit tranche, collect, salva, poi continua automaticamente.",
    )
    parser.add_argument("--batch-max-bytes", type=int, default=DEFAULT_BATCH_TARGET_BYTES)
    parser.add_argument("--batch-poll-interval", type=int, default=DEFAULT_BATCH_POLL_INTERVAL)
    parser.add_argument("--batch-initial-limit", type=int, default=DEFAULT_ADAPTIVE_BATCH_INITIAL_LIMIT)
    parser.add_argument("--batch-min-limit", type=int, default=DEFAULT_ADAPTIVE_BATCH_MIN_LIMIT)
    parser.add_argument("--batch-max-limit", type=int, default=DEFAULT_ADAPTIVE_BATCH_MAX_LIMIT)
    parser.add_argument("--batch-backoff-factor", type=float, default=DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR)
    parser.add_argument("--batch-growth-factor", type=float, default=DEFAULT_ADAPTIVE_BATCH_GROWTH_FACTOR)
    parser.add_argument("--batch-max-rounds", type=int, default=None)
    args = parser.parse_args()
    selected_batch_modes = int(args.batch_submit) + int(args.batch_collect) + int(args.batch_and_collect)
    if selected_batch_modes > 1:
        raise RuntimeError("Usa una sola modalità batch: --batch-submit, --batch-collect, oppure --batch-and-collect")
    if args.batch_and_collect and args.batch_min_limit <= 0:
        raise RuntimeError("--batch-min-limit deve essere > 0")
    if args.batch_and_collect and args.batch_max_limit < args.batch_min_limit:
        raise RuntimeError("--batch-max-limit deve essere >= --batch-min-limit")
    if args.batch_and_collect and not (0 < args.batch_backoff_factor <= 1):
        raise RuntimeError("--batch-backoff-factor deve essere in (0, 1]")
    if args.batch_and_collect and args.batch_growth_factor < 1:
        raise RuntimeError("--batch-growth-factor deve essere >= 1")

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    save_to_db = args.save_db or args.limit == 0 or args.batch_collect or args.batch_and_collect
    conn = connect_motherduck(cfg) if save_to_db else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(CRONOPROGRAMMI_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Nessun file .xlsx in {CRONOPROGRAMMI_DIR}")

    tasks_by_file = {}
    for prim in files:
        try:
            tasks_by_file[prim.stem] = load_task_with_wbs(prim)
        except Exception as exc:
            print(f"Skip {prim.name}: {exc}")

    sample_map = None
    if args.limit > 0:
        sample_file = CONFIG_FILE.parent / f"task_sample_limit_{args.limit}.csv"
        sample_map, sample_metadata, created = build_or_load_sample_map(tasks_by_file, args.limit, sample_file)
        print(f"{'Creato' if created else 'Riutilizzato'} campione: {sample_file} ({len(sample_metadata)} task)")

    if args.batch_submit:
        task_refs = _flatten_tasks_for_batch(tasks_by_file, sample_map=sample_map)
        if not task_refs:
            print("Nessuna task selezionata per batch submit.")
            if conn is not None:
                conn.close()
            return
        batch_ids = run_batch_submit(task_refs=task_refs, cfg=cfg, target_max_bytes=args.batch_max_bytes)
        if conn is not None:
            conn.close()
        if not batch_ids:
            print("Nessun batch inviato.")
            return
        print(f"Batch inviati: {len(batch_ids)}")
        print(f"Lista batch salvata in {BATCH_IDS_FILE.name}")
        print(f"Manifest salvato in {BATCH_MANIFEST_FILE.name}")
        print("Esegui dopo con --batch-collect per scrivere output e DB.")
        return

    if args.batch_collect:
        try:
            collected = run_batch_collect(
                cfg,
                poll_interval_sec=max(1, int(args.batch_poll_interval)),
                fill_missing_fallback=True,
            )
            if not collected:
                if conn is not None:
                    conn.close()
                return
            manifest, result_by_custom_id, _failed_batch_ids = collected
            saved_rows = persist_classification_outputs(
                manifest=manifest,
                result_by_custom_id=result_by_custom_id,
                cfg=cfg,
                conn=conn,
                db_name=db_name,
            )
            if conn is not None:
                print(f"Totale righe salvate: {saved_rows}")
        finally:
            if conn is not None:
                conn.close()
        return

    if args.batch_and_collect:
        if conn is None:
            raise RuntimeError("Connessione DB non disponibile in --batch-and-collect")
        try:
            task_refs = _flatten_tasks_for_batch(tasks_by_file, sample_map=sample_map)
            if not task_refs:
                print("Nessuna task selezionata per batch-and-collect.")
                return
            saved_rows, pending_left = run_batch_and_collect_adaptive(
                task_refs=task_refs,
                cfg=cfg,
                conn=conn,
                db_name=db_name,
                target_max_bytes=args.batch_max_bytes,
                poll_interval_sec=max(1, int(args.batch_poll_interval)),
                initial_limit=args.batch_initial_limit,
                min_limit=args.batch_min_limit,
                max_limit=args.batch_max_limit,
                backoff_factor=args.batch_backoff_factor,
                growth_factor=args.batch_growth_factor,
                max_rounds=args.batch_max_rounds,
            )
            print(f"Totale righe salvate: {saved_rows}")
            if pending_left > 0:
                print(f"Task ancora pending dopo loop adattivo: {pending_left}")
        finally:
            conn.close()
        return

    saved_rows = 0
    try:
        for prim in files:
            timeline = prim.stem
            if timeline not in tasks_by_file:
                continue
            task = sample_map.get(timeline, tasks_by_file[timeline].iloc[0:0].copy()) if sample_map else tasks_by_file[timeline]
            if task.empty:
                print(f"Skip {prim.name}: nessuna task selezionata")
                continue
            print(f"Processing {prim.name} - tasks: {len(task)}")
            classified = classify_tasks(task, cfg, file_label=timeline, progress_every=args.progress_every)

            out_path = OUTPUT_DIR / f"classification_{timeline}.xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                classified[
                    [
                        "task_code",
                        "task_name",
                        "wbs_name",
                        "task_start_date",
                        "task_finish_date",
                        "task_actual_start_date",
                        "task_actual_finish_date",
                        "task_class",
                        "classification_confidence",
                        "classification_reason",
                    ]
                ].to_excel(writer, sheet_name="Task_LLM_classification", index=False)
            print(f"Creato: {out_path}")

            if conn is not None:
                rows = build_staging_rows(classified, timeline, cfg)
                saved_rows += save_classified_tasks(conn, db_name, rows)
                print(f"[{timeline}] righe salvate in TimelineTasksClassified: {len(rows)}")
    finally:
        if conn is not None:
            conn.close()
    if conn is not None:
        print(f"Totale righe salvate: {saved_rows}")


if __name__ == "__main__":
    main()
