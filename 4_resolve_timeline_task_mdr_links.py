import argparse
import json
import tempfile
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from timeline_reconciliation_common import CONFIG_FILE, OUTPUT_DIR, chat_json, connect_motherduck, parse_config_txt, remove_prefix


CREATED_BY = "4_resolve_timeline_task_mdr_links.py"
LINK_METHOD = "embedding_topk_llm_resolver"
BATCH_IDS_FILE = Path(__file__).resolve().parent / ".timeline_resolver_last_batch_ids.json"
BATCH_MANIFEST_FILE = Path(__file__).resolve().parent / ".timeline_resolver_last_batch_manifest.json"
BATCH_ENDPOINT = "/v1/chat/completions"
OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES = 209_715_200
DEFAULT_BATCH_TARGET_BYTES = 120_000_000
DEFAULT_BATCH_POLL_INTERVAL = 60


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp01(value):
    return max(0.0, min(1.0, safe_float(value)))


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
    body_prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "batch\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode("utf-8")
    body_suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{_llm_base_url(cfg)}/files",
        data=body_prefix + file_bytes + body_suffix,
        headers=_llm_headers(cfg, content_type=f"multipart/form-data; boundary={boundary}"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_topk_for_resolver(conn, db_name, embedding_model, timeline_name=None, top_k=30):
    timeline_filter = ""
    params = [embedding_model, top_k]
    if timeline_name:
        timeline_filter = "AND k.TimelineName = ?"
        params.append(timeline_name)
    return conn.execute(
        f"""
        WITH ranked AS (
            SELECT *
            FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrCandidates k
            WHERE k.EmbeddingModel = ?
              AND k.Rank <= ?
              {timeline_filter}
        ),
        ranked_latest AS (
            SELECT *
            FROM (
                SELECT
                    r.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            r.TimelineName,
                            r.TaskRowId,
                            r.MdrDocumentTitle,
                            r.ConsolidatedTitleKey,
                            r.EmbeddingModel,
                            r.Rank
                        ORDER BY r.CreatedAt DESC
                    ) AS rn
                FROM ranked r
            ) x
            WHERE x.rn = 1
        ),
        tasks_latest AS (
            SELECT *
            FROM (
                SELECT
                    t.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.TimelineName, t.TaskRowId
                        ORDER BY t.UpdatedAt DESC, t.CreatedAt DESC
                    ) AS rn
                FROM {db_name}.timeline_reconciliation.TimelineTasksClassified t
            ) y
            WHERE y.rn = 1
        ),
        candidates_latest AS (
            SELECT *
            FROM (
                SELECT
                    c.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.TimelineName, c.MdrDocumentTitle, c.ConsolidatedTitleKey, c.EmbeddingModel
                        ORDER BY c.UpdatedAt DESC, c.CreatedAt DESC
                    ) AS rn
                FROM {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings c
                WHERE c.EmbeddingModel = ?
            ) z
            WHERE z.rn = 1
        )
        SELECT
            k.TimelineName,
            k.ProjectCode,
            k.TaskRowId,
            t.TaskCode,
            k.TaskName,
            k.WbsName,
            k.TaskStartDate,
            k.TaskFinishDate,
            k.TaskActualStartDate,
            k.TaskActualFinishDate,
            k.TaskDateFieldsJson,
            t.TaskClass,
            t.TaskClassConfidence,
            t.TaskClassReason,
            k.MdrDocumentTitle,
            k.MdrTitleKey,
            k.ConsolidatedTitleKey,
            k.ConsolidatedRaciTitle,
            k.Similarity,
            k.Rank AS RetrievalRank,
            c.ConsolidatedDecisionType,
            c.ConsolidatedConfidence,
            c.ConsolidatedReason,
            c.ConsolidatedSource,
            c.EffectiveDescription,
            c.DisciplineName,
            c.TypeName,
            c.CategoryDescription,
            c.ChapterName
        FROM ranked_latest k
        JOIN tasks_latest t
          ON t.TimelineName = k.TimelineName
         AND t.TaskRowId = k.TaskRowId
        LEFT JOIN candidates_latest c
          ON c.TimelineName = k.TimelineName
         AND c.MdrDocumentTitle = k.MdrDocumentTitle
         AND c.ConsolidatedTitleKey = k.ConsolidatedTitleKey
         AND c.EmbeddingModel = k.EmbeddingModel
        ORDER BY k.TimelineName, k.TaskRowId, k.Rank
        """,
        params + [embedding_model],
    ).fetchdf()


def validate_resolver_output(parsed, task_group):
    if not isinstance(parsed, dict):
        return _invalid_result("invalid_root", "LLM response root is not a JSON object")
    links = parsed.get("links", [])
    if not isinstance(links, list):
        return _invalid_result("invalid_links", "LLM response field links is not a list")

    valid_ids = set(task_group["RetrievalRank"].astype(int).tolist())
    best_by_candidate = {}
    dropped_invalid_count = 0
    duplicate_candidate_count = 0
    for link in links:
        if not isinstance(link, dict):
            dropped_invalid_count += 1
            continue
        try:
            candidate_id = int(link.get("candidate_id"))
        except Exception:
            dropped_invalid_count += 1
            continue
        if candidate_id not in valid_ids:
            dropped_invalid_count += 1
            continue
        confidence = clamp01(link.get("confidence", 0.0))
        reason_short = str(link.get("reason_short", "") or "")[:300]
        prev = best_by_candidate.get(candidate_id)
        if prev is not None:
            duplicate_candidate_count += 1
        if prev is None or confidence > prev["confidence"]:
            best_by_candidate[candidate_id] = {
                "candidate_id": candidate_id,
                "confidence": confidence,
                "reason_short": reason_short,
            }

    out_links = list(best_by_candidate.values())
    return {
        "status": "ok",
        "links": out_links,
        "error_type": "",
        "error_message": "",
        "raw_links_count": len(links),
        "valid_links_count": len(out_links),
        "dropped_invalid_count": dropped_invalid_count,
        "duplicate_candidate_count": duplicate_candidate_count,
    }


def _invalid_result(error_type, error_message):
    return {
        "status": "invalid_json",
        "links": [],
        "error_type": error_type,
        "error_message": error_message,
        "raw_links_count": 0,
        "valid_links_count": 0,
        "dropped_invalid_count": 0,
        "duplicate_candidate_count": 0,
    }


def build_resolver_prompts(task_group):
    first = task_group.iloc[0]
    candidates = []
    for _, row in task_group.iterrows():
        candidates.append(
            {
                "candidate_id": int(row["RetrievalRank"]),
                "similarity": float(row["Similarity"]),
                "mdr_document_title": str(row.get("MdrDocumentTitle", "")),
                "raci_title": str(row.get("ConsolidatedRaciTitle", "")),
                "raci_description": str(row.get("EffectiveDescription", "")),
                "discipline": str(row.get("DisciplineName", "")),
                "type": str(row.get("TypeName", "")),
                "category": str(row.get("CategoryDescription", "")),
                "chapter": str(row.get("ChapterName", "")),
            }
        )

    system = """
You resolve links between one Primavera schedule task and MDR/RACI document candidates.

Return ONLY valid JSON.

The task was previously classified as ENG_DOC, but you must still be conservative.
Select zero, one, or multiple MDR candidates from the provided Top-K list.

Core rule:
Link a candidate only if the Primavera task clearly represents progress, issue, review,
approval, revision, delivery, or update of that specific MDR/RACI document or document group.

Do NOT link when:
- the candidate is only generally related by discipline, chapter, category, or keywords
- the task is about procurement/material process rather than document progress
- the task is about RFQ, technical alignment, commercial alignment, issue of order,
  purchase order, vendor follow-up, logistics, construction, testing, commissioning,
  meetings, milestones, or generic project activities
- the match is only based on broad words such as document, drawing, specification,
  procedure, engineering, vendor, package, system
- the candidate title is semantically different from the task title

Use these signals in order:
1. MDR document title vs task_name_clean
2. RACI title / description as supporting context
3. discipline, type, category, chapter only as weak supporting metadata
4. embedding similarity only as retrieval evidence, never as proof

Multiple links are allowed only when the task clearly covers a bundle/group of documents,
not merely because several candidates are similar.

If task_class_confidence is LOW, be extra conservative and prefer no links.
If uncertain, return:
{"links": []}

Confidence guide:
- 0.90-1.00: near-certain same document/group
- 0.75-0.89: strong semantic match with supporting context
- 0.50-0.74: plausible but not certain
- below 0.50: do not return the link

JSON schema:
{
  "links": [
    {
      "candidate_id": 1,
      "confidence": 0.0,
      "reason_short": "brief reason in English"
    }
  ]
}
"""
    user = {
        "task": {
            "task_code": str(first.get("TaskCode", "")),
            "task_name": str(first.get("TaskName", "")),
            "task_name_clean": remove_prefix(first.get("TaskName", "")),
            "wbs_name": str(first.get("WbsName", "")),
            "task_start_date": str(first.get("TaskStartDate", "")),
            "task_finish_date": str(first.get("TaskFinishDate", "")),
            "task_actual_start_date": str(first.get("TaskActualStartDate", "")),
            "task_actual_finish_date": str(first.get("TaskActualFinishDate", "")),
            "task_date_fields_json": str(first.get("TaskDateFieldsJson", "")),
            "task_class_reason": str(first.get("TaskClassReason", "")),
            "task_class_confidence": str(first.get("TaskClassConfidence", "")),
        },
        "candidates": candidates,
    }
    return system, user


def resolve_task_links(task_group, cfg, llm_timeout_sec=60, retry_max=0, retry_backoff_sec=2.0):
    system, user = build_resolver_prompts(task_group)
    last_error = None
    for attempt in range(max(0, retry_max) + 1):
        try:
            parsed = chat_json(cfg, system, user, timeout=llm_timeout_sec)
            return validate_resolver_output(parsed, task_group)
        except Exception as exc:
            last_error = exc
            if attempt < retry_max:
                time.sleep(max(0.0, retry_backoff_sec) * (attempt + 1))
    return {
        "status": "llm_error",
        "links": [],
        "error_type": type(last_error).__name__ if last_error else "llm_error",
        "error_message": str(last_error or "LLM call failed")[:500],
        "raw_links_count": 0,
        "valid_links_count": 0,
        "dropped_invalid_count": 0,
        "duplicate_candidate_count": 0,
    }


def build_final_rows_for_group(
    timeline_name,
    task_row_id,
    group,
    resolved,
    min_link_confidence=0.0,
    max_links_per_task=0,
):
    rows = []
    first = group.iloc[0]
    scope_all = {"TimelineName": timeline_name, "TaskRowId": int(task_row_id)}
    scope_ok = None
    status = resolved.get("status", "invalid_json")
    raw_link_count = int(resolved.get("raw_links_count", 0) or 0)
    valid_link_count = int(resolved.get("valid_links_count", 0) or 0)
    duplicate_candidate_count = int(resolved.get("duplicate_candidate_count", 0) or 0)
    dropped_invalid_count = int(resolved.get("dropped_invalid_count", 0) or 0)
    dropped_by_threshold = 0
    saved_link_count = 0

    if status == "ok":
        scope_ok = {"TimelineName": timeline_name, "TaskRowId": int(task_row_id)}
        selected = resolved.get("links", [])
        before_threshold = len(selected)
        selected = [x for x in selected if safe_float(x.get("confidence", 0.0)) >= min_link_confidence]
        dropped_by_threshold = before_threshold - len(selected)
        similarity_by_id = {
            int(row["RetrievalRank"]): safe_float(row.get("Similarity", 0.0))
            for _, row in group.iterrows()
        }
        selected = sorted(
            selected,
            key=lambda x: (
                -safe_float(x.get("confidence", 0.0)),
                -similarity_by_id.get(int(x["candidate_id"]), 0.0),
                int(x["candidate_id"]),
            ),
        )
        if max_links_per_task and max_links_per_task > 0:
            selected = selected[:max_links_per_task]
        saved_link_count = len(selected)
        for link_rank, link in enumerate(selected, 1):
            cand = group[group["RetrievalRank"] == link["candidate_id"]].iloc[0]
            rows.append(
                {
                    "TimelineName": cand["TimelineName"],
                    "ProjectCode": cand["ProjectCode"],
                    "TaskRowId": int(cand["TaskRowId"]),
                    "TaskCode": cand.get("TaskCode"),
                    "TaskName": cand.get("TaskName"),
                    "WbsName": cand.get("WbsName"),
                    "TaskStartDate": cand.get("TaskStartDate"),
                    "TaskFinishDate": cand.get("TaskFinishDate"),
                    "TaskActualStartDate": cand.get("TaskActualStartDate"),
                    "TaskActualFinishDate": cand.get("TaskActualFinishDate"),
                    "TaskDateFieldsJson": cand.get("TaskDateFieldsJson"),
                    "TaskClass": cand.get("TaskClass"),
                    "TaskClassConfidence": cand.get("TaskClassConfidence"),
                    "TaskClassReason": cand.get("TaskClassReason"),
                    "MdrDocumentTitle": cand.get("MdrDocumentTitle"),
                    "MdrTitleKey": cand.get("MdrTitleKey"),
                    "LinkRank": link_rank,
                    "LinkScore": link["confidence"],
                    "LinkMethod": LINK_METHOD,
                    "LinkReason": link["reason_short"],
                    "ConsolidatedDecisionType": cand.get("ConsolidatedDecisionType"),
                    "ConsolidatedTitleKey": cand.get("ConsolidatedTitleKey"),
                    "ConsolidatedRaciTitle": cand.get("ConsolidatedRaciTitle"),
                    "ConsolidatedConfidence": cand.get("ConsolidatedConfidence"),
                    "ConsolidatedReason": cand.get("ConsolidatedReason"),
                    "ConsolidatedSource": cand.get("ConsolidatedSource"),
                    "CreatedBy": CREATED_BY,
                }
            )

    diagnostic = {
        "TimelineName": timeline_name,
        "ProjectCode": first.get("ProjectCode"),
        "TaskRowId": int(task_row_id),
        "TaskCode": first.get("TaskCode"),
        "TaskName": first.get("TaskName"),
        "TaskStatus": status,
        "ErrorType": resolved.get("error_type", ""),
        "ErrorMessage": resolved.get("error_message", ""),
        "RawLinkCount": raw_link_count,
        "ValidLinkCount": valid_link_count,
        "SavedLinkCount": saved_link_count,
        "DroppedInvalidCount": dropped_invalid_count,
        "DroppedByThreshold": dropped_by_threshold,
        "DuplicateCandidateCount": duplicate_candidate_count,
        "CreatedBy": CREATED_BY,
    }
    return rows, diagnostic, scope_all, scope_ok, status


def process_group_realtime(
    item,
    cfg,
    min_link_confidence,
    max_links_per_task,
    llm_timeout_sec,
    retry_max,
    retry_backoff_sec,
):
    (timeline_name, task_row_id), group = item
    resolved = resolve_task_links(
        group,
        cfg,
        llm_timeout_sec=llm_timeout_sec,
        retry_max=retry_max,
        retry_backoff_sec=retry_backoff_sec,
    )
    return build_final_rows_for_group(
        timeline_name,
        task_row_id,
        group,
        resolved,
        min_link_confidence=min_link_confidence,
        max_links_per_task=max_links_per_task,
    )


def combine_group_results(group_results):
    rows = []
    diagnostics = []
    resolved_scope_all = []
    resolved_scope_ok = []
    status_counts = {"ok": 0, "llm_error": 0, "invalid_json": 0}
    for group_rows, diagnostic, scope_all, scope_ok, status in group_results:
        rows.extend(group_rows)
        diagnostics.append(diagnostic)
        resolved_scope_all.append(scope_all)
        if scope_ok is not None:
            resolved_scope_ok.append(scope_ok)
        status_counts[status] = status_counts.get(status, 0) + 1
    return (
        pd.DataFrame(rows),
        pd.DataFrame(diagnostics),
        pd.DataFrame(resolved_scope_all),
        pd.DataFrame(resolved_scope_ok),
        status_counts,
    )


def build_final_links(
    topk,
    cfg,
    progress_every,
    min_link_confidence=0.0,
    max_links_per_task=0,
    llm_timeout_sec=60,
    retry_max=0,
    retry_backoff_sec=2.0,
    workers=1,
):
    groups = list(topk.groupby(["TimelineName", "TaskRowId"], sort=True))
    started = time.time()
    group_results = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_group_realtime,
                    item,
                    cfg,
                    min_link_confidence,
                    max_links_per_task,
                    llm_timeout_sec,
                    retry_max,
                    retry_backoff_sec,
                )
                for item in groups
            ]
            for idx, future in enumerate(as_completed(futures), 1):
                group_results.append(future.result())
                if progress_every > 0 and (idx % progress_every == 0 or idx == len(groups)):
                    print(f"Resolved {idx}/{len(groups)} tasks (elapsed {round(time.time() - started, 1)}s)")
    else:
        for idx, item in enumerate(groups, 1):
            group_results.append(
                process_group_realtime(
                    item,
                    cfg,
                    min_link_confidence,
                    max_links_per_task,
                    llm_timeout_sec,
                    retry_max,
                    retry_backoff_sec,
                )
            )
            if progress_every > 0 and (idx % progress_every == 0 or idx == len(groups)):
                print(f"Resolved {idx}/{len(groups)} tasks (elapsed {round(time.time() - started, 1)}s)")
    return combine_group_results(group_results)


def _extract_batch_text(row):
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


def _parse_json_text(text):
    content = str(text or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def _build_batch_line(custom_id, task_group, cfg):
    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    system, user = build_resolver_prompts(task_group)
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
            "custom_id": custom_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": body,
        },
        ensure_ascii=False,
    )


def run_batch_submit(topk, cfg, target_max_bytes, timeline_name=None, top_k=30):
    if target_max_bytes <= 0 or target_max_bytes > OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES:
        raise ValueError(f"--batch-max-bytes deve essere > 0 e <= {OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES}")

    chunks = []
    current_lines = []
    current_refs = []
    current_bytes = 0
    for (timeline, task_row_id), group in topk.groupby(["TimelineName", "TaskRowId"], sort=True):
        custom_id = uuid.uuid4().hex
        line = _build_batch_line(custom_id, group, cfg)
        line_bytes = len((line + "\n").encode("utf-8"))
        ref = {"custom_id": custom_id, "timeline_name": timeline, "task_row_id": int(task_row_id)}
        if line_bytes > target_max_bytes:
            print(f"Skip {timeline}::{task_row_id}: request troppo grande ({line_bytes} bytes)")
            continue
        if current_lines and current_bytes + line_bytes > target_max_bytes:
            chunks.append({"lines": current_lines, "refs": current_refs, "size_bytes": current_bytes})
            current_lines = []
            current_refs = []
            current_bytes = 0
        current_lines.append(line)
        current_refs.append(ref)
        current_bytes += line_bytes
    if current_lines:
        chunks.append({"lines": current_lines, "refs": current_refs, "size_bytes": current_bytes})

    batch_ids = []
    submitted_refs = []
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
            submitted_refs.extend(chunk["refs"])
            print(f"Submitted chunk {i}/{len(chunks)} -> batch_id={batch_id}, tasks={len(chunk['refs'])}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    BATCH_IDS_FILE.write_text(json.dumps(batch_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    BATCH_MANIFEST_FILE.write_text(
        json.dumps(
            {
                "created_at": int(time.time()),
                "model": cfg.get("LLM_MODEL", "gpt-4o-mini"),
                "timeline_name": timeline_name or "",
                "top_k": int(top_k),
                "batch_ids": batch_ids,
                "task_refs": submitted_refs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return batch_ids


def _wait_batch_completed(cfg, batch_id, poll_interval_sec):
    while True:
        batch = _http_get_json(cfg, f"/batches/{batch_id}")
        status = str(batch.get("status", ""))
        if status == "completed":
            return batch
        if status in ("failed", "cancelled", "canceled", "expired"):
            return batch
        print(f"Batch {batch_id} status={status}, attendo {poll_interval_sec}s...")
        time.sleep(poll_interval_sec)


def collect_batch_results(topk, cfg, min_link_confidence, max_links_per_task, poll_interval_sec):
    if not BATCH_MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest batch non trovato: {BATCH_MANIFEST_FILE}")
    manifest = json.loads(BATCH_MANIFEST_FILE.read_text(encoding="utf-8"))
    batch_ids = manifest.get("batch_ids") or []
    refs = manifest.get("task_refs") or []
    ref_by_id = {str(x["custom_id"]): x for x in refs}
    group_by_key = {
        (str(timeline), int(task_row_id)): group
        for (timeline, task_row_id), group in topk.groupby(["TimelineName", "TaskRowId"], sort=True)
    }
    group_results = []
    seen_custom_ids = set()

    for batch_id in batch_ids:
        batch = _wait_batch_completed(cfg, str(batch_id), poll_interval_sec)
        status = str(batch.get("status", ""))
        if status != "completed":
            print(f"Batch {batch_id} non completato: status={status}")
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
            ref = ref_by_id.get(custom_id)
            if not ref:
                continue
            seen_custom_ids.add(custom_id)
            key = (str(ref["timeline_name"]), int(ref["task_row_id"]))
            group = group_by_key.get(key)
            if group is None:
                continue
            text = _extract_batch_text(row)
            try:
                parsed = _parse_json_text(text)
                resolved = validate_resolver_output(parsed, group)
            except Exception as exc:
                resolved = _invalid_result(type(exc).__name__, str(exc)[:500])
            group_results.append(
                build_final_rows_for_group(
                    key[0],
                    key[1],
                    group,
                    resolved,
                    min_link_confidence=min_link_confidence,
                    max_links_per_task=max_links_per_task,
                )
            )

    missing = [x for x in refs if str(x["custom_id"]) not in seen_custom_ids]
    for ref in missing:
        key = (str(ref["timeline_name"]), int(ref["task_row_id"]))
        group = group_by_key.get(key)
        if group is None:
            continue
        resolved = {
            "status": "llm_error",
            "links": [],
            "error_type": "missing_batch_result",
            "error_message": "No output row found for this batch custom_id",
            "raw_links_count": 0,
            "valid_links_count": 0,
            "dropped_invalid_count": 0,
            "duplicate_candidate_count": 0,
        }
        group_results.append(
            build_final_rows_for_group(
                key[0],
                key[1],
                group,
                resolved,
                min_link_confidence=min_link_confidence,
                max_links_per_task=max_links_per_task,
            )
        )

    return combine_group_results(group_results)


def build_resolved_task_scope(topk):
    if topk.empty:
        return pd.DataFrame(columns=["TimelineName", "TaskRowId"])
    return topk[["TimelineName", "TaskRowId"]].drop_duplicates().copy()


def save_resolver_diagnostics(rows):
    if rows.empty:
        return ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"resolver_diagnostics_{timestamp}.csv"
    rows.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)


def save_final_links(conn, db_name, rows, resolved_scope):
    if resolved_scope.empty:
        return 0
    conn.register("resolved_scope", resolved_scope)
    try:
        conn.execute("BEGIN;")
        conn.execute(
            f"""
            DELETE FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrLinks t
            USING resolved_scope s
            WHERE t.TimelineName = s.TimelineName
              AND t.TaskRowId = s.TaskRowId
            """
        )
        inserted = 0
        if not rows.empty:
            conn.register("final_links", rows)
            try:
                conn.execute(
                    f"""
                    INSERT INTO {db_name}.timeline_reconciliation.TimelineTaskToMdrLinks (
                        TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                        TaskStartDate, TaskFinishDate, TaskActualStartDate, TaskActualFinishDate,
                        TaskDateFieldsJson,
                        TaskClass, TaskClassConfidence, TaskClassReason,
                        MdrDocumentTitle, MdrTitleKey, LinkRank, LinkScore, LinkMethod, LinkReason,
                        ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                        ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource, CreatedBy
                    )
                    SELECT
                        TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                        TaskStartDate, TaskFinishDate, TaskActualStartDate, TaskActualFinishDate,
                        TaskDateFieldsJson,
                        TaskClass, TaskClassConfidence, TaskClassReason,
                        MdrDocumentTitle, MdrTitleKey, LinkRank, LinkScore, LinkMethod, LinkReason,
                        ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                        ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource, CreatedBy
                    FROM final_links
                    """
                )
                inserted = len(rows)
            finally:
                conn.unregister("final_links")
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.unregister("resolved_scope")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="4 LLM resolver final timeline task -> MDR links")
    parser.add_argument("--timeline", default="", help="Processa una sola TimelineName.")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="Worker paralleli per modalita realtime.")
    parser.add_argument("--min-link-confidence", type=float, default=0.75, help="Scarta link con confidenza inferiore.")
    parser.add_argument("--max-links-per-task", type=int, default=3, help="0 = nessun limite, altrimenti massimo link per task.")
    parser.add_argument("--llm-timeout-sec", type=int, default=60)
    parser.add_argument("--retry-max", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--batch-submit", action="store_true", help="Invia i task alla Batch API e termina.")
    parser.add_argument("--batch-collect", action="store_true", help="Colleziona gli output batch e salva DB.")
    parser.add_argument("--batch-and-collect", action="store_true", help="Submit batch e attende/colleziona nello stesso run.")
    parser.add_argument("--batch-max-bytes", type=int, default=DEFAULT_BATCH_TARGET_BYTES)
    parser.add_argument("--batch-poll-interval", type=int, default=DEFAULT_BATCH_POLL_INTERVAL)
    args = parser.parse_args()
    selected_batch_modes = int(args.batch_submit) + int(args.batch_collect) + int(args.batch_and_collect)
    if selected_batch_modes > 1:
        raise RuntimeError("Usa una sola modalita batch: --batch-submit, --batch-collect oppure --batch-and-collect")

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    embedding_model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    timeline_name = args.timeline or None
    top_k = args.top_k
    if args.batch_collect and BATCH_MANIFEST_FILE.exists():
        manifest = json.loads(BATCH_MANIFEST_FILE.read_text(encoding="utf-8"))
        timeline_name = manifest.get("timeline_name") or timeline_name
        top_k = int(manifest.get("top_k") or top_k)

    conn = connect_motherduck(cfg)
    try:
        topk = load_topk_for_resolver(conn, db_name, embedding_model, timeline_name=timeline_name, top_k=top_k)
        print(f"Top-K rows for resolver: {len(topk)}")
        resolved_scope = build_resolved_task_scope(topk)
        print(f"Task scope (all): {len(resolved_scope)}")

        if args.batch_submit or args.batch_and_collect:
            batch_ids = run_batch_submit(
                topk,
                cfg,
                args.batch_max_bytes,
                timeline_name=timeline_name,
                top_k=top_k,
            )
            print(f"Batch submitted: {len(batch_ids)} batch(es)")
            if args.batch_submit:
                return

        if args.batch_collect or args.batch_and_collect:
            final_links, diagnostics, resolved_scope_all, resolved_scope_ok, status_counts = collect_batch_results(
                topk,
                cfg,
                args.min_link_confidence,
                args.max_links_per_task,
                args.batch_poll_interval,
            )
        else:
            final_links, diagnostics, resolved_scope_all, resolved_scope_ok, status_counts = build_final_links(
                topk,
                cfg,
                args.progress_every,
                min_link_confidence=args.min_link_confidence,
                max_links_per_task=args.max_links_per_task,
                llm_timeout_sec=args.llm_timeout_sec,
                retry_max=args.retry_max,
                retry_backoff_sec=args.retry_backoff_sec,
                workers=args.workers,
            )
        print(
            "Resolver status counts: "
            f"ok={status_counts.get('ok', 0)}, "
            f"llm_error={status_counts.get('llm_error', 0)}, "
            f"invalid_json={status_counts.get('invalid_json', 0)}"
        )
        print(f"Task scope resolved (all): {len(resolved_scope_all)}")
        print(f"Task scope resolved (ok): {len(resolved_scope_ok)}")
        print(f"Final links created: {len(final_links)}")
        diagnostics_path = save_resolver_diagnostics(diagnostics)
        print(f"Resolver diagnostics saved: {diagnostics_path or 0}")
        print(
            "Final links saved: "
            f"{save_final_links(conn, db_name, final_links, resolved_scope_ok.drop_duplicates())}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
