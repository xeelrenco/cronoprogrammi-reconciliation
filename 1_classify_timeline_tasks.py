import argparse
import math
import time

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
)


DEFAULT_SAMPLE_SEED = 42
DEFAULT_PROGRESS_EVERY = 25
CREATED_BY = "1_classify_timeline_tasks.py"
PROMPT_VERSION = "timeline_task_classification_v1"


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
  "reason_short": "brief reason in Italian"
}
"""
    user = {
        "task_name": task_name,
        "task_name_clean": remove_prefix(task_name),
        "wbs_name": wbs_name,
    }
    fallback = {"task_class": "OTHER", "confidence": "LOW", "reason_short": "LLM unavailable or invalid response"}
    try:
        parsed = chat_json(cfg, system, user, timeout=45)
    except Exception:
        return fallback
    label = str(parsed.get("task_class", "OTHER")).upper().strip()
    if label not in ("ENG_DOC", "OTHER"):
        return fallback
    confidence = str(parsed.get("confidence", "LOW")).upper().strip()
    if confidence not in ("HIGH", "MEDIUM", "LOW"):
        confidence = "LOW"
    return {"task_class": label, "confidence": confidence, "reason_short": str(parsed.get("reason_short", ""))[:300]}


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
                TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName, TaskText,
                TaskClass, TaskClassConfidence, TaskClassReason, ClassifierModel,
                ClassifierPromptVersion, CreatedBy
            )
            SELECT
                TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName, TaskText,
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
    args = parser.parse_args()

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    save_to_db = args.save_db or args.limit == 0
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
                    ["task_code", "task_name", "wbs_name", "task_class", "classification_confidence", "classification_reason"]
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
