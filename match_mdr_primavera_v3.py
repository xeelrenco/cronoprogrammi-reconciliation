import json
import re
import urllib.request
from pathlib import Path
import math
import time
import argparse

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.txt"
CRONOPROGRAMMI_DIR = BASE_DIR / "cronoprogrammi"
OUTPUT_DIR = BASE_DIR / "output_v3"
TASK_SHEET = "TASK"
DEFAULT_SAMPLE_FILE = BASE_DIR / "v3_task_sample.csv"
DEFAULT_SAMPLE_SEED = 42
DEFAULT_PROGRESS_EVERY = 25


def parse_config_txt(path):
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_prefix(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    parts = text.split("-", 1)
    if len(parts) == 2 and len(parts[0]) <= 4:
        return parts[1].strip()
    return text


def normalize_task_columns(task):
    rename_map = {}
    for c in task.columns:
        c_norm = str(c).strip().lower().replace(" ", "_")
        if c_norm in ("task_name", "task_code") and c != c_norm:
            rename_map[c] = c_norm
    if rename_map:
        task = task.rename(columns=rename_map)
    return task


def load_task_with_wbs(prim_file):
    task = pd.read_excel(prim_file, sheet_name=TASK_SHEET)
    task = normalize_task_columns(task)
    if "task_name" not in task.columns:
        raise ValueError("foglio TASK senza colonna 'task_name'")
    if "wbs_id" not in task.columns:
        task["wbs_name"] = ""
        return task
    try:
        wbs = pd.read_excel(prim_file, sheet_name="PROJWBS", usecols=["wbs_id", "wbs_name", "wbs_short_name"])
        task = task.merge(wbs, on="wbs_id", how="left")
    except Exception:
        task["wbs_name"] = ""
    task["task_row_id"] = task.index
    return task


def _compute_proportional_quotas(counts, total_limit):
    """
    Quota proporzionale per file, con eventuale minima copertura 1/file quando possibile.
    """
    non_empty = {k: v for k, v in counts.items() if v > 0}
    if not non_empty:
        return {k: 0 for k in counts}

    total_rows = sum(non_empty.values())
    raw = {k: (total_limit * v / total_rows) for k, v in non_empty.items()}
    quotas = {k: int(math.floor(x)) for k, x in raw.items()}

    # Se il limite lo consente, garantisce almeno 1 task per file non vuoto.
    if total_limit >= len(non_empty):
        for k in non_empty:
            if quotas[k] == 0:
                quotas[k] = 1

    assigned = sum(quotas.values())
    remaining = max(0, total_limit - assigned)

    if remaining > 0:
        remainders = sorted(
            ((k, raw[k] - math.floor(raw[k])) for k in non_empty),
            key=lambda x: x[1],
            reverse=True,
        )
        i = 0
        while remaining > 0 and remainders:
            k = remainders[i % len(remainders)][0]
            if quotas[k] < non_empty[k]:
                quotas[k] += 1
                remaining -= 1
            i += 1
            if i > 10 * len(remainders) and remaining > 0:
                # fallback: distribuisce ai file che hanno ancora spazio
                candidates = [kk for kk, vv in non_empty.items() if quotas[kk] < vv]
                if not candidates:
                    break
                for kk in candidates:
                    if remaining <= 0:
                        break
                    quotas[kk] += 1
                    remaining -= 1

    # Non supera il numero disponibile per file.
    for k, v in non_empty.items():
        quotas[k] = min(quotas[k], v)

    out = {k: 0 for k in counts}
    out.update(quotas)
    return out


def build_or_load_sample_map(tasks_by_file, sample_limit, sample_file, random_seed=42):
    """
    Se sample_file esiste, riusa task_row_id già campionati.
    Altrimenti crea campione random proporzionale e lo salva.
    """
    if sample_file.exists():
        sample_df = pd.read_csv(sample_file, dtype={"cronoprogramma": str, "task_row_id": int})
        file_names = set(tasks_by_file.keys())
        valid_existing = True
        # Deve avere lo stesso numero totale richiesto.
        if len(sample_df) != sample_limit:
            valid_existing = False
        # Deve contenere solo file attuali.
        if not set(sample_df["cronoprogramma"].unique().tolist()).issubset(file_names):
            valid_existing = False
        # Deve contenere almeno una riga ricostruibile dai file correnti.
        if valid_existing:
            reconstructed_total = 0
            for name, task in tasks_by_file.items():
                ids = set(sample_df[sample_df["cronoprogramma"] == name]["task_row_id"].tolist())
                if ids:
                    reconstructed_total += int(task["task_row_id"].isin(ids).sum())
            if reconstructed_total != len(sample_df):
                valid_existing = False

        if valid_existing:
            sample_map = {}
            for name, task in tasks_by_file.items():
                ids = set(sample_df[sample_df["cronoprogramma"] == name]["task_row_id"].tolist())
                if ids:
                    sample_map[name] = task[task["task_row_id"].isin(ids)].copy()
                else:
                    sample_map[name] = task.iloc[0:0].copy()
            return sample_map, sample_df, False

    counts = {name: len(df) for name, df in tasks_by_file.items()}
    quotas = _compute_proportional_quotas(counts, sample_limit)
    rows = []
    sample_map = {}
    for i, (name, task) in enumerate(tasks_by_file.items()):
        q = quotas.get(name, 0)
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


def llm_classify_task(task_name, wbs_name, cfg):
    fallback = {
        "task_class": "OTHER",
        "confidence": "LOW",
        "reason_short": "LLM unavailable or invalid response",
    }
    api_key = cfg.get("LLM_API_KEY", "")
    if not api_key:
        fallback["reason_short"] = "LLM_API_KEY missing"
        return fallback
    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    base_url = cfg.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    system = """
You classify Primavera P6 schedule tasks.

Return ONLY valid JSON.

Classify each task into exactly one class:

ENG_DOC:
The task represents progress, issue, review, approval, revision, delivery, or update
of engineering documents/deliverables that can be linked to MDR/RACI documents.
Examples: issue for review, issue for construction, as-built, document approval,
engineering drawings, datasheets, calculations, specifications, document list,
reports, procedures, layouts, P&ID, vendor document review when the focus is document progress.

OTHER:
The task is not direct MDR/RACI document progress.
This includes procurement/material phases, RFQ, technical alignment, commercial alignment,
issue of order, purchase order, vendor follow-up, logistics, construction, installation,
testing, commissioning, meetings, milestones, site activities, manufacturing, delivery,
and generic project activities.

Important:
- If the activity refers to procurement/material purchasing phases, classify OTHER,
  even if datasheets/specifications/IDS are mentioned.
- If the task refers to a component/item/chapter rather than document progress, classify OTHER.
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
        "expected_json": {
            "task_class": "ENG_DOC|OTHER",
            "confidence": "HIGH|MEDIUM|LOW",
            "reason_short": "string",
        },
    }
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
        content = json.loads(raw)["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        parsed = json.loads(content)
        label = str(parsed.get("task_class", "OTHER")).upper().strip()
        if label not in ("ENG_DOC", "OTHER"):
            return fallback
        conf = str(parsed.get("confidence", "LOW")).upper().strip()
        if conf not in ("HIGH", "MEDIUM", "LOW"):
            conf = "LOW"
        return {
            "task_class": label,
            "confidence": conf,
            "reason_short": str(parsed.get("reason_short", ""))[:300],
        }
    except Exception:
        return fallback


def classify_tasks_with_llm(task, cfg, file_label="", progress_every=25):
    out = task.copy()
    out["name_clean"] = out["task_name"].apply(remove_prefix)
    out["name_norm"] = out["name_clean"].apply(normalize)
    results = []
    total = len(out)
    started = time.time()
    for idx, (_, row) in enumerate(out.iterrows(), 1):
        results.append(llm_classify_task(str(row.get("task_name", "")), str(row.get("wbs_name", "")), cfg))
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            elapsed = round(time.time() - started, 1)
            print(f"[{file_label}] classified {idx}/{total} tasks (elapsed {elapsed}s)")
    out["task_class"] = [x["task_class"] for x in results]
    out["classification_confidence"] = [x["confidence"] for x in results]
    out["classification_reason"] = [x["reason_short"] for x in results]
    return out


def stats_with_pct(series, label):
    out = series.value_counts().rename_axis(label).reset_index(name="count")
    total = int(out["count"].sum())
    out["pct"] = (out["count"] / total * 100).round(2) if total > 0 else 0
    return out


def main():
    parser = argparse.ArgumentParser(description="LLM-only task classification (v3)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Totale task da classificare (campione proporzionale). 0 = nessun limite.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Log avanzamento ogni N task classificati.",
    )
    args = parser.parse_args()

    started_all = time.time()
    cfg = parse_config_txt(CONFIG_FILE)
    if not CRONOPROGRAMMI_DIR.exists():
        raise FileNotFoundError(f"Cartella cronoprogrammi non trovata: {CRONOPROGRAMMI_DIR}")
    if not cfg.get("LLM_API_KEY", "").strip():
        raise ValueError("LLM_API_KEY mancante in config.txt")
    progress_every = int(args.progress_every)

    files = sorted(CRONOPROGRAMMI_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Nessun file .xlsx in {CRONOPROGRAMMI_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Start v3 classification - files found: {len(files)}")
    print(f"Progress log every {progress_every} tasks")

    sample_limit = int(args.limit)
    sample_seed = DEFAULT_SAMPLE_SEED
    sample_file = DEFAULT_SAMPLE_FILE

    # Preload task per file (serve per campionamento proporzionale).
    tasks_by_file = {}
    for prim in files:
        ref_stem = prim.stem
        try:
            tasks_by_file[ref_stem] = load_task_with_wbs(prim)
        except Exception as exc:
            print(f"Skip {prim.name}: {exc}")
            tasks_by_file[ref_stem] = None

    sample_map = None
    sample_metadata = None
    sample_created = False
    if sample_limit > 0:
        valid_tasks = {k: v for k, v in tasks_by_file.items() if v is not None and len(v) > 0}
        sample_map, sample_metadata, sample_created = build_or_load_sample_map(
            valid_tasks, sample_limit, sample_file, random_seed=sample_seed
        )
        if sample_created:
            print(f"Creato campione test: {sample_file}")
        else:
            print(f"Riutilizzato campione test: {sample_file}")
        print(f"Sample limit totale richiesto: {sample_limit}")
        print(f"Task nel campione effettivo: {len(sample_metadata)}")
        if sample_metadata is not None and len(sample_metadata) > 0:
            by_file = sample_metadata["cronoprogramma"].value_counts()
            print("Distribuzione campione (task per file):")
            for name, n in by_file.items():
                print(f" - {name}: {n}")

    all_task_class = []
    all_task_rows = []
    completed_files = 0
    for prim in files:
        file_started = time.time()
        ref_stem = prim.stem
        try:
            task = tasks_by_file.get(ref_stem)
            if task is None:
                continue
            if sample_limit > 0:
                task = sample_map.get(ref_stem, task.iloc[0:0].copy())
            if task.empty:
                print(f"Skip {prim.name}: nessuna task selezionata nel campione")
                continue
            print(f"Processing {prim.name} - tasks to classify: {len(task)}")
            task = classify_tasks_with_llm(task, cfg, file_label=ref_stem, progress_every=progress_every)
        except Exception as exc:
            print(f"Skip {prim.name}: {exc}")
            continue
        task["cronoprogramma"] = ref_stem

        out_path = OUTPUT_DIR / f"classification_{ref_stem}.xlsx"
        summary = pd.DataFrame(
            [
                {"metric": "task_rows", "value": len(task)},
                {"metric": "eng_doc_task_rows", "value": int((task["task_class"] == "ENG_DOC").sum())},
                {"metric": "other_task_rows", "value": int((task["task_class"] == "OTHER").sum())},
            ]
        )
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            task[
                [
                    "task_code",
                    "task_name",
                    "wbs_name",
                    "task_class",
                    "classification_confidence",
                    "classification_reason",
                ]
            ].to_excel(writer, sheet_name="Task_LLM_classification", index=False)
            stats_with_pct(task["task_class"], "task_class").to_excel(writer, sheet_name="Task_class_stats", index=False)
        print(f"Creato: {out_path}")
        eng_doc_count = int((task["task_class"] == "ENG_DOC").sum())
        other_count = int((task["task_class"] == "OTHER").sum())
        print(
            f"[{ref_stem}] ENG_DOC={eng_doc_count}, OTHER={other_count}, "
            f"file_elapsed={round(time.time() - file_started, 1)}s"
        )
        all_task_class.append(task[["task_class"]].copy())
        all_task_rows.append(task[["cronoprogramma", "task_class"]].copy())
        completed_files += 1

    if all_task_class:
        agg = OUTPUT_DIR / "classification_aggregate_v3.xlsx"
        with pd.ExcelWriter(agg, engine="openpyxl") as writer:
            stats_with_pct(pd.concat(all_task_class, ignore_index=True)["task_class"], "task_class").to_excel(
                writer, sheet_name="Task_class", index=False
            )
            if sample_metadata is not None and len(sample_metadata) > 0:
                sample_dist = sample_metadata["cronoprogramma"].value_counts().rename_axis("cronoprogramma").reset_index(name="count")
                if all_task_rows:
                    classified = pd.concat(all_task_rows, ignore_index=True)
                    eng_counts = (
                        classified[classified["task_class"] == "ENG_DOC"]["cronoprogramma"]
                        .value_counts()
                        .rename_axis("cronoprogramma")
                        .reset_index(name="eng_doc_count")
                    )
                    other_counts = (
                        classified[classified["task_class"] == "OTHER"]["cronoprogramma"]
                        .value_counts()
                        .rename_axis("cronoprogramma")
                        .reset_index(name="other_count")
                    )
                    sample_dist = sample_dist.merge(eng_counts, on="cronoprogramma", how="left").merge(
                        other_counts, on="cronoprogramma", how="left"
                    )
                    sample_dist["eng_doc_count"] = sample_dist["eng_doc_count"].fillna(0).astype(int)
                    sample_dist["other_count"] = sample_dist["other_count"].fillna(0).astype(int)
                else:
                    sample_dist["eng_doc_count"] = 0
                    sample_dist["other_count"] = 0
                sample_dist.to_excel(writer, sheet_name="Sample_distribution", index=False)
        print(f"Creato: {agg}")
    print(f"Completed files: {completed_files}/{len(files)}")
    print(f"Fatto. Elapsed totale: {round(time.time() - started_all, 1)}s")


if __name__ == "__main__":
    main()
