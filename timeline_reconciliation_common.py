import hashlib
import json
import re
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.txt"
CRONOPROGRAMMI_DIR = BASE_DIR / "cronoprogrammi"
OUTPUT_DIR = BASE_DIR / "output"
TASK_SHEET = "TASK"
CREATED_BY = "timeline_reconciliation_pipeline"


def parse_config_txt(path=CONFIG_FILE):
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


def raci_dedupe_key(
    consolidated_title_key=None,
    mdr_title_key=None,
    consolidated_raci_title=None,
    mdr_document_title=None,
):
    """Chiave stabile per un solo documento RACI logico (dedupe Top-K / shortlist / link)."""
    for raw in (consolidated_title_key, mdr_title_key):
        s = ("" if raw is None else str(raw)).strip().lower()
        if s:
            return s
    for raw in (consolidated_raci_title, mdr_document_title):
        s = normalize(raw)
        if s:
            return s
    return ""


def remove_prefix(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    parts = text.split("-", 1)
    if len(parts) == 2 and len(parts[0]) <= 4:
        return parts[1].strip()
    return text


def extract_project_code(timeline_name):
    match = re.search(r"\d{4,}", str(timeline_name))
    if match:
        return match.group(0)
    return str(timeline_name)


# Date persistite in DB: input al COALESCE actualized (Actual → Early → Target)
PRIMERA_CORE_DATE_FIELDS = (
    ("early_start_date", "EarlyStartDate"),
    ("early_end_date", "EarlyEndDate"),
    ("actual_start_date", "ActualStartDate"),
    ("actual_end_date", "ActualEndDate"),
    ("target_start_date", "TargetStartDate"),
    ("target_end_date", "TargetEndDate"),
)

# Alias retrocompatibile per script che importano PRIMERA_ALL_SCHEDULE_FIELDS
PRIMERA_ALL_SCHEDULE_FIELDS = PRIMERA_CORE_DATE_FIELDS

PRIMERA_CORE_JSON_KEYS = {snake: snake for snake, _ in PRIMERA_CORE_DATE_FIELDS}
PRIMERA_CORE_JSON_KEYS.update(
    {
        "act_start_date": "actual_start_date",
        "act_end_date": "actual_end_date",
    }
)

PRIMERA_CORE_DATE_ALIASES = {
    "early_start_date": ("early_start_date", "early_start", "ag"),
    "early_end_date": ("early_end_date", "early_finish", "early_end", "ah"),
    "actual_start_date": ("actual_start_date", "actual_start", "act_start_date", "ab"),
    "actual_end_date": ("actual_end_date", "actual_finish", "act_end_date", "ac"),
    "target_start_date": ("target_start_date", "target_start"),
    "target_end_date": ("target_end_date", "target_end", "target_finish"),
}

SELECTED_DATE_SCENARIO = "actualized"


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def serialize_date_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def first_matching_date(row, aliases):
    normalized_columns = {normalize_column_name(c): c for c in row.index}
    for alias in aliases:
        original_col = normalized_columns.get(alias)
        if original_col is None:
            continue
        value = row.get(original_col)
        if not pd.isna(value):
            return value
    return None


def first_matching_text(row, aliases):
    normalized_columns = {normalize_column_name(c): c for c in row.index}
    for alias in aliases:
        original_col = normalized_columns.get(alias)
        if original_col is None:
            continue
        value = row.get(original_col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _coalesce_schedule(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and not str(value).strip():
            continue
        return value
    return None


def _parse_date_from_json_value(value):
    if value is None or value == "":
        return None
    if hasattr(value, "isoformat"):
        return value
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def extract_primavera_schedule_fields_from_row(row):
    """Extract core Primavera start/finish fields from a TASK row (Excel columns and/or JSON keys)."""
    out = {snake: None for snake, _ in PRIMERA_CORE_DATE_FIELDS}
    if row is None:
        return out
    for snake, aliases in PRIMERA_CORE_DATE_ALIASES.items():
        val = first_matching_date(row, aliases)
        if val is not None and not pd.isna(val):
            out[snake] = val
    json_raw = row.get("task_date_fields_json") if hasattr(row, "get") else None
    if json_raw:
        try:
            payload = json.loads(json_raw) if isinstance(json_raw, str) else {}
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                norm_key = normalize_column_name(key)
                target = PRIMERA_CORE_JSON_KEYS.get(norm_key)
                if not target or out.get(target) is not None:
                    continue
                parsed = _parse_date_from_json_value(value)
                if parsed is not None and not pd.isna(parsed):
                    out[target] = parsed
    return out


def extract_primavera_raw_dates_from_row(row):
    """Backward-compatible alias: date keys only from schedule extraction."""
    return extract_primavera_schedule_fields_from_row(row)


def compute_selected_schedule(schedule, scenario=None):
    """Actualized start/finish: Actual → Early → Target."""
    early_start = schedule.get("early_start_date")
    early_end = schedule.get("early_end_date")
    actual_start = schedule.get("actual_start_date")
    actual_end = schedule.get("actual_end_date")
    target_start = schedule.get("target_start_date")
    target_end = schedule.get("target_end_date")
    start = _coalesce_schedule(actual_start, early_start, target_start)
    finish = _coalesce_schedule(actual_end, early_end, target_end)
    return {
        "scenario": SELECTED_DATE_SCENARIO,
        "start": None if start is None or (isinstance(start, float) and pd.isna(start)) else start,
        "finish": None if finish is None or (isinstance(finish, float) and pd.isna(finish)) else finish,
    }


def raw_schedule_snake_from_row(row):
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    out = {}
    for snake, pascal in PRIMERA_ALL_SCHEDULE_FIELDS:
        val = row.get(snake)
        if val is None:
            val = row.get(pascal)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            out[snake] = None
        else:
            out[snake] = val
    return out


def raw_dates_snake_from_row(row):
    """Backward-compatible alias."""
    return raw_schedule_snake_from_row(row)


def build_link_date_snapshot(row, scenario=None):
    """PascalCase core schedule fields + Selected* actualized for TimelineTaskToMdrLinks."""
    schedule = raw_schedule_snake_from_row(row)
    selected = compute_selected_schedule(schedule, scenario)
    payload = {
        "SelectedStartDate": selected["start"],
        "SelectedFinishDate": selected["finish"],
    }
    for snake, pascal in PRIMERA_CORE_DATE_FIELDS:
        payload[pascal] = schedule.get(snake)
    return payload


def classified_row_dates_for_ref(row):
    """Snake_case core schedule fields for batch manifest from a classified row."""
    schedule = extract_primavera_schedule_fields_from_row(row)
    out = {"task_date_fields_json": row.get("task_date_fields_json")}
    for snake, _ in PRIMERA_CORE_DATE_FIELDS:
        out[snake] = serialize_date_value(schedule.get(snake))
    return out


def build_task_date_fields_json(row):
    out = {}
    for col in row.index:
        col_norm = normalize_column_name(col)
        value = row.get(col)
        if pd.isna(value):
            continue
        is_date_like_value = hasattr(value, "isoformat")
        is_date_like_name = any(x in col_norm for x in ("date", "start", "finish", "cstr"))
        if is_date_like_value or is_date_like_name:
            out[str(col)] = serialize_date_value(value)
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def add_task_date_columns(task):
    out = task.copy()
    out["task_date_fields_json"] = out.apply(build_task_date_fields_json, axis=1)

    def _apply_dates(r):
        schedule = extract_primavera_schedule_fields_from_row(r)
        payload = {"task_date_fields_json": r.get("task_date_fields_json")}
        for snake, _ in PRIMERA_CORE_DATE_FIELDS:
            payload[snake] = schedule.get(snake)
        return pd.Series(payload)

    date_cols = out.apply(_apply_dates, axis=1)
    for col in date_cols.columns:
        out[col] = date_cols[col]
    return out


def text_hash(text):
    normalized = " ".join(str(text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def float32_to_blob(values):
    arr = np.asarray(values, dtype=np.float32)
    return arr.tobytes()


def blob_to_float32(blob):
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if blob is None:
        return np.array([], dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def cosine_from_blobs(left_blob, right_blob):
    left = blob_to_float32(left_blob)
    right = blob_to_float32(right_blob)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    return float(np.dot(left, right))


def refresh_timeline_classified_dates_view(conn, db_name):
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW {db_name}.timeline_reconciliation.v_TimelineTasksClassified_Dates AS
        SELECT
            c.*,
            COALESCE(c.ActualStartDate, c.EarlyStartDate, c.TargetStartDate) AS StartActualized,
            COALESCE(c.ActualEndDate, c.EarlyEndDate, c.TargetEndDate) AS FinishActualized
        FROM {db_name}.timeline_reconciliation.TimelineTasksClassified AS c
        """
    )


def refresh_timeline_links_dates_view(conn, db_name):
    """Recreate v_TimelineTaskToMdrLinks_Dates after schema changes."""
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW {db_name}.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates AS
        SELECT
            l.*,
            COALESCE(l.SelectedStartDate, COALESCE(l.ActualStartDate, l.EarlyStartDate, l.TargetStartDate)) AS StartActualized,
            COALESCE(l.SelectedFinishDate, COALESCE(l.ActualEndDate, l.EarlyEndDate, l.TargetEndDate)) AS FinishActualized
        FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrLinks AS l
        """
    )


def connect_motherduck(cfg):
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    token = cfg.get("MOTHERDUCK_TOKEN", "").strip()
    if not token:
        raise ValueError("MOTHERDUCK_TOKEN mancante in config.txt")
    return duckdb.connect(f"md:{db_name}?motherduck_token={token}")


def chat_json(cfg, system, user, timeout=60):
    api_key = cfg.get("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_API_KEY mancante in config.txt")
    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    base_url = cfg.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    content = json.loads(raw)["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def embed_text(cfg, text, timeout=60):
    api_key = cfg.get("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_API_KEY mancante in config.txt")
    model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = cfg.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {"model": model, "input": str(text or "")}
    req = urllib.request.Request(
        f"{base_url}/embeddings",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    values = json.loads(raw)["data"][0]["embedding"]
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def embed_texts(cfg, texts, batch_size=256, timeout=60):
    api_key = cfg.get("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_API_KEY mancante in config.txt")
    model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = cfg.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    vectors = []
    for i in range(0, len(texts), batch_size):
        chunk = [str(x or "") for x in texts[i : i + batch_size]]
        body = {"model": model, "input": chunk}
        req = urllib.request.Request(
            f"{base_url}/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)["data"]
        for item in data:
            arr = np.asarray(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vectors.append(arr)
    return vectors


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
        task["task_row_id"] = task.index
        return add_task_date_columns(task)
    try:
        wbs = pd.read_excel(prim_file, sheet_name="PROJWBS", usecols=["wbs_id", "wbs_name", "wbs_short_name"])
        task = task.merge(wbs, on="wbs_id", how="left")
    except Exception:
        task["wbs_name"] = ""
    task["task_row_id"] = task.index
    return add_task_date_columns(task)


def build_task_text(task_name, wbs_name, task_class="ENG_DOC"):
    return "\n".join(
        [
            f"Task name: {remove_prefix(task_name)}",
            f"WBS: {wbs_name or ''}",
            f"Task class: {task_class}",
        ]
    )


def build_mdr_candidate_text(row):
    parts = [
        f"MDR title: {row.get('MdrDocumentTitle', '')}",
        f"RACI title: {row.get('ConsolidatedRaciTitle', '')}",
        f"RACI description: {row.get('EffectiveDescription', '')}",
        f"Discipline: {row.get('DisciplineName', '')}",
        f"Document type: {row.get('TypeName', '')}",
        f"Category: {row.get('CategoryDescription', '')}",
        f"Chapter: {row.get('ChapterName', '')}",
    ]
    return "\n".join(parts)
