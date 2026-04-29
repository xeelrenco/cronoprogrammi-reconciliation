import hashlib
import json
import re
import struct
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
    return np.asarray(struct.unpack(f"{len(blob) // 4}f", blob), dtype=np.float32)


def cosine_from_blobs(left_blob, right_blob):
    left = blob_to_float32(left_blob)
    right = blob_to_float32(right_blob)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    return float(np.dot(left, right))


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
        return task
    try:
        wbs = pd.read_excel(prim_file, sheet_name="PROJWBS", usecols=["wbs_id", "wbs_name", "wbs_short_name"])
        task = task.merge(wbs, on="wbs_id", how="left")
    except Exception:
        task["wbs_name"] = ""
    task["task_row_id"] = task.index
    return task


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
