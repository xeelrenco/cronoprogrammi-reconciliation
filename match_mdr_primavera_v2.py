import re
from pathlib import Path
import json
import urllib.request

import pandas as pd
import duckdb

# ===== CONFIG =====
CRONOPROGRAMMI_DIR = Path(__file__).resolve().parent / "cronoprogrammi"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_v2"

MDR_REF_COL = "MDR Ref"
DOC_NUMBER_COL = "Doc. Number"
DOC_TITLE_COL = "Doc. Title"
MDR_CODE_NAME_REF_COL = "Mdr_code_name_ref"
TASK_SHEET = "TASK"
CONFIG_FILE = Path(__file__).resolve().parent / "config.txt"

# ===== LLM JUDGE (OPTIONAL) =====
DEFAULT_RUNTIME_CONFIG = {
    "mdr_source": {
        "motherduck_db": "my_db",
        "motherduck_token": ""
    },
    "matching": {
        "allow_multiple_tasks_per_doc": True,
        "max_tasks_per_doc": 0,
        "min_score_for_additional_tasks": 70
    },
    "llm_judge": {
        "enabled": False,
        "provider": "openai",
        "api_key": "",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "top_k_candidates": 5,
        "good_score_gap_threshold": 5,
    }
}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_config_txt(path):
    """
    Legge un config.txt stile KEY=VALUE.
    Ignora righe vuote e commenti che iniziano con #.
    """
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_runtime_config(path=CONFIG_FILE):
    cfg = json.loads(json.dumps(DEFAULT_RUNTIME_CONFIG))
    kv = parse_config_txt(path)
    if not kv:
        return cfg
    try:
        cfg["mdr_source"]["motherduck_db"] = str(kv.get("MOTHERDUCK_DB", cfg["mdr_source"]["motherduck_db"])).strip()
        cfg["mdr_source"]["motherduck_token"] = str(kv.get("MOTHERDUCK_TOKEN", cfg["mdr_source"]["motherduck_token"])).strip()
        cfg["matching"]["allow_multiple_tasks_per_doc"] = _to_bool(
            kv.get("MATCH_ALLOW_MULTIPLE_TASKS_PER_DOC", cfg["matching"]["allow_multiple_tasks_per_doc"])
        )
        cfg["matching"]["max_tasks_per_doc"] = int(
            kv.get("MATCH_MAX_TASKS_PER_DOC", cfg["matching"]["max_tasks_per_doc"])
        )
        cfg["matching"]["min_score_for_additional_tasks"] = int(
            kv.get("MATCH_MIN_SCORE_FOR_ADDITIONAL_TASKS", cfg["matching"]["min_score_for_additional_tasks"])
        )

        cfg["llm_judge"]["enabled"] = _to_bool(kv.get("LLM_ENABLED", cfg["llm_judge"]["enabled"]))
        cfg["llm_judge"]["provider"] = str(kv.get("LLM_PROVIDER", cfg["llm_judge"]["provider"])).strip().lower()
        cfg["llm_judge"]["api_key"] = str(kv.get("LLM_API_KEY", cfg["llm_judge"]["api_key"])).strip()
        cfg["llm_judge"]["model"] = str(kv.get("LLM_MODEL", cfg["llm_judge"]["model"])).strip()
        cfg["llm_judge"]["base_url"] = str(kv.get("LLM_BASE_URL", cfg["llm_judge"]["base_url"])).strip().rstrip("/")
        cfg["llm_judge"]["top_k_candidates"] = int(kv.get("LLM_TOP_K_CANDIDATES", cfg["llm_judge"]["top_k_candidates"]))
        cfg["llm_judge"]["good_score_gap_threshold"] = int(
            kv.get("LLM_GOOD_SCORE_GAP_THRESHOLD", cfg["llm_judge"]["good_score_gap_threshold"])
        )
    except Exception:
        return cfg
    return cfg


RUNTIME_CONFIG = load_runtime_config()
MDR_SOURCE_CFG = RUNTIME_CONFIG["mdr_source"]
MATCH_CFG = RUNTIME_CONFIG["matching"]
LLM_CFG = RUNTIME_CONFIG["llm_judge"]
MOTHERDUCK_DB = MDR_SOURCE_CFG["motherduck_db"]
MOTHERDUCK_TOKEN = MDR_SOURCE_CFG["motherduck_token"]
MDR_CACHE_FILE = Path(__file__).resolve().parent / "mdr_cache.csv"
MDR_CACHE_REFRESH = False
MOTHERDUCK_QUERY = (
    "SELECT DISTINCT "
    "Document_title AS \"Doc. Title\", "
    "Mdr_code_name_ref AS \"Mdr_code_name_ref\" "
    "FROM my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All "
    "WHERE Document_title IS NOT NULL AND trim(Document_title) <> '' "
    "ORDER BY Document_title"
)
ALLOW_MULTIPLE_TASKS_PER_DOC = _to_bool(MATCH_CFG["allow_multiple_tasks_per_doc"])
MAX_TASKS_PER_DOC = int(MATCH_CFG["max_tasks_per_doc"])
MIN_SCORE_FOR_ADDITIONAL_TASKS = int(MATCH_CFG["min_score_for_additional_tasks"])
ENABLE_LLM_JUDGE = LLM_CFG["enabled"]
OPENAI_API_KEY = LLM_CFG["api_key"]
OPENAI_MODEL = LLM_CFG["model"]
OPENAI_BASE_URL = LLM_CFG["base_url"]
TOP_K_CANDIDATES = LLM_CFG["top_k_candidates"]
GOOD_SCORE_GAP_THRESHOLD = LLM_CFG["good_score_gap_threshold"]

# ===== FUZZY =====
try:
    from rapidfuzz import fuzz

    def sim(a, b):
        return fuzz.token_set_ratio(a, b)
except Exception:
    from difflib import SequenceMatcher

    def sim(a, b):
        return int(100 * SequenceMatcher(None, a, b).ratio())


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


def contains_any(text, keywords):
    return bool(text) and any(k in text for k in keywords)


# ===== CLASSIFICAZIONE TASK =====
DOC_KEYWORDS = (
    "ifi",
    "ifr",
    "ifc",
    "review",
    "comment resolution",
    "document",
    "drawing",
    "calculation",
    "datasheet approval",
    "method statement",
)

PROCUREMENT_KEYWORDS = (
    "rfq",
    "request for quotation",
    "technical alignment",
    "commercial alignment",
    "vendor",
    "rdd",
    "purchase",
    "po ",
    "bid",
    "offer",
    "datasheet",
    "specification",
    "ids",
)

CONSTRUCTION_KEYWORDS = ("construction", "site", "installation", "erection", "cantiere", "field")
TESTING_KEYWORDS = ("test", "testing", "precommissioning", "commissioning", "sat", "fat")
LOGISTICS_KEYWORDS = ("logistic", "transport", "shipment", "delivery", "packing", "expediting")
MEETING_KEYWORDS = ("meeting", "kick off", "kickoff", "coordination", "workshop")
DISCIPLINE_WBS_KEYWORDS = (
    "engineering",
    "ingegneria",
    "electrical",
    "instrumentation",
    "civil",
    "mechanical",
    "process",
    "piping",
)
CHAPTER_WBS_KEYWORDS = ("chapter", "item")
ENGINEERING_WBS_HINTS = ("engineering", "ingegneria", "document", "drawing", "calculation")
NON_DOC_WBS_HINTS = (
    "procurement",
    "purchase",
    "construction",
    "commissioning",
    "precommissioning",
    "logistic",
    "site",
)

# Mappa esplicita dai casi condivisi.
PROJECT_CASE_OVERRIDES = {
    "8001": "CASE_1_WBS_USEFUL_STRUCTURED",
    "8080": "CASE_1B_WBS_DISCIPLINE",
    "8189": "CASE_1B_WBS_DISCIPLINE",
    "7910": "CASE_2_WBS_CHAPTER_ITEM",
    "7920": "CASE_2_WBS_CHAPTER_ITEM",
    "8816": "CASE_3_WBS_INCOMPLETE_MDR_COVERAGE",
    "8540": "CASE_3_WBS_INCOMPLETE_MDR_COVERAGE",
    "7350": "CASE_4_WBS_WRONG_MISALIGNED",
    "7090": "CASE_5_WBS_USELESS_SINGLE",
    "6060": "CASE_6_MISSING_EXPORT",
}


def classify_task(name_norm, wbs_norm):
    """
    Ritorna:
    - task_class: ENG_DOC / OTHER
    - task_subclass: dettaglio
    - confidence: HIGH / MEDIUM / LOW
    - reason: regola applicata

    Nota: procurement ha priorita rispetto a keyword documentali.
    """
    if not name_norm and not wbs_norm:
        return "OTHER", "LOW", "empty_task_name_and_wbs"

    # Priorita assoluta: procurement = OTHER
    if contains_any(name_norm, PROCUREMENT_KEYWORDS):
        return "OTHER", "HIGH", "procurement_keyword"

    doc_signal = 0
    nondoc_signal = 0
    reasons = []

    if contains_any(name_norm, DOC_KEYWORDS):
        doc_signal += 2
        reasons.append("doc_keyword")
    if contains_any(name_norm, CONSTRUCTION_KEYWORDS):
        nondoc_signal += 2
        reasons.append("construction_keyword")
    if contains_any(name_norm, TESTING_KEYWORDS):
        nondoc_signal += 2
        reasons.append("testing_keyword")
    if contains_any(name_norm, LOGISTICS_KEYWORDS):
        nondoc_signal += 2
        reasons.append("logistics_keyword")
    if contains_any(name_norm, MEETING_KEYWORDS):
        nondoc_signal += 2
        reasons.append("meeting_keyword")

    # Segnali da WBS reale (wbs_name / wbs_short_name).
    if contains_any(wbs_norm, ENGINEERING_WBS_HINTS):
        doc_signal += 1
        reasons.append("wbs_engineering_hint")
    if contains_any(wbs_norm, NON_DOC_WBS_HINTS):
        nondoc_signal += 1
        reasons.append("wbs_non_doc_hint")

    if doc_signal > nondoc_signal and doc_signal > 0:
        conf = "HIGH" if (doc_signal - nondoc_signal) >= 2 else "MEDIUM"
        return "ENG_DOC", conf, "|".join(reasons) if reasons else "doc_signal"

    if nondoc_signal > 0:
        conf = "HIGH" if (nondoc_signal - doc_signal) >= 2 else "MEDIUM"
        return "OTHER", conf, "|".join(reasons) if reasons else "other_signal"

    return "OTHER", "LOW", "no_clear_keyword"


def derive_real_subclass(task):
    """
    Sottoclassi da dati reali cronoprogramma:
    - preferenza a WBS (wbs_name/wbs_short_name)
    - fallback a task_type
    """
    if "wbs_name" in task.columns:
        source = task["wbs_name"]
    elif "wbs_short_name" in task.columns:
        source = task["wbs_short_name"]
    else:
        source = pd.Series([""] * len(task), index=task.index)
    wbs_subclass = source.fillna("").astype(str).str.strip()
    if "task_type" in task.columns:
        tt = task["task_type"].fillna("").astype(str).str.strip()
        return wbs_subclass.where(wbs_subclass != "", tt)
    return wbs_subclass


def score_category(score):
    if pd.isna(score):
        return "NO_SCORE"
    score = int(score)
    if score >= 95:
        return "PERFECT"
    if score >= 85:
        return "GOOD"
    if score >= 70:
        return "MEDIUM"
    return "LOW"


def score_band(score):
    if pd.isna(score):
        return "NO_SCORE"
    score = int(score)
    if score >= 95:
        return "95-100 (PERFECT)"
    if score >= 85:
        return "85-94 (GOOD)"
    if score >= 70:
        return "70-84 (MEDIUM)"
    return "0-69 (LOW)"


def normalize_task_columns(task):
    rename_map = {}
    for c in task.columns:
        c_norm = str(c).strip().lower().replace(" ", "_")
        if c_norm in ("task_name", "task_code") and c != c_norm:
            rename_map[c] = c_norm
    if rename_map:
        task = task.rename(columns=rename_map)
    return task


def infer_project_id(ref_stem):
    m = re.search(r"(\d{4,6})", str(ref_stem))
    return m.group(1) if m else str(ref_stem)


def detect_wbs_column(task):
    candidates = [c for c in task.columns if "wbs" in str(c).lower()]
    preferred = ("wbs_name", "wbs", "wbs_path", "wbs code", "wbs_code")
    for p in preferred:
        for c in candidates:
            if str(c).lower() == p:
                return c
    return candidates[0] if candidates else None


def load_task_with_wbs(prim_file):
    """
    Carica TASK e, se disponibile, arricchisce con metadati WBS da PROJWBS.
    """
    task_raw = pd.read_excel(prim_file, sheet_name=TASK_SHEET)
    task_raw = normalize_task_columns(task_raw)
    if "task_name" not in task_raw.columns:
        raise ValueError("foglio TASK senza colonna 'task_name'")

    if "wbs_id" not in task_raw.columns:
        task_raw["wbs_name"] = ""
        task_raw["wbs_short_name"] = ""
        task_raw["wbs_parent_id"] = ""
        return task_raw

    try:
        wbs = pd.read_excel(prim_file, sheet_name="PROJWBS")
        wbs = wbs.rename(
            columns={
                "parent_wbs_id": "wbs_parent_id",
            }
        )
        keep = [c for c in ("wbs_id", "wbs_name", "wbs_short_name", "wbs_parent_id") if c in wbs.columns]
        if "wbs_id" in keep:
            task_raw = task_raw.merge(wbs[keep], on="wbs_id", how="left")
    except Exception:
        task_raw["wbs_name"] = ""
        task_raw["wbs_short_name"] = ""
        task_raw["wbs_parent_id"] = ""
    return task_raw


def build_wbs_profile(task):
    """
    Estrae metriche WBS reali usando i nomi WBS (se presenti).
    """
    if "wbs_name" in task.columns:
        source = task["wbs_name"]
        source_col = "wbs_name"
    elif "wbs_short_name" in task.columns:
        source = task["wbs_short_name"]
        source_col = "wbs_short_name"
    elif "wbs_id" in task.columns:
        source = task["wbs_id"]
        source_col = "wbs_id"
    else:
        source = pd.Series([""] * len(task))
        source_col = "none"

    s = source.fillna("").astype(str).str.strip()
    non_empty = s[s != ""]
    unique_count = int(non_empty.nunique()) if len(non_empty) > 0 else 0
    total_rows = int(len(task))
    top_share = 0.0
    if len(non_empty) > 0:
        top_share = float(non_empty.value_counts(normalize=True).iloc[0])

    norm_values = [normalize(x) for x in non_empty.unique().tolist()]
    has_discipline = any(contains_any(x, DISCIPLINE_WBS_KEYWORDS) for x in norm_values)
    has_chapter = any(contains_any(x, CHAPTER_WBS_KEYWORDS) for x in norm_values)
    has_engineering = any("engineering" in x or "ingegneria" in x for x in norm_values)
    hierarchy_depth_hint = int(task["wbs_parent_id"].notna().sum()) if "wbs_parent_id" in task.columns else 0

    return {
        "wbs_source_col": source_col,
        "wbs_non_empty_rows": int((s != "").sum()),
        "wbs_unique_count": unique_count,
        "wbs_top_share": round(top_share, 4),
        "wbs_has_discipline_keywords": has_discipline,
        "wbs_has_chapter_keywords": has_chapter,
        "wbs_has_engineering_keywords": has_engineering,
        "wbs_hierarchy_rows": hierarchy_depth_hint,
        "wbs_top5": non_empty.value_counts().head(5).to_dict() if len(non_empty) > 0 else {},
        "task_rows": total_rows,
    }


def classify_project_case(task, mdr_subset, ref_stem):
    """
    STEP 0: classifica progetto e suggerisce strategia.
    Restituisce (project_case, strategy, notes, wbs_col).
    """
    project_id = infer_project_id(ref_stem)
    profile = build_wbs_profile(task)
    unique_count = profile["wbs_unique_count"]
    top_share = profile["wbs_top_share"]
    has_discipline = profile["wbs_has_discipline_keywords"]
    has_chapter = profile["wbs_has_chapter_keywords"]
    has_engineering = profile["wbs_has_engineering_keywords"]

    # Data-driven: usa la struttura reale WBS; override solo per casi noti non osservabili (es. dati mancanti).
    case = None
    if unique_count <= 1 or top_share >= 0.95:
        case = "CASE_5_WBS_USELESS_SINGLE"
    elif has_chapter and not has_engineering:
        case = "CASE_2_WBS_CHAPTER_ITEM"
    elif has_discipline:
        case = "CASE_1B_WBS_DISCIPLINE"
    elif unique_count >= 100:
        case = "CASE_1_WBS_USEFUL_STRUCTURED"
    else:
        case = "CASE_1_WBS_USEFUL_STRUCTURED"

    # Applica override solo se presente per commessa nota.
    override_case = PROJECT_CASE_OVERRIDES.get(project_id)
    if override_case and override_case != "CASE_6_MISSING_EXPORT":
        case = override_case

    doc_task_count = int((task.get("task_class", pd.Series(dtype=str)) == "ENG_DOC").sum())
    mdr_count = int(len(mdr_subset))
    if case in ("CASE_1_WBS_USEFUL_STRUCTURED", "CASE_1B_WBS_DISCIPLINE") and mdr_count > 0 and doc_task_count < max(1, int(0.35 * mdr_count)):
        case = "CASE_3_WBS_INCOMPLETE_MDR_COVERAGE"

    if case == "CASE_1_WBS_USEFUL_STRUCTURED":
        return case, "TITLE_PLUS_WBS_STRONG", "WBS detailed and useful", profile
    if case == "CASE_1B_WBS_DISCIPLINE":
        return case, "TITLE_PLUS_WBS_DISCIPLINE", "WBS by discipline supports matching", profile
    if case == "CASE_2_WBS_CHAPTER_ITEM":
        return case, "CHAPTER_TO_ACTIVITY", "Use chapter/item logic, not direct doc mapping", profile
    if case == "CASE_3_WBS_INCOMPLETE_MDR_COVERAGE":
        return case, "PARTIAL_MATCH_ACCEPT_GAPS", "Accept incomplete MDR coverage", profile
    if case == "CASE_4_WBS_WRONG_MISALIGNED":
        return case, "TITLE_ONLY_IGNORE_WBS", "WBS misaligned, ignore WBS", profile
    if case == "CASE_5_WBS_USELESS_SINGLE":
        return case, "TITLE_ACTIVITY_ONLY", "WBS unusable/single bucket", profile
    return "CASE_UNKNOWN", "TITLE_ACTIVITY_ONLY", f"Fallback classification (MDR rows: {len(mdr_subset)})", profile


def prepare_task(task):
    task = task.copy()
    task["name_clean"] = task["task_name"].apply(remove_prefix)
    task["name_norm"] = task["name_clean"].apply(normalize)
    wbs_series = pd.Series([""] * len(task), index=task.index)
    if "wbs_name" in task.columns:
        wbs_series = task["wbs_name"].fillna("").astype(str)
    elif "wbs_short_name" in task.columns:
        wbs_series = task["wbs_short_name"].fillna("").astype(str)
    task["wbs_norm_for_classification"] = wbs_series.apply(normalize)
    classified = task.apply(lambda r: classify_task(r["name_norm"], r["wbs_norm_for_classification"]), axis=1)
    task["task_class"] = classified.apply(lambda x: x[0])
    task["classification_confidence"] = classified.apply(lambda x: x[1])
    task["classification_reason"] = classified.apply(lambda x: x[2])
    task["task_subclass_real"] = derive_real_subclass(task)
    return task


def load_mdr_full():
    """
    Carica MDR da MotherDuck usando DISTINCT Document_title non vuoti.
    """
    if (not MDR_CACHE_REFRESH) and MDR_CACHE_FILE.exists():
        mdr_full = pd.read_csv(MDR_CACHE_FILE, dtype=str, keep_default_na=False)
        if DOC_TITLE_COL in mdr_full.columns and MDR_CODE_NAME_REF_COL in mdr_full.columns:
            if DOC_NUMBER_COL not in mdr_full.columns:
                mdr_full[DOC_NUMBER_COL] = ""
            if MDR_REF_COL not in mdr_full.columns:
                mdr_full[MDR_REF_COL] = ""
            mdr_full[DOC_TITLE_COL] = mdr_full[DOC_TITLE_COL].fillna("").astype(str).str.strip()
            mdr_full[MDR_CODE_NAME_REF_COL] = mdr_full[MDR_CODE_NAME_REF_COL].fillna("").astype(str).str.strip()
            mdr_full = mdr_full[mdr_full[DOC_TITLE_COL] != ""]
            return mdr_full

    if not MOTHERDUCK_TOKEN:
        raise ValueError("Config mancante: imposta MOTHERDUCK_TOKEN in config.txt (oppure usa cache locale esistente).")
    conn = None
    try:
        # Metodo principale: connessione diretta MotherDuck via DSN md:
        dsn = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"
        conn = duckdb.connect(dsn)
    except Exception:
        # Fallback: connessione locale + attach esplicito dell'istanza MotherDuck
        if conn is not None:
            conn.close()
        conn = duckdb.connect()
        try:
            conn.execute("INSTALL motherduck")
        except Exception:
            pass
        conn.execute("LOAD motherduck")
        conn.execute(f"ATTACH 'md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}' AS {MOTHERDUCK_DB}")
    try:
        mdr_full = conn.execute(MOTHERDUCK_QUERY).df()
    finally:
        if conn is not None:
            conn.close()
    mdr_full.columns = [str(c).strip() for c in mdr_full.columns]
    if DOC_TITLE_COL not in mdr_full.columns:
        # supporta anche alias standard non quotati
        if "Document_title" in mdr_full.columns:
            mdr_full = mdr_full.rename(columns={"Document_title": DOC_TITLE_COL})
        elif "document_title" in mdr_full.columns:
            mdr_full = mdr_full.rename(columns={"document_title": DOC_TITLE_COL})
    if DOC_TITLE_COL not in mdr_full.columns:
        raise ValueError(f"La query MotherDuck deve restituire la colonna '{DOC_TITLE_COL}'.")
    if MDR_CODE_NAME_REF_COL not in mdr_full.columns:
        # support alias case variant
        if "mdr_code_name_ref" in mdr_full.columns:
            mdr_full = mdr_full.rename(columns={"mdr_code_name_ref": MDR_CODE_NAME_REF_COL})
        else:
            raise ValueError(f"La query MotherDuck deve restituire la colonna '{MDR_CODE_NAME_REF_COL}'.")
    if DOC_NUMBER_COL not in mdr_full.columns:
        mdr_full[DOC_NUMBER_COL] = ""
    if MDR_REF_COL not in mdr_full.columns:
        mdr_full[MDR_REF_COL] = ""
    mdr_full[DOC_TITLE_COL] = mdr_full[DOC_TITLE_COL].fillna("").astype(str).str.strip()
    mdr_full[MDR_CODE_NAME_REF_COL] = mdr_full[MDR_CODE_NAME_REF_COL].fillna("").astype(str).str.strip()
    mdr_full = mdr_full[mdr_full[DOC_TITLE_COL] != ""]
    try:
        MDR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        mdr_full.to_csv(MDR_CACHE_FILE, index=False, encoding="utf-8")
    except Exception:
        # La cache è opzionale: se non scrive, il run continua.
        pass
    return mdr_full


def build_mdr_subset_for_file(mdr_full, ref_stem):
    """
    Filtra MDR per cronoprogramma usando il codice progetto estratto dal nome file
    sulla colonna Mdr_code_name_ref (LIKE testuale).
    """
    project_code = infer_project_id(ref_stem)
    if not project_code:
        return mdr_full.iloc[0:0].copy()
    subset = mdr_full[mdr_full[MDR_CODE_NAME_REF_COL].str.contains(project_code, case=False, na=False)].copy()
    if subset.empty:
        return subset
    # Mantiene un solo record per titolo all'interno della commessa.
    subset = subset.drop_duplicates(subset=[DOC_TITLE_COL], keep="first")
    return subset


def match_mdr_to_doc_tasks(mdr_subset, task_doc, top_k=TOP_K_CANDIDATES):
    """
    Matching ENG_DOC-only, accettando relazione 1:N e N:N:
    ogni documento puo avere piu task migliori con stesso score.
    """
    mdr = mdr_subset.copy()
    mdr["title_norm"] = mdr[DOC_TITLE_COL].fillna("").astype(str).apply(normalize)
    results = []
    candidate_map = {}

    for _, row in mdr.iterrows():
        doc_title = "" if pd.isna(row[DOC_TITLE_COL]) else str(row[DOC_TITLE_COL]).strip()
        doc_number = "" if pd.isna(row.get(DOC_NUMBER_COL, "")) else str(row.get(DOC_NUMBER_COL, "")).strip()
        title_norm = row["title_norm"]

        if not title_norm:
            results.append(
                {
                    "Doc. Number": doc_number,
                    "Document Title": doc_title,
                    "match_N": "0",
                    "best_task_code": None,
                    "best_task_name": None,
                    "best_task_name_clean": None,
                    "best_score": None,
                    "category": "NO_TITLE",
                }
            )
            continue

        scored = [(sim(title_norm, t), i) for i, t in enumerate(task_doc["name_norm"]) if t]
        if not scored:
            doc_key = f"{doc_number}||{doc_title}"
            candidate_map[doc_key] = {
                "doc_number": doc_number,
                "doc_title": doc_title,
                "top1_score": None,
                "top2_score": None,
                "score_gap_top1_top2": None,
                "top_candidates": [],
            }
            results.append(
                {
                    "Doc. Number": doc_number,
                    "Document Title": doc_title,
                    "match_N": "0",
                    "best_task_code": None,
                    "best_task_name": None,
                    "best_task_name_clean": None,
                    "best_score": None,
                    "category": "NO_ENG_DOC_TASKS",
                }
            )
            continue

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        best_score = scored_sorted[0][0]
        best_indices = [i for s, i in scored_sorted if s == best_score]
        selected_indices = []
        if ALLOW_MULTIPLE_TASKS_PER_DOC:
            limit = None if MAX_TASKS_PER_DOC <= 0 else max(1, MAX_TASKS_PER_DOC)
            for s, i in scored_sorted:
                if limit is not None and len(selected_indices) >= limit:
                    break
                # Mantiene sempre i best tie; aggiunge altri task solo sopra soglia.
                if s == best_score or int(s) >= MIN_SCORE_FOR_ADDITIONAL_TASKS:
                    if i not in selected_indices:
                        selected_indices.append(i)
            if not selected_indices:
                selected_indices = best_indices[:1]
        else:
            selected_indices = best_indices
        top1 = scored_sorted[0][0] if scored_sorted else None
        top2 = scored_sorted[1][0] if len(scored_sorted) > 1 else None
        score_gap = (top1 - top2) if (top1 is not None and top2 is not None) else None
        top_candidates = []
        for rank, (score, idx) in enumerate(scored_sorted[:max(1, top_k)], 1):
            t = task_doc.iloc[idx]
            top_candidates.append(
                {
                    "rank": rank,
                    "score": int(score),
                    "task_code": t["task_code"] if "task_code" in task_doc.columns else None,
                    "task_name": t["task_name"],
                    "wbs_name": t.get("wbs_name"),
                }
            )
        doc_key = f"{doc_number}||{doc_title}"
        candidate_map[doc_key] = {
            "doc_number": doc_number,
            "doc_title": doc_title,
            "top1_score": top1,
            "top2_score": top2,
            "score_gap_top1_top2": score_gap,
            "top_candidates": top_candidates,
        }
        for rank, idx in enumerate(selected_indices, 1):
            t = task_doc.iloc[idx]
            candidate_score = next((s for s, i in scored_sorted if i == idx), best_score)
            category = score_category(candidate_score)
            results.append(
                {
                    "Doc. Number": doc_number,
                    "Document Title": doc_title,
                    "match_N": f"{rank} of {len(best_indices)}",
                    "best_task_code": t["task_code"] if "task_code" in task_doc.columns else None,
                    "best_task_name": t["task_name"],
                    "best_task_name_clean": t["name_clean"],
                    "best_score": candidate_score,
                    "category": category,
                    "top1_score": top1,
                    "top2_score": top2,
                    "score_gap_top1_top2": score_gap,
                    "association_mode": "multi_task_per_doc" if ALLOW_MULTIPLE_TASKS_PER_DOC else "best_tie_only",
                }
            )
    return pd.DataFrame(results), candidate_map


def stats_with_pct(series, label):
    out = series.value_counts().rename_axis(label).reset_index(name="count")
    total = int(out["count"].sum())
    out["pct"] = (out["count"] / total * 100).round(2) if total > 0 else 0
    return out


def explain_matching_strategy(strategy):
    explanations = {
        "TITLE_PLUS_WBS_STRONG": "Matching su titolo task con supporto forte della WBS (WBS strutturata e informativa).",
        "TITLE_PLUS_WBS_DISCIPLINE": "Matching su titolo task con supporto WBS per disciplina (engineering/procurement/construction).",
        "CHAPTER_TO_ACTIVITY": "Approccio chapter/item: collega attivita chiave, non forza mapping diretto documento->task.",
        "PARTIAL_MATCH_ACCEPT_GAPS": "Matching parziale: accetta buchi MDR/cronoprogramma senza forzare copertura completa.",
        "TITLE_ONLY_IGNORE_WBS": "Ignora la WBS per il matching e usa solo titolo/contenuto task.",
        "TITLE_ACTIVITY_ONLY": "WBS poco utile: classifica per tipo attivita e usa matching basato su titolo.",
        "NOT_PROCESSABLE": "Dati insufficienti per elaborare il matching (es. export Primavera mancante).",
    }
    return explanations.get(strategy, "Strategia non riconosciuta: usa matching conservativo basato su titolo task.")


def explain_summary_metric(metric):
    explanations = {
        "task_rows": "Numero totale di attivita presenti nel foglio TASK del cronoprogramma.",
        "eng_doc_task_rows": "Numero di attivita classificate come ENG_DOC (progresso documentale).",
        "mdr_rows": "Numero di titoli MDR usati per la commessa dopo filtro su Mdr_code_name_ref.",
        "project_case": "Categoria della commessa derivata dalla struttura WBS.",
        "matching_strategy": "Strategia di matching scelta in base al project_case.",
        "matching_strategy_explanation": "Spiegazione testuale della strategia applicata.",
        "llm_enabled": "Indica se il giudizio LLM e attivo (true/false).",
        "llm_rows_reviewed": "Numero di record effettivamente valutati dal judge LLM.",
    }
    return explanations.get(metric, "")


def explain_classification_reason(reason_code):
    code = str(reason_code or "").strip().lower()
    mapping = {
        "empty_task_name_and_wbs": "Task senza testo e senza contesto WBS: classificata OTHER.",
        "procurement_keyword": "Rilevata keyword procurement nel task_name: classificata OTHER.",
        "doc_signal": "Segnali complessivi orientati a documento di ingegneria.",
        "other_signal": "Segnali complessivi orientati ad attivita non documentale.",
        "no_clear_keyword": "Nessuna keyword forte rilevata: fallback conservativo su OTHER.",
    }
    if code in mapping:
        return mapping[code]

    parts = [p for p in code.split("|") if p]
    part_map = {
        "doc_keyword": "keyword documentale nel task_name",
        "construction_keyword": "keyword construction nel task_name",
        "testing_keyword": "keyword testing nel task_name",
        "logistics_keyword": "keyword logistics nel task_name",
        "meeting_keyword": "keyword meeting nel task_name",
        "wbs_engineering_hint": "contesto WBS con hint engineering/documentale",
        "wbs_non_doc_hint": "contesto WBS con hint non documentale",
    }
    if parts:
        readable = [part_map.get(p, p) for p in parts]
        return "Segnali rilevati: " + ", ".join(readable) + "."
    return ""


def should_send_to_llm(row):
    category = str(row.get("category", ""))
    match_n = str(row.get("match_N", ""))
    gap = row.get("score_gap_top1_top2")
    has_tie = " of " in match_n and not match_n.startswith("1 of 1")
    if category in ("MEDIUM", "LOW"):
        return True
    if category == "GOOD" and has_tie:
        return True
    if category == "GOOD" and pd.notna(gap) and int(gap) < GOOD_SCORE_GAP_THRESHOLD:
        return True
    return False


def call_llm_judge(payload):
    """
    Chiama un endpoint OpenAI-compatible e restituisce decisione JSON strutturata.
    Fallback conservativo: REVIEW.
    """
    fallback = {
        "decision": "REVIEW",
        "selected_task_code": None,
        "confidence": "LOW",
        "reason_short": "LLM unavailable or invalid response",
        "reason_flags": ["LLM_FALLBACK"],
    }
    if not ENABLE_LLM_JUDGE or not OPENAI_API_KEY:
        fallback["reason_short"] = "LLM disabled or missing API key"
        return fallback
    system_msg = (
        "You are a reconciliation judge for engineering documents. "
        "Select at most one task from provided candidates. "
        "Never invent task codes. If uncertain choose REVIEW. "
        "Return ONLY valid JSON."
    )
    user_msg = {
        "instructions": "Assess if one candidate is a valid match for the MDR document.",
        "expected_json_schema": {
            "decision": "ACCEPT_MATCH|REJECT_MATCH|REVIEW",
            "selected_task_code": "string|null",
            "confidence": "HIGH|MEDIUM|LOW",
            "reason_short": "string",
            "reason_flags": ["string"],
        },
        "payload": payload,
    }
    body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        f"{OPENAI_BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        parsed = json.loads(content)
        decision = str(parsed.get("decision", "REVIEW")).upper().strip()
        selected = parsed.get("selected_task_code")
        confidence = str(parsed.get("confidence", "LOW")).upper().strip()
        if decision not in ("ACCEPT_MATCH", "REJECT_MATCH", "REVIEW"):
            return fallback
        if confidence not in ("HIGH", "MEDIUM", "LOW"):
            confidence = "LOW"
        return {
            "decision": decision,
            "selected_task_code": selected,
            "confidence": confidence,
            "reason_short": str(parsed.get("reason_short", ""))[:300],
            "reason_flags": parsed.get("reason_flags", []) if isinstance(parsed.get("reason_flags", []), list) else [],
        }
    except Exception:
        return fallback


def slim_task_output(task):
    out = task.copy()
    out["classification_reason_explanation"] = out["classification_reason"].apply(explain_classification_reason)
    cols = [
        "task_code",
        "task_name",
        "wbs_name",
        "task_class",
        "task_subclass_real",
        "classification_confidence",
        "classification_reason",
        "classification_reason_explanation",
    ]
    available = [c for c in cols if c in out.columns]
    return out[available].copy()


def slim_result_output(result):
    cols = [
        "Doc. Number",
        "Document Title",
        "match_N",
        "best_task_code",
        "best_task_name",
        "best_score",
        "category",
        "llm_decision",
        "llm_selected_task_code",
        "llm_confidence",
        "llm_reason_short",
    ]
    available = [c for c in cols if c in result.columns]
    return result[available].copy()


def main():
    if not CRONOPROGRAMMI_DIR.exists():
        raise FileNotFoundError(f"Cartella cronoprogrammi non trovata: {CRONOPROGRAMMI_DIR}")

    mdr_full = load_mdr_full()

    cronoprogramma_files = sorted(CRONOPROGRAMMI_DIR.glob("*.xlsx"))
    if not cronoprogramma_files:
        raise FileNotFoundError(f"Nessun file .xlsx in {CRONOPROGRAMMI_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_task_rows = []
    project_diagnosis_rows = []

    for prim_file in cronoprogramma_files:
        ref_name = prim_file.name
        ref_stem = prim_file.stem

        mdr_subset = build_mdr_subset_for_file(mdr_full, ref_stem)
        if mdr_subset.empty:
            print(f"Skip {ref_name}: nessuna riga MDR con {MDR_CODE_NAME_REF_COL} LIKE '%{infer_project_id(ref_stem)}%'")
            continue

        try:
            task_raw = load_task_with_wbs(prim_file)
            task = prepare_task(task_raw)
        except Exception as exc:
            print(f"Skip {ref_name}: {exc}")
            continue

        task["cronoprogramma"] = ref_stem
        task_doc = task[task["task_class"] == "ENG_DOC"].copy()
        project_case, strategy, case_note, wbs_profile = classify_project_case(task, mdr_subset, ref_stem)

        result, candidate_map = match_mdr_to_doc_tasks(mdr_subset, task_doc, top_k=TOP_K_CANDIDATES)
        result["score_band"] = result["best_score"].apply(score_band)
        result["cronoprogramma"] = ref_stem
        result["project_case"] = project_case
        result["matching_strategy"] = strategy

        # LLM cascade: solo su casi incerti.
        result["llm_needed"] = result.apply(should_send_to_llm, axis=1)
        result["llm_decision"] = ""
        result["llm_selected_task_code"] = None
        result["llm_confidence"] = ""
        result["llm_reason_short"] = ""
        llm_review_rows = []
        # Una chiamata per documento, non per ogni riga tie.
        for doc_key, group in result.groupby(["Doc. Number", "Document Title"], dropna=False):
            group_idx = group.index.tolist()
            if not any(bool(result.loc[i, "llm_needed"]) for i in group_idx):
                continue
            key = f"{'' if pd.isna(doc_key[0]) else doc_key[0]}||{'' if pd.isna(doc_key[1]) else doc_key[1]}"
            payload = {
                "project_case": project_case,
                "matching_strategy": strategy,
                "mdr_doc_number": doc_key[0],
                "mdr_doc_title": doc_key[1],
                "top_candidates": candidate_map.get(key, {}).get("top_candidates", []),
                "top1_score": candidate_map.get(key, {}).get("top1_score"),
                "top2_score": candidate_map.get(key, {}).get("top2_score"),
                "score_gap_top1_top2": candidate_map.get(key, {}).get("score_gap_top1_top2"),
            }
            llm = call_llm_judge(payload)
            for i in group_idx:
                result.at[i, "llm_decision"] = llm["decision"]
                result.at[i, "llm_selected_task_code"] = llm["selected_task_code"]
                result.at[i, "llm_confidence"] = llm["confidence"]
                result.at[i, "llm_reason_short"] = llm["reason_short"]
            llm_review_rows.append(
                {
                    "Doc. Number": doc_key[0],
                    "Document Title": doc_key[1],
                    "top1_score": payload["top1_score"],
                    "top2_score": payload["top2_score"],
                    "score_gap_top1_top2": payload["score_gap_top1_top2"],
                    "llm_decision": llm["decision"],
                    "llm_selected_task_code": llm["selected_task_code"],
                    "llm_confidence": llm["confidence"],
                    "llm_reason_short": llm["reason_short"],
                }
            )

        project_diagnosis_rows.append(
            {
                "cronoprogramma": ref_stem,
                "project_id": infer_project_id(ref_stem),
                "project_case": project_case,
                "matching_strategy": strategy,
                "wbs_column": wbs_profile.get("wbs_source_col", "N/A"),
                "total_task_rows": len(task),
                "eng_doc_task_rows": int((task["task_class"] == "ENG_DOC").sum()),
                "mdr_rows": len(mdr_subset),
                "diagnosis_note": case_note,
                "wbs_non_empty_rows": wbs_profile.get("wbs_non_empty_rows", 0),
                "wbs_unique_count": wbs_profile.get("wbs_unique_count", 0),
                "wbs_top_share": wbs_profile.get("wbs_top_share", 0.0),
                "wbs_has_discipline_keywords": wbs_profile.get("wbs_has_discipline_keywords", False),
                "wbs_has_chapter_keywords": wbs_profile.get("wbs_has_chapter_keywords", False),
                "wbs_top5": str(wbs_profile.get("wbs_top5", {})),
            }
        )

        out_path = OUTPUT_DIR / f"matching_{ref_stem}.xlsx"
        task_out = slim_task_output(task)
        result_out = slim_result_output(result)
        stat_summary = pd.DataFrame(
            [
                {"metric": "task_rows", "value": len(task)},
                {"metric": "eng_doc_task_rows", "value": int((task["task_class"] == "ENG_DOC").sum())},
                {"metric": "mdr_rows", "value": len(mdr_subset)},
                {"metric": "project_case", "value": project_case},
                {"metric": "matching_strategy", "value": strategy},
                {"metric": "matching_strategy_explanation", "value": explain_matching_strategy(strategy)},
                {"metric": "llm_enabled", "value": ENABLE_LLM_JUDGE},
                {"metric": "llm_rows_reviewed", "value": int((result["llm_decision"] != "").sum())},
            ]
        )
        stat_summary["explanation"] = stat_summary["metric"].apply(explain_summary_metric)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            stat_summary.to_excel(writer, sheet_name="Summary", index=False)
            task_out.to_excel(writer, sheet_name="Task_ENG_DOC_flag", index=False)
            result_out.to_excel(writer, sheet_name="Match_ENG_DOC_only", index=False)
            stats_with_pct(result["category"], "category").to_excel(writer, sheet_name="Match_category", index=False)
            if llm_review_rows:
                pd.DataFrame(llm_review_rows).to_excel(writer, sheet_name="LLM_review", index=False)
        print(f"Creato: {out_path}")

        all_results.append(result)
        all_task_rows.append(task[["cronoprogramma", "task_class", "task_subclass_real"]].copy())

    if all_results:
        results_all = pd.concat(all_results, ignore_index=True)
        tasks_all = pd.concat(all_task_rows, ignore_index=True) if all_task_rows else pd.DataFrame()
        out_agg = OUTPUT_DIR / "statistiche_aggregate_v2.xlsx"
        with pd.ExcelWriter(out_agg, engine="openpyxl") as writer:
            stats_with_pct(results_all["category"], "category").to_excel(writer, sheet_name="Match_category", index=False)
            if "project_case" in results_all.columns:
                stats_with_pct(results_all["project_case"], "project_case").to_excel(writer, sheet_name="Project_case", index=False)
            if not tasks_all.empty:
                stats_with_pct(tasks_all["task_class"], "task_class").to_excel(writer, sheet_name="Task_class", index=False)
        print(f"Creato: {out_agg}")

    # Diagnosi esplicita includendo commesse note mancanti (es. 6060).
    seen_project_ids = {row["project_id"] for row in project_diagnosis_rows}
    for pid, case in PROJECT_CASE_OVERRIDES.items():
        if pid not in seen_project_ids and case == "CASE_6_MISSING_EXPORT":
            project_diagnosis_rows.append(
                {
                    "cronoprogramma": pid,
                    "project_id": pid,
                    "project_case": case,
                    "matching_strategy": "NOT_PROCESSABLE",
                    "wbs_column": "N/A",
                    "total_task_rows": 0,
                    "doc_task_rows": 0,
                    "mdr_rows": 0,
                    "diagnosis_note": "No Primavera export available",
                }
            )
    print("Fatto.")


if __name__ == "__main__":
    main()
