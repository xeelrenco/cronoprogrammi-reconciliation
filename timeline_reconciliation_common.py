import hashlib
import json
import re
import time
import urllib.error
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


def document_title_join_key(text) -> str:
    """Stable key for joining historical MDR titles to consolidated titles."""
    return normalize(text)


def sql_document_title_join_key(column_sql: str) -> str:
    """SQL expression matching document_title_join_key() for DuckDB joins."""
    col = column_sql
    return (
        f"lower(trim(regexp_replace("
        f"regexp_replace("
        f"regexp_replace(replace(lower({col}), '&', ' and '), '\\([^)]*\\)', ' ', 'g'), "
        f"'[^a-z0-9]+', ' ', 'g'), "
        f"'\\s+', ' ', 'g')))"
    )


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


DOC_STATUS_PREFIX_RE = re.compile(
    r"^(IFI\+?|IFC|IDC|IFR|IFA|IFD|ASB|IFF|IFT|IFO|AB|APP|INT)\s*[-–—]\s*",
    re.I,
)
GENERIC_SUBJECT_TOKENS = frozenset(
    {
        "specification", "specifications", "sheet", "data", "ids", "rdds", "mto", "drawing",
        "drawings", "layout", "report", "diagram", "system", "engineering", "document",
        "documents", "revision", "approval", "period", "issue", "purchase", "including",
        "material", "materials", "technical", "design", "plan", "details", "detail",
        "general", "typical", "vendor", "package", "progress", "update", "delivery",
        "and", "for", "the", "with", "from", "area", "period", "cancel", "deleted",
    }
)
TITLE_MATCH_MIN_SCORE = 0.88
TITLE_MATCH_STRONG_SCORE = 0.97
SUBJECT_TOKEN_ALIASES = {
    "bld": "building",
    "arch": "architectural",
    "architecturals": "architectural",
    "dwgs": "drawings",
}
BUILDING_CONTEXT_TOKENS = frozenset(
    {
        "security", "house", "building", "main", "architectural", "arch",
        "architecturals", "dwgs", "drawings", "doors", "windows", "schedule",
        "abacus", "buildings", "cancel",
    }
)
DRAINAGE_SUBTYPE_TOKENS = {
    "groundwater": frozenset({"groundwater"}),
    "rain": frozenset({"rain", "storm", "rainwater", "stormwater"}),
    "sewer": frozenset({"sewer", "sewerage", "sanitary"}),
    "sump": frozenset({"sump", "closed"}),
}
INCOMPATIBLE_DRAINAGE_SUBTYPES = frozenset(
    {
        ("groundwater", "rain"),
        ("groundwater", "sewer"),
        ("groundwater", "sump"),
        ("rain", "groundwater"),
        ("sewer", "groundwater"),
        ("sump", "groundwater"),
    }
)
EQUIPMENT_ANCHOR_RULES = (
    frozenset({"cccw", "circulating"}),
    frozenset({"soft", "start"}),
)
EQUIPMENT_ANCHOR_MATCH_SCORE = 0.93
BUNDLE_DOC_TYPE_KEYWORDS = (
    "specification",
    "data sheet",
    "ids",
    "rdds",
    "mto",
    "drawing",
    "layout",
    "report",
    "diagram",
    "schedule",
    "follow up",
    "follow-up",
)
INCOMPATIBLE_DOCUMENT_TYPE_PAIRS = frozenset(
    {
        ("tbe", "specification"),
        ("tbe", "data_sheet"),
        ("tbe", "ids"),
        ("tbe", "dossier"),
        ("drawing", "mto"),
        ("drawing", "calculation"),
        ("drawing", "model_3d"),
        ("specification", "ids"),
        ("specification", "mto"),
        ("specification", "drawing"),
        ("data_sheet", "specification"),
        ("data_sheet", "mto"),
        ("data_sheet", "ids"),
        ("data_sheet", "calculation"),
        ("drawing", "data_sheet"),
        ("tbe", "drawing"),
        ("procedure", "ids"),
        ("procedure", "data_sheet"),
        ("ids", "dossier"),
        ("ids", "calculation"),
        ("data_sheet", "dossier"),
        ("arrangement", "model_3d"),
        ("pid", "mto"),
    }
)
FACILITY_IDENTITY_MARKERS = {
    "operation": frozenset({"operation"}),
    "security": frozenset({"security"}),
    "analyser": frozenset({"analyser", "analyzer"}),
    "co2": frozenset({"co2", "extinguishing", "extinguisher"}),
    "gt_st": frozenset({"gt", "st", "turbine", "steam"}),
}
INCOMPATIBLE_FACILITY_MARKERS = frozenset(
    {
        ("operation", "security"),
        ("gt_st", "co2"),
    }
)
EQUIPMENT_MODIFIER_CONFLICTS = (
    (frozenset({"manual"}), frozenset({"actuated"})),
    (frozenset({"scu", "ucp"}), frozenset({"fgs"})),
    (frozenset({"firefighting"}), frozenset({"inert"})),
)
REQUIRED_DESCRIPTOR_TOKENS = frozenset(
    {
        "architectural",
        "structural",
        "manual",
        "actuated",
        "arrangement",
    }
)
SCOPE_NARROWING_MARKERS = frozenset({"portable", "inert"})
SCOPE_DOMAIN_TOKENS = frozenset(
    {
        "firefighting",
        "fire",
        "fighting",
        "extinguishing",
        "lifting",
        "hoists",
        "hoist",
        "cranes",
        "crane",
        "davits",
        "davit",
    }
)


def task_lists_document_bundle(task_name: str) -> bool:
    clean = remove_prefix(str(task_name or "")).lower()
    if " / " not in clean:
        return False
    parts = [p.strip() for p in clean.split(" / ")]
    hits = sum(1 for part in parts if any(kw in part for kw in BUNDLE_DOC_TYPE_KEYWORDS))
    return hits >= 2


def task_subject_for_match(task_name: str) -> str:
    s = str(task_name or "").strip()
    s = DOC_STATUS_PREFIX_RE.sub("", s).strip()
    s = remove_prefix(s).strip()
    s = re.sub(r"\s*[-–—]\s*Revision and Approval Period\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*[-–—]\s*\d+(?:st|nd|rd|th)?\s+Issue\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*[-–—]\s*Cancel\s*$", "", s, flags=re.I)
    s = re.sub(r"/\d+\+?\s*$", "", s).strip()
    return normalize(s)


def _match_text_for_document_types(text: str) -> str:
    return task_subject_for_match(text) if text else normalize(text)


def extract_document_types(text: str) -> set:
    n = _match_text_for_document_types(text)
    if not n:
        return set()
    types = set()
    if re.search(r"\b(tbe|technical evaluation|tech(?:nical)?\s+eval(?:uation)?|te for)\b", n):
        types.add("tbe")
    if re.search(r"\b(inspection data sheets?|inspections data sheets?)\b", n) or re.search(
        r"\bids\b", n
    ):
        types.add("ids")
    elif re.search(r"\b(data sheets?|datasheet|foglio dati|technical data sheet)\b", n):
        types.add("data_sheet")
    if re.search(
        r"\b(specifications?|supply specs?|capitolato|design specification|specification for purchase|technical supply specification)\b",
        n,
    ):
        types.add("specification")
    if re.search(r"\b(3d model|3 d model)\b", n):
        types.add("model_3d")
    elif re.search(
        r"\b(calculation report|sizing report|study report|relazione di calcolo|structural calcul\w*)\b",
        n,
    ):
        types.add("calculation")
    elif re.search(
        r"\b(drawing|drawings|dwg|dwgs|planimetria|layout|arrangement|isometric|architectural design|structural drawings|grading plan|plot plan|general arrangement|piping assembly)\b",
        n,
    ):
        types.add("drawing")
    elif re.search(r"\bpiping arrangement\b", n):
        types.add("arrangement")
    elif re.search(r"\b(plan|sections|elevations)\b", n) and not types & {"specification", "tbe"}:
        types.add("drawing")
    if re.search(r"\b(mto|boq|bill of quantities|take off|takeoff|material list|materials mto)\b", n):
        types.add("mto")
    if re.search(r"\b(sat procedure|fat procedure|check sheets|check sheet)\b", n) or (
        re.search(r"\bprocedure\b", n) and "specification" not in n
    ):
        types.add("procedure")
    if re.search(
        r"\b(p id|p&id|pid|piping instrumentation diagrams?|piping and instrument diagrams?|instrumentation diagrams?)\b",
        n,
    ):
        types.add("pid")
    if re.search(r"\b(dossier|pre commissioning|precommissioning)\b", n):
        types.add("dossier")
    if re.search(r"\b(platform|access platform)\b", n) or re.search(r"\bplt[- ]?\d", n):
        types.add("drawing")
    if re.search(r"\b(formwork|reinforcement)\b", n):
        types.add("calculation")
    return types


def _document_types_compatible(task_type: str, mdr_type: str) -> bool:
    if task_type == mdr_type:
        return True
    if {task_type, mdr_type} <= {"drawing", "arrangement"}:
        return True
    return (task_type, mdr_type) not in INCOMPATIBLE_DOCUMENT_TYPE_PAIRS and (
        mdr_type,
        task_type,
    ) not in INCOMPATIBLE_DOCUMENT_TYPE_PAIRS


def document_type_conflict(task_name: str, mdr_title: str) -> bool:
    is_bundle = task_lists_document_bundle(task_name)
    task_types = extract_document_types(task_name)
    mdr_types = extract_document_types(mdr_title)
    if not task_types or not mdr_types:
        return False
    if is_bundle:
        return not any(
            _document_types_compatible(task_type, mdr_type)
            for task_type in task_types
            for mdr_type in mdr_types
        )
    for task_type in task_types:
        for mdr_type in mdr_types:
            if not _document_types_compatible(task_type, mdr_type):
                return True
    return False


def _area_codes(text: str) -> set:
    return {m.group(1) for m in re.finditer(r"\barea\s+([a-z0-9]+)\b", normalize(text))}


def _facility_marker_keys(tokens) -> set:
    keys = set()
    for key, markers in FACILITY_IDENTITY_MARKERS.items():
        if tokens & markers:
            keys.add(key)
    return keys


def facility_identity_conflict(task_name: str, mdr_title: str) -> bool:
    task_text = task_subject_for_match(task_name)
    mdr_text = normalize(mdr_title)
    task_tokens = set(task_text.split()) | _parenthetical_tokens(task_name)
    mdr_tokens = set(mdr_text.split()) | _parenthetical_tokens(mdr_title)
    task_areas = _area_codes(task_text)
    mdr_areas = _area_codes(mdr_text)
    if task_areas and mdr_areas and not (task_areas & mdr_areas):
        return True
    if {"gt", "st"} <= task_tokens and not ({"gt", "st"} & mdr_tokens):
        if "pid" in extract_document_types(task_name) or "diagram" in task_text:
            return True
    task_facility = _facility_marker_keys(task_tokens)
    mdr_facility = _facility_marker_keys(mdr_tokens)
    if not task_facility or not mdr_facility:
        return False
    if task_facility & mdr_facility:
        return False
    for task_key in task_facility:
        for mdr_key in mdr_facility:
            if (task_key, mdr_key) in INCOMPATIBLE_FACILITY_MARKERS or (
                mdr_key,
                task_key,
            ) in INCOMPATIBLE_FACILITY_MARKERS:
                return True
    return False


def equipment_modifier_conflict(task_name: str, mdr_title: str) -> bool:
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    for task_mods, mdr_mods in EQUIPMENT_MODIFIER_CONFLICTS:
        if (task_tokens & task_mods) and (mdr_tokens & mdr_mods) and not (task_tokens & mdr_mods):
            return True
        if (mdr_tokens & task_mods) and (task_tokens & mdr_mods) and not (mdr_tokens & task_mods):
            return True
    return False


def _document_types_share_family(task_types, mdr_types) -> bool:
    if not task_types or not mdr_types:
        return False
    return any(
        _document_types_compatible(task_type, mdr_type)
        for task_type in task_types
        for mdr_type in mdr_types
    )


def descriptor_mismatch_conflict(task_name: str, mdr_title: str) -> bool:
    task_types = extract_document_types(task_name)
    mdr_types = extract_document_types(mdr_title)
    if not _document_types_share_family(task_types, mdr_types):
        return False
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    for desc in REQUIRED_DESCRIPTOR_TOKENS:
        if desc in task_tokens and desc not in mdr_tokens:
            return True
    return False


def significant_tokens(text: str):
    return [t for t in normalize(text).split() if t not in GENERIC_SUBJECT_TOKENS and len(t) > 2]


def _parenthetical_tokens(text: str) -> set:
    tokens = set()
    for m in re.finditer(r"\(([A-Za-z0-9/&+\-\s]{2,40})\)", str(text or "")):
        inner = normalize(m.group(1))
        tokens.update(t for t in inner.split() if len(t) > 1)
    return tokens


def normalized_match_tokens(text: str):
    expanded = set(significant_tokens(text))
    expanded |= _parenthetical_tokens(text)
    for token in list(expanded):
        alias = SUBJECT_TOKEN_ALIASES.get(token)
        if alias:
            expanded.add(alias)
    return expanded


def _drainage_subtype_keys(tokens) -> set:
    keys = set()
    for key, markers in DRAINAGE_SUBTYPE_TOKENS.items():
        if tokens & markers:
            keys.add(key)
    return keys


def drainage_subtype_conflict(task_name: str, mdr_title: str) -> bool:
    """Block links across incompatible hydraulic subtypes (e.g. groundwater vs rain)."""
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    task_keys = _drainage_subtype_keys(task_tokens)
    mdr_keys = _drainage_subtype_keys(mdr_tokens)
    if not task_keys or not mdr_keys or task_keys & mdr_keys:
        return False
    for task_key in task_keys:
        for mdr_key in mdr_keys:
            if (task_key, mdr_key) in INCOMPATIBLE_DRAINAGE_SUBTYPES:
                return True
    return False


def shares_equipment_anchor(task_name: str, mdr_title: str) -> bool:
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    return any(rule <= task_tokens and rule <= mdr_tokens for rule in EQUIPMENT_ANCHOR_RULES)


def equipment_scope_narrowing_conflict(task_name: str, mdr_title: str) -> bool:
    """Block when MDR is a narrower equipment scope (e.g. portable) than the task."""
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    mdr_narrow = SCOPE_NARROWING_MARKERS & mdr_tokens
    if not mdr_narrow or (SCOPE_NARROWING_MARKERS & task_tokens):
        return False
    if not (task_tokens & SCOPE_DOMAIN_TOKENS) or not (mdr_tokens & SCOPE_DOMAIN_TOKENS):
        return False
    return True


def link_subject_gates_block(task_name: str, mdr_title: str) -> bool:
    if drainage_subtype_conflict(task_name, mdr_title):
        return True
    if document_type_conflict(task_name, mdr_title):
        return True
    if facility_identity_conflict(task_name, mdr_title):
        return True
    if equipment_modifier_conflict(task_name, mdr_title):
        return True
    if equipment_scope_narrowing_conflict(task_name, mdr_title):
        return True
    if descriptor_mismatch_conflict(task_name, mdr_title):
        return True
    return title_has_hard_subject_conflict(task_name, mdr_title)


def _candidate_match_titles(row, mdr_title_col="MdrDocumentTitle"):
    titles = []
    mdr = str(row.get(mdr_title_col, "") or "").strip()
    if mdr:
        titles.append(mdr)
    raci = str(row.get("ConsolidatedRaciTitle", "") or "").strip()
    if raci and raci not in titles:
        titles.append(raci)
    return titles


def candidate_title_match_score(task_name: str, row, mdr_title_col="MdrDocumentTitle") -> float:
    scores = [mdr_title_match_score(task_name, title) for title in _candidate_match_titles(row, mdr_title_col)]
    return max(scores) if scores else 0.0


def title_has_hard_subject_conflict(task_name: str, mdr_title: str, score=None) -> bool:
    """True only for near-duplicate titles with swapped equipment/subject (e.g. Compressor vs Gearbox)."""
    if shares_equipment_anchor(task_name, mdr_title):
        return False
    if score is None:
        score = mdr_title_match_score(task_name, mdr_title)
    if score >= TITLE_MATCH_STRONG_SCORE:
        return False
    if score < TITLE_MATCH_MIN_SCORE:
        return False
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    unmatched_task = task_tokens - mdr_tokens
    unmatched_mdr = mdr_tokens - task_tokens
    if not unmatched_task or not unmatched_mdr:
        return False
    overlap = task_tokens & mdr_tokens
    return len(overlap) >= 3


def mdr_title_match_score(task_name: str, mdr_title: str) -> float:
    a = task_subject_for_match(task_name)
    b = normalize(mdr_title)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 12 and (a in b or b in a):
        return 0.97
    a_tokens = normalized_match_tokens(a)
    b_tokens = normalized_match_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = a_tokens & b_tokens
    token_score = 0.0
    if len(overlap) >= 3:
        token_score = 0.92
    elif len(overlap) >= 2 and len(overlap) / max(len(a_tokens), 1) >= 0.5:
        token_score = 0.88
    anchor_score = EQUIPMENT_ANCHOR_MATCH_SCORE if shares_equipment_anchor(task_name, mdr_title) else 0.0
    return max(token_score, anchor_score)


def title_match_qualifies(task_name: str, mdr_title: str, score: float) -> bool:
    if drainage_subtype_conflict(task_name, mdr_title):
        return False
    if document_type_conflict(task_name, mdr_title):
        return False
    if facility_identity_conflict(task_name, mdr_title):
        return False
    if equipment_modifier_conflict(task_name, mdr_title):
        return False
    if equipment_scope_narrowing_conflict(task_name, mdr_title):
        return False
    if descriptor_mismatch_conflict(task_name, mdr_title):
        return False
    if score < TITLE_MATCH_MIN_SCORE:
        return False
    if score >= TITLE_MATCH_STRONG_SCORE:
        return True
    return not title_has_hard_subject_conflict(task_name, mdr_title, score)


def exact_title_match_strong_enough(task_name: str, mdr_title: str, score: float) -> bool:
    """True for near-duplicate titles or equipment-anchor matches, not token-overlap-only."""
    if not title_match_qualifies(task_name, mdr_title, score):
        return False
    if score >= TITLE_MATCH_STRONG_SCORE:
        return True
    if score >= EQUIPMENT_ANCHOR_MATCH_SCORE and shares_equipment_anchor(task_name, mdr_title):
        return True
    return False


def _title_match_rank_key(task_name: str, mdr_title: str, score: float):
    task_tokens = normalized_match_tokens(task_subject_for_match(task_name))
    mdr_tokens = normalized_match_tokens(str(mdr_title or ""))
    subject_overlap = len(task_tokens & mdr_tokens)
    return (score, subject_overlap, -len(mdr_tokens - task_tokens))


def find_best_title_match(candidates, task_name: str, id_col=None, mdr_title_col="MdrDocumentTitle"):
    best_id = None
    best_score = 0.0
    best_rank_key = None
    best_title = ""
    for idx, row in candidates.iterrows():
        row_titles = _candidate_match_titles(row, mdr_title_col)
        row_score = 0.0
        row_title = ""
        row_rank_key = None
        for title in row_titles:
            score = mdr_title_match_score(task_name, title)
            if not title_match_qualifies(task_name, title, score):
                continue
            rank_key = _title_match_rank_key(task_name, title, score)
            if row_rank_key is None or rank_key > row_rank_key:
                row_rank_key = rank_key
                row_score = score
                row_title = title
        if row_rank_key is None:
            continue
        if best_rank_key is None or row_rank_key > best_rank_key:
            best_rank_key = row_rank_key
            best_score = row_score
            best_id = int(row[id_col]) if id_col else int(idx)
            best_title = row_title
    if best_id is None or best_score < TITLE_MATCH_MIN_SCORE:
        return None, 0.0
    return best_id, best_score


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


def llm_supports_custom_temperature(model: str) -> bool:
    """Some OpenAI models only accept the default temperature (1)."""
    m = (model or "").strip().lower()
    if not m:
        return True
    if m.startswith(("o1", "o3", "o4")):
        return False
    if "gpt-5" in m:
        return False
    return True


def llm_temperature_kwargs(model: str, temperature=0):
    if llm_supports_custom_temperature(model):
        return {"temperature": temperature}
    return {}


def build_chat_completion_body(model: str, messages, **extra):
    body = {"model": model, "messages": messages}
    body.update(llm_temperature_kwargs(model))
    body.update(extra)
    return body


def chat_json(cfg, system, user, timeout=60):
    api_key = cfg.get("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("LLM_API_KEY mancante in config.txt")
    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    base_url = cfg.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = build_chat_completion_body(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
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


def embed_texts(cfg, texts, batch_size=256, timeout=180, retry_max=3, retry_backoff=2.0):
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
        last_err = None
        for attempt in range(retry_max + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                last_err = None
                break
            except (TimeoutError, urllib.error.URLError) as exc:
                last_err = exc
                if attempt >= retry_max:
                    raise
                wait_s = retry_backoff * (attempt + 1)
                print(
                    f"[embeddings] batch {i // batch_size + 1} timeout/rete "
                    f"(tentativo {attempt + 1}/{retry_max + 1}), attendo {wait_s:.0f}s..."
                )
                time.sleep(wait_s)
        if last_err is not None:
            raise last_err
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
