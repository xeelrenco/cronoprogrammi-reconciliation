import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from timeline_reconciliation_common import (
    CONFIG_FILE,
    OUTPUT_DIR,
    build_chat_completion_body,
    build_link_date_snapshot,
    chat_json,
    connect_motherduck,
    find_best_title_match,
    mdr_title_match_score,
    normalize,
    parse_config_txt,
    refresh_timeline_classified_dates_view,
    refresh_timeline_links_dates_view,
    raci_dedupe_key,
    remove_prefix,
    serialize_date_value,
    task_subject_for_match,
    link_subject_gates_block,
    significant_tokens,
)


CREATED_BY = "4_resolve_timeline_task_mdr_links.py"
LINK_METHOD = "embedding_topk_llm_resolver"
LINK_METHOD_EXACT = "exact_mdr_title_match"
LINK_METHOD_FALLBACK = "llm_resolver_rank1_fallback"
LINK_METHOD_TOP_CANDIDATE_FALLBACK = "llm_top_candidate_fallback"
MIN_TOP_CANDIDATE_FALLBACK_CONFIDENCE = 0.60
MIN_TOP_CANDIDATE_FALLBACK_SIMILARITY = 0.63
DEFAULT_LLM_SHORTLIST_MAX = 5
LLM_SHORTLIST_HARD_MAX = 5
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


def _subject_overlap(task_name: str, mdr_title: str, min_shared: int = 2) -> bool:
    task_tokens = set(significant_tokens(task_subject_for_match(task_name)))
    mdr_tokens = set(significant_tokens(mdr_title))
    if not task_tokens or not mdr_tokens:
        return False
    shared = task_tokens & mdr_tokens
    if len(shared) >= min_shared:
        return True
    return len(shared) >= 1 and len(shared) / max(len(task_tokens), 1) >= 0.5


def task_lists_document_bundle(task_name: str) -> bool:
    clean = remove_prefix(str(task_name or "")).lower()
    if " / " not in clean:
        return False
    parts = [p.strip() for p in clean.split(" / ")]
    hits = sum(1 for part in parts if any(kw in part for kw in BUNDLE_DOC_TYPE_KEYWORDS))
    return hits >= 2


def _find_best_exact_match_candidate(task_group, task_name: str):
    best_id, best_score = find_best_title_match(
        task_group,
        task_name,
        id_col="RetrievalRank",
    )
    return best_id, best_score


def _link_blocked_by_subject_conflict(task_name: str, mdr_title: str) -> bool:
    return link_subject_gates_block(task_name, mdr_title)


def _link_mdr_title(row) -> str:
    if row is None:
        return ""
    return str(row.get("MdrDocumentTitle", "") or "")


def _fallback_llm_rank1_link(llm_links, task_group, task_name: str):
    if not llm_links:
        return []
    rank_by_id = _rank_by_retrieval_id(task_group)
    rank1 = dict(llm_links[0])
    row = rank_by_id.get(int(rank1["candidate_id"]))
    mdr = _link_mdr_title(row)
    if mdr and _link_blocked_by_subject_conflict(task_name, mdr):
        return []
    fallback = dict(rank1)
    fallback["link_method"] = LINK_METHOD_FALLBACK
    if not str(fallback.get("reason_short", "")).strip():
        fallback["reason_short"] = "Auto: LLM rank 1 fallback"
    return [fallback]


def _min_confidence_for_link(link, min_link_confidence: float) -> float:
    method = str(link.get("link_method") or "")
    if method == LINK_METHOD_TOP_CANDIDATE_FALLBACK:
        return MIN_TOP_CANDIDATE_FALLBACK_CONFIDENCE
    return min_link_confidence


def _promote_top_candidate_fallback(resolved, task_group, task_name: str):
    top_candidates = resolved.get("top_candidates") or []
    if not top_candidates:
        return None
    rank_by_id = _rank_by_retrieval_id(task_group)
    for item in top_candidates:
        try:
            candidate_id = int(item["candidate_id"])
        except Exception:
            continue
        row = rank_by_id.get(candidate_id)
        if row is None:
            continue
        mdr = _link_mdr_title(row)
        if not mdr or _link_blocked_by_subject_conflict(task_name, mdr):
            continue
        similarity = safe_float(row.get("Similarity", 0.0))
        confidence = safe_float(item.get("confidence", 0.0))
        if similarity < MIN_TOP_CANDIDATE_FALLBACK_SIMILARITY:
            continue
        if confidence < MIN_TOP_CANDIDATE_FALLBACK_CONFIDENCE:
            continue
        reason = str(item.get("why_plausible", "") or "").strip()
        if not reason:
            reason = "Auto: promoted from LLM top_candidates shortlist"
        return {
            "candidate_id": candidate_id,
            "confidence": confidence,
            "reason_short": reason[:300],
            "link_method": LINK_METHOD_TOP_CANDIDATE_FALLBACK,
        }
    return None


def _filter_links_by_subject(links, task_group, task_name: str):
    if not links:
        return links
    rank_by_id = _rank_by_retrieval_id(task_group)
    kept = []
    for idx, link in enumerate(links):
        cid = int(link["candidate_id"])
        row = rank_by_id.get(cid)
        mdr = str(row.get("MdrDocumentTitle", "")) if row is not None else ""
        if mdr and _link_blocked_by_subject_conflict(task_name, mdr):
            continue
        if idx == 0 or _subject_overlap(task_name, mdr):
            kept.append(link)
    return kept


def refine_resolver_links(resolved, task_group):
    if resolved.get("status") != "ok":
        return resolved
    first = task_group.iloc[0]
    task_name = str(first.get("TaskName", ""))
    is_bundle = task_lists_document_bundle(task_name)
    llm_links_original = list(resolved.get("links") or [])
    exact_id, exact_score = _find_best_exact_match_candidate(task_group, task_name)
    links = list(llm_links_original)
    rank_by_id = _rank_by_retrieval_id(task_group)

    if exact_id is not None and not is_bundle:
        exact_link = None
        for link in links:
            if int(link["candidate_id"]) == exact_id:
                exact_link = dict(link)
                break
        if exact_link is None:
            exact_link = {
                "candidate_id": exact_id,
                "confidence": max(0.97, exact_score),
                "reason_short": "Auto: exact MDR title match",
                "link_method": LINK_METHOD_EXACT,
            }
        else:
            exact_link = dict(exact_link)
            exact_link["confidence"] = max(safe_float(exact_link.get("confidence")), 0.97, exact_score)
            exact_link["link_method"] = LINK_METHOD_EXACT
            if not str(exact_link.get("reason_short", "")).strip():
                exact_link["reason_short"] = "Auto: exact MDR title match"
        links = [exact_link]
    elif exact_id is not None and is_bundle:
        has_exact = any(int(x["candidate_id"]) == exact_id for x in links)
        if not has_exact:
            links.insert(
                0,
                {
                    "candidate_id": exact_id,
                    "confidence": max(0.97, exact_score),
                    "reason_short": "Auto: exact MDR title match",
                    "link_method": LINK_METHOD_EXACT,
                },
            )
        links = _filter_links_by_subject(links, task_group, task_name)
    else:
        links = _filter_links_by_subject(links, task_group, task_name)
        if not is_bundle and links:
            rank1 = links[0]
            row = rank_by_id.get(int(rank1["candidate_id"]))
            mdr = _link_mdr_title(row)
            if mdr and _link_blocked_by_subject_conflict(task_name, mdr):
                alt_id, alt_score = _find_best_exact_match_candidate(task_group, task_name)
                if alt_id is not None:
                    alt_link = next((dict(x) for x in links if int(x["candidate_id"]) == alt_id), None)
                    if alt_link is None:
                        alt_link = {
                            "candidate_id": alt_id,
                            "confidence": max(0.88, alt_score),
                            "reason_short": "Auto: subject token title match",
                            "link_method": LINK_METHOD_EXACT,
                        }
                    links = [alt_link]
                else:
                    links = _fallback_llm_rank1_link(llm_links_original, task_group, task_name)

    if not links and not is_bundle:
        links = _fallback_llm_rank1_link(llm_links_original, task_group, task_name)

    deduped = []
    seen_keys = set()
    for link in links:
        cid = int(link["candidate_id"])
        row = rank_by_id.get(cid)
        mdr = _link_mdr_title(row)
        if mdr and _link_blocked_by_subject_conflict(task_name, mdr):
            continue
        dkey = _raci_key_from_row(row) if row is not None else f"cid:{cid}"
        if dkey in seen_keys:
            continue
        seen_keys.add(dkey)
        deduped.append(link)

    if not deduped and not is_bundle:
        promoted = _promote_top_candidate_fallback(resolved, task_group, task_name)
        if promoted is not None:
            deduped = [promoted]

    out = dict(resolved)
    out["links"] = deduped
    out["valid_links_count"] = len(deduped)
    return out


def _raci_key_from_row(row) -> str:
    key = raci_dedupe_key(
        row.get("ConsolidatedTitleKey"),
        row.get("MdrTitleKey"),
        row.get("ConsolidatedRaciTitle"),
        row.get("MdrDocumentTitle"),
    )
    if key:
        return key
    try:
        return f"cid:{int(row['RetrievalRank'])}"
    except Exception:
        return "cid:unknown"


def _rank_by_retrieval_id(task_group):
    return {int(r["RetrievalRank"]): r for _, r in task_group.iterrows()}


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
            t.EarlyStartDate,
            t.EarlyEndDate,
            t.ActualStartDate,
            t.ActualEndDate,
            t.TargetStartDate,
            t.TargetEndDate,
            t.TaskClass,
            t.TaskClassConfidence,
            t.TaskClassReason,
            k.MdrDocumentTitle,
            k.MdrTitleKey,
            k.ConsolidatedTitleKey,
            k.ConsolidatedRaciTitle,
            k.EmbeddingModel,
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


def validate_resolver_output(parsed, task_group, max_shortlist=DEFAULT_LLM_SHORTLIST_MAX):
    max_shortlist = max(0, min(int(max_shortlist), LLM_SHORTLIST_HARD_MAX))
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

    out_links = sorted(
        best_by_candidate.values(),
        key=lambda x: (-safe_float(x.get("confidence", 0.0)), int(x["candidate_id"])),
    )
    rank_by_id = _rank_by_retrieval_id(task_group)
    seen_link_keys = set()
    deduped_links = []
    for link in out_links:
        row = rank_by_id.get(int(link["candidate_id"]))
        dkey = _raci_key_from_row(row) if row is not None else f"cid:{link['candidate_id']}"
        if dkey in seen_link_keys:
            duplicate_candidate_count += 1
            continue
        seen_link_keys.add(dkey)
        deduped_links.append(link)
    out_links = deduped_links

    top_candidates = []
    raw_top_candidates_count = 0
    dropped_invalid_top_count = 0
    duplicate_top_candidates_count = 0

    if max_shortlist > 0:
        raw_top = parsed.get("top_candidates")
        if raw_top is None:
            raw_top = []
        if not isinstance(raw_top, list):
            return _invalid_result(
                "invalid_top_candidates",
                "LLM response field top_candidates is not a list",
            )
        raw_top_candidates_count = len(raw_top)
        seen_tc = set()
        seen_top_keys = set()
        for item in raw_top:
            if len(top_candidates) >= max_shortlist:
                break
            if not isinstance(item, dict):
                dropped_invalid_top_count += 1
                continue
            try:
                cid = int(item.get("candidate_id"))
            except Exception:
                dropped_invalid_top_count += 1
                continue
            if cid not in valid_ids:
                dropped_invalid_top_count += 1
                continue
            if cid in seen_tc:
                duplicate_top_candidates_count += 1
                continue
            row = rank_by_id.get(cid)
            dkey = _raci_key_from_row(row) if row is not None else f"cid:{cid}"
            if dkey in seen_top_keys:
                duplicate_top_candidates_count += 1
                continue
            seen_tc.add(cid)
            seen_top_keys.add(dkey)
            top_candidates.append(
                {
                    "candidate_id": cid,
                    "confidence": clamp01(item.get("confidence", 0.0)),
                    "why_plausible": str(item.get("why_plausible", "") or "")[:500],
                }
            )
        if len(top_candidates) < max_shortlist:
            pool = task_group.sort_values(
                ["Similarity", "RetrievalRank"],
                ascending=[False, True],
            )
            for _, row in pool.iterrows():
                if len(top_candidates) >= max_shortlist:
                    break
                cid = int(row["RetrievalRank"])
                if cid in seen_tc:
                    continue
                dkey = _raci_key_from_row(row)
                if dkey in seen_top_keys:
                    continue
                seen_tc.add(cid)
                seen_top_keys.add(dkey)
                top_candidates.append(
                    {
                        "candidate_id": cid,
                        "confidence": clamp01(row.get("Similarity", 0.0)),
                        "why_plausible": "",
                    }
                )

    return {
        "status": "ok",
        "links": out_links,
        "top_candidates": top_candidates,
        "error_type": "",
        "error_message": "",
        "raw_links_count": len(links),
        "valid_links_count": len(out_links),
        "dropped_invalid_count": dropped_invalid_count,
        "duplicate_candidate_count": duplicate_candidate_count,
        "raw_top_candidates_count": raw_top_candidates_count,
        "valid_top_candidates_count": len(top_candidates),
        "dropped_invalid_top_count": dropped_invalid_top_count,
        "duplicate_top_candidates_count": duplicate_top_candidates_count,
    }


def _invalid_result(error_type, error_message):
    return {
        "status": "invalid_json",
        "links": [],
        "top_candidates": [],
        "error_type": error_type,
        "error_message": error_message,
        "raw_links_count": 0,
        "valid_links_count": 0,
        "dropped_invalid_count": 0,
        "duplicate_candidate_count": 0,
        "raw_top_candidates_count": 0,
        "valid_top_candidates_count": 0,
        "dropped_invalid_top_count": 0,
        "duplicate_top_candidates_count": 0,
    }


def build_resolver_prompts(task_group, cfg=None):
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

Rule 1 is MANDATORY and blocking: if the MDR document title does not share
at least one specific subject concept with task_name_clean (beyond generic words
like document, drawing, specification, system, engineering), do NOT link
regardless of embedding similarity or discipline match.

Document type blocking rule:
When task_name_clean contains a highly specific drawing/document type descriptor
(e.g. cross sectional, general arrangement, isometric, wiring diagram, hook-up,
single line diagram, part list, bill of materials), that specific descriptor —
or a direct synonym or translation — MUST be present in the MDR candidate title.
Matching only on equipment name or system name is NOT sufficient.

This rule does NOT apply to generic document types such as:
report, diagram, data sheet, specification, sheet, plan, layout, list, schedule.
For these, a strong subject/equipment name match is sufficient to allow the link.

Multilingual equivalence: treat the following Italian–English pairs as equivalent
when comparing task_name_clean to MDR candidate titles:
- planimetria / plan
- sezioni / sections
- percorso / routing
- schema / diagram
- relazione / report
- foglio dati / data sheet
- capitolato / specification
- computo / bill of materials

When a task name explicitly lists multiple document types separated by "/" or "and"
(e.g. "Specification / Data Sheet / IDS"), treat each type independently.
Link each candidate that matches one of the listed document types — do not require
a single candidate to cover all types. Each linked candidate must still share the same
specific subject/equipment as the task (e.g. MV Cables, not switchgear).

Rank 1 priority rule:
If one candidate's mdr_document_title is an exact or near-exact match to task_name_clean
(same subject and document scope), that candidate MUST be link rank 1 with highest confidence.
Prefer the specific MDR document title over a generic RACI category title.

Secondary links (rank 2+):
Add extra links ONLY when the task is a document bundle (multiple types in the name)
OR when the additional candidate shares the same specific subject tokens as the task.
Do NOT add secondary links that match only a generic document type (IDS, specification, layout)
on a different subject (e.g. task "Bolts and Nuts" must not link IDS for "Fittings").

Multiple links are allowed when the task clearly covers a bundle/group of documents,
not merely because several candidates are semantically nearby in the same discipline.

If task_class_confidence is LOW, be extra conservative and prefer no links.
If uncertain, return:
{"links": [], "top_candidates": []}

Industry acronym equivalence: treat the following as strong title matches
when they appear in both the task and the candidate:
- MCC / PMCC / MCC-P (Motor Control Center variants)
- MV / MT (Medium Voltage)
- LV / BT (Low Voltage)
- HV / AT (High Voltage)
- P&ID / PID
More generally: if two titles differ only by an industry-standard prefix or
abbreviation variant referring to the same equipment type, consider them equivalent.

Also return top_candidates: an ordered shortlist of up to 5 distinct candidates (best first)
that are the most semantically plausible matches from the provided list — even when links is empty.
Use candidate_id values exactly as in the candidates list. At most one candidate_id per consolidated
RACI document (same raci_title / same document group — do not list multiple ids for the same RACI title).
Each entry needs confidence (0-1) and why_plausible (brief, factual). Do not invent candidates.
If nothing is plausible, use [].

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
  ],
  "top_candidates": [
    {
      "candidate_id": 1,
      "confidence": 0.0,
      "why_plausible": "brief factual justification in English"
    }
  ]
}
"""
    selected_dates = build_link_date_snapshot(first)
    user = {
        "task": {
            "task_code": str(first.get("TaskCode", "")),
            "task_name": str(first.get("TaskName", "")),
            "task_name_clean": remove_prefix(first.get("TaskName", "")),
            "wbs_name": str(first.get("WbsName", "")),
            "selected_start_date": str(serialize_date_value(selected_dates.get("SelectedStartDate")) or ""),
            "selected_finish_date": str(serialize_date_value(selected_dates.get("SelectedFinishDate")) or ""),
            "task_class_reason": str(first.get("TaskClassReason", "")),
            "task_class_confidence": str(first.get("TaskClassConfidence", "")),
        },
        "candidates": candidates,
    }
    return system, user


def resolve_task_links(
    task_group,
    cfg,
    llm_timeout_sec=60,
    retry_max=0,
    retry_backoff_sec=2.0,
    llm_shortlist_max=DEFAULT_LLM_SHORTLIST_MAX,
):
    system, user = build_resolver_prompts(task_group, cfg=cfg)
    last_error = None
    for attempt in range(max(0, retry_max) + 1):
        try:
            parsed = chat_json(cfg, system, user, timeout=llm_timeout_sec)
            return validate_resolver_output(parsed, task_group, max_shortlist=llm_shortlist_max)
        except Exception as exc:
            last_error = exc
            if attempt < retry_max:
                time.sleep(max(0.0, retry_backoff_sec) * (attempt + 1))
    return {
        "status": "llm_error",
        "links": [],
        "top_candidates": [],
        "error_type": type(last_error).__name__ if last_error else "llm_error",
        "error_message": str(last_error or "LLM call failed")[:500],
        "raw_links_count": 0,
        "valid_links_count": 0,
        "dropped_invalid_count": 0,
        "duplicate_candidate_count": 0,
        "raw_top_candidates_count": 0,
        "valid_top_candidates_count": 0,
        "dropped_invalid_top_count": 0,
        "duplicate_top_candidates_count": 0,
    }


def build_final_rows_for_group(
    timeline_name,
    task_row_id,
    group,
    resolved,
    min_link_confidence=0.0,
    max_links_per_task=0,
    save_llm_top_max=DEFAULT_LLM_SHORTLIST_MAX,
    cfg=None,
):
    rows = []
    llm_shortlist_rows = []
    first = group.iloc[0]
    date_snapshot = build_link_date_snapshot(first)
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
        resolved = refine_resolver_links(resolved, group)
        status = resolved.get("status", status)
        scope_ok = {"TimelineName": timeline_name, "TaskRowId": int(task_row_id)}
        selected = resolved.get("links", [])
        before_threshold = len(selected)
        selected = [
            x
            for x in selected
            if safe_float(x.get("confidence", 0.0)) >= _min_confidence_for_link(x, min_link_confidence)
        ]
        dropped_by_threshold = before_threshold - len(selected)
        similarity_by_id = {
            int(row["RetrievalRank"]): safe_float(row.get("Similarity", 0.0))
            for _, row in group.iterrows()
        }

        def _link_sort_key(link):
            method = str(link.get("link_method") or "")
            if method == LINK_METHOD_EXACT:
                tier = 0
            elif method == LINK_METHOD_TOP_CANDIDATE_FALLBACK:
                tier = 1
            elif method == LINK_METHOD_FALLBACK:
                tier = 2
            else:
                tier = 3
            return (
                tier,
                -safe_float(link.get("confidence", 0.0)),
                -similarity_by_id.get(int(link["candidate_id"]), 0.0),
                int(link["candidate_id"]),
            )

        selected = sorted(selected, key=_link_sort_key)
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
                    **date_snapshot,
                    "TaskClass": cand.get("TaskClass"),
                    "TaskClassConfidence": cand.get("TaskClassConfidence"),
                    "TaskClassReason": cand.get("TaskClassReason"),
                    "MdrDocumentTitle": cand.get("MdrDocumentTitle"),
                    "MdrTitleKey": cand.get("MdrTitleKey"),
                    "LinkRank": link_rank,
                    "LinkScore": link["confidence"],
                    "LinkMethod": link.get("link_method") or LINK_METHOD,
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

        if save_llm_top_max > 0:
            created_ts = datetime.now(timezone.utc).replace(tzinfo=None)
            emb = str(first.get("EmbeddingModel") or "")
            for rank, tc in enumerate((resolved.get("top_candidates") or [])[:save_llm_top_max], 1):
                cid = int(tc["candidate_id"])
                match = group[group["RetrievalRank"] == cid]
                if match.empty:
                    continue
                cand = match.iloc[0]
                llm_shortlist_rows.append(
                    {
                        "TimelineName": cand["TimelineName"],
                        "ProjectCode": cand.get("ProjectCode"),
                        "TaskRowId": int(cand["TaskRowId"]),
                        "EmbeddingModel": emb,
                        "CandidateRankWithinResolver": rank,
                        "RetrievalRank": cid,
                        "MdrDocumentTitle": cand.get("MdrDocumentTitle"),
                        "MdrTitleKey": cand.get("MdrTitleKey"),
                        "ConsolidatedTitleKey": cand.get("ConsolidatedTitleKey"),
                        "ConsolidatedRaciTitle": cand.get("ConsolidatedRaciTitle"),
                        "RetrievalSimilarity": safe_float(cand.get("Similarity")),
                        "LlmConfidence": safe_float(tc.get("confidence")),
                        "WhyPlausible": str(tc.get("why_plausible", "") or "")[:500],
                        "EffectiveDescription": cand.get("EffectiveDescription"),
                        "DisciplineName": cand.get("DisciplineName"),
                        "TypeName": cand.get("TypeName"),
                        "CategoryDescription": cand.get("CategoryDescription"),
                        "ChapterName": cand.get("ChapterName"),
                        "CreatedAt": created_ts,
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
    return rows, diagnostic, scope_all, scope_ok, status, llm_shortlist_rows


def process_group_realtime(
    item,
    cfg,
    min_link_confidence,
    max_links_per_task,
    llm_timeout_sec,
    retry_max,
    retry_backoff_sec,
    save_llm_top_max=DEFAULT_LLM_SHORTLIST_MAX,
):
    (timeline_name, task_row_id), group = item
    resolved = resolve_task_links(
        group,
        cfg,
        llm_timeout_sec=llm_timeout_sec,
        retry_max=retry_max,
        retry_backoff_sec=retry_backoff_sec,
        llm_shortlist_max=save_llm_top_max,
    )
    return build_final_rows_for_group(
        timeline_name,
        task_row_id,
        group,
        resolved,
        min_link_confidence=min_link_confidence,
        max_links_per_task=max_links_per_task,
        save_llm_top_max=save_llm_top_max,
        cfg=cfg,
    )


def combine_group_results(group_results):
    rows = []
    diagnostics = []
    resolved_scope_all = []
    resolved_scope_ok = []
    llm_top_rows = []
    status_counts = {"ok": 0, "llm_error": 0, "invalid_json": 0}
    for group_rows, diagnostic, scope_all, scope_ok, status, shortlist in group_results:
        rows.extend(group_rows)
        diagnostics.append(diagnostic)
        resolved_scope_all.append(scope_all)
        if scope_ok is not None:
            resolved_scope_ok.append(scope_ok)
        llm_top_rows.extend(shortlist)
        status_counts[status] = status_counts.get(status, 0) + 1
    return (
        pd.DataFrame(rows),
        pd.DataFrame(diagnostics),
        pd.DataFrame(resolved_scope_all),
        pd.DataFrame(resolved_scope_ok),
        pd.DataFrame(llm_top_rows),
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
    save_llm_top_max=DEFAULT_LLM_SHORTLIST_MAX,
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
                    save_llm_top_max,
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
                    save_llm_top_max,
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
    system, user = build_resolver_prompts(task_group, cfg=cfg)
    body = build_chat_completion_body(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
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


def collect_batch_results(
    topk,
    cfg,
    min_link_confidence,
    max_links_per_task,
    poll_interval_sec,
    llm_shortlist_max=DEFAULT_LLM_SHORTLIST_MAX,
):
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
                resolved = validate_resolver_output(parsed, group, max_shortlist=llm_shortlist_max)
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
                    save_llm_top_max=llm_shortlist_max,
                    cfg=cfg,
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
            "top_candidates": [],
            "error_type": "missing_batch_result",
            "error_message": "No output row found for this batch custom_id",
            "raw_links_count": 0,
            "valid_links_count": 0,
            "dropped_invalid_count": 0,
            "duplicate_candidate_count": 0,
            "raw_top_candidates_count": 0,
            "valid_top_candidates_count": 0,
            "dropped_invalid_top_count": 0,
            "duplicate_top_candidates_count": 0,
        }
        group_results.append(
            build_final_rows_for_group(
                key[0],
                key[1],
                group,
                resolved,
                min_link_confidence=min_link_confidence,
                max_links_per_task=max_links_per_task,
                save_llm_top_max=llm_shortlist_max,
                cfg=cfg,
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


def ensure_link_schedule_columns(conn, db_name):
    for col, col_type in (
        ("EarlyStartDate", "TIMESTAMP"),
        ("EarlyEndDate", "TIMESTAMP"),
        ("ActualStartDate", "TIMESTAMP"),
        ("ActualEndDate", "TIMESTAMP"),
        ("TargetStartDate", "TIMESTAMP"),
        ("TargetEndDate", "TIMESTAMP"),
        ("SelectedStartDate", "TIMESTAMP"),
        ("SelectedFinishDate", "TIMESTAMP"),
    ):
        conn.execute(
            f"""
            ALTER TABLE {db_name}.timeline_reconciliation.TimelineTaskToMdrLinks
            ADD COLUMN IF NOT EXISTS {col} {col_type}
            """
        )


def save_final_links(conn, db_name, rows, resolved_scope):
    if resolved_scope.empty:
        return 0
    ensure_link_schedule_columns(conn, db_name)
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
                        EarlyStartDate, EarlyEndDate, ActualStartDate, ActualEndDate,
                        TargetStartDate, TargetEndDate,
                        SelectedStartDate, SelectedFinishDate,
                        TaskClass, TaskClassConfidence, TaskClassReason,
                        MdrDocumentTitle, MdrTitleKey, LinkRank, LinkScore, LinkMethod, LinkReason,
                        ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                        ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource, CreatedBy
                    )
                    SELECT
                        TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                        EarlyStartDate, EarlyEndDate, ActualStartDate, ActualEndDate,
                        TargetStartDate, TargetEndDate,
                        SelectedStartDate, SelectedFinishDate,
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
        if inserted:
            refresh_timeline_classified_dates_view(conn, db_name)
            refresh_timeline_links_dates_view(conn, db_name)
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.unregister("resolved_scope")
    return inserted


def ensure_resolver_llm_top_candidates_table(conn, db_name):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {db_name}.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates (
            TimelineName VARCHAR NOT NULL,
            ProjectCode VARCHAR,
            TaskRowId BIGINT NOT NULL,
            EmbeddingModel VARCHAR NOT NULL,
            CandidateRankWithinResolver INTEGER NOT NULL,
            RetrievalRank INTEGER NOT NULL,
            MdrDocumentTitle VARCHAR,
            MdrTitleKey VARCHAR,
            ConsolidatedTitleKey VARCHAR,
            ConsolidatedRaciTitle VARCHAR,
            RetrievalSimilarity DOUBLE,
            LlmConfidence DOUBLE,
            WhyPlausible VARCHAR,
            EffectiveDescription VARCHAR,
            DisciplineName VARCHAR,
            TypeName VARCHAR,
            CategoryDescription VARCHAR,
            ChapterName VARCHAR,
            CreatedAt TIMESTAMP NOT NULL,
            CreatedBy VARCHAR,
            PRIMARY KEY (TimelineName, TaskRowId, EmbeddingModel, CandidateRankWithinResolver)
        );
        """
    )


def save_resolver_llm_top_candidates(conn, db_name, rows, task_scope, embedding_model):
    if task_scope.empty:
        return 0
    conn.register("resolver_llm_top_scope", task_scope)
    try:
        conn.execute("BEGIN;")
        conn.execute(
            f"""
            DELETE FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates t
            USING resolver_llm_top_scope s
            WHERE t.TimelineName = s.TimelineName
              AND t.TaskRowId = s.TaskRowId
              AND t.EmbeddingModel = ?
            """,
            [embedding_model],
        )
        inserted = 0
        if not rows.empty:
            conn.register("resolver_llm_top_rows", rows)
            try:
                conn.execute(
                    f"""
                    INSERT INTO {db_name}.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates (
                        TimelineName, ProjectCode, TaskRowId, EmbeddingModel,
                        CandidateRankWithinResolver, RetrievalRank,
                        MdrDocumentTitle, MdrTitleKey, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                        RetrievalSimilarity, LlmConfidence, WhyPlausible,
                        EffectiveDescription,
                        DisciplineName, TypeName, CategoryDescription, ChapterName,
                        CreatedAt, CreatedBy
                    )
                    SELECT
                        TimelineName, ProjectCode, TaskRowId, EmbeddingModel,
                        CandidateRankWithinResolver, RetrievalRank,
                        MdrDocumentTitle, MdrTitleKey, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                        RetrievalSimilarity, LlmConfidence, WhyPlausible,
                        EffectiveDescription,
                        DisciplineName, TypeName, CategoryDescription, ChapterName,
                        CreatedAt, CreatedBy
                    FROM resolver_llm_top_rows
                    """
                )
                inserted = len(rows)
            finally:
                conn.unregister("resolver_llm_top_rows")
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.unregister("resolver_llm_top_scope")
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
    parser.add_argument(
        "--save-llm-top-max",
        type=int,
        default=DEFAULT_LLM_SHORTLIST_MAX,
        help=(
            "Persisti in DB la shortlist top_candidates del LLM: da 0 a 5 voci per task (0=disattiva salvataggio)."
        ),
    )
    args = parser.parse_args()
    if not (0 <= args.save_llm_top_max <= LLM_SHORTLIST_HARD_MAX):
        raise RuntimeError(
            f"--save-llm-top-max deve essere tra 0 e {LLM_SHORTLIST_HARD_MAX} (inclusi)."
        )
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
            (
                final_links,
                diagnostics,
                resolved_scope_all,
                resolved_scope_ok,
                llm_top_df,
                status_counts,
            ) = collect_batch_results(
                topk,
                cfg,
                args.min_link_confidence,
                args.max_links_per_task,
                args.batch_poll_interval,
                llm_shortlist_max=args.save_llm_top_max,
            )
        else:
            (
                final_links,
                diagnostics,
                resolved_scope_all,
                resolved_scope_ok,
                llm_top_df,
                status_counts,
            ) = build_final_links(
                topk,
                cfg,
                args.progress_every,
                min_link_confidence=args.min_link_confidence,
                max_links_per_task=args.max_links_per_task,
                llm_timeout_sec=args.llm_timeout_sec,
                retry_max=args.retry_max,
                retry_backoff_sec=args.retry_backoff_sec,
                workers=args.workers,
                save_llm_top_max=args.save_llm_top_max,
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
        if args.save_llm_top_max > 0:
            ensure_resolver_llm_top_candidates_table(conn, db_name)
            n_llm_top = save_resolver_llm_top_candidates(
                conn,
                db_name,
                llm_top_df,
                resolved_scope_ok.drop_duplicates(),
                embedding_model,
            )
            print(f"Resolver LLM top_candidates rows saved: {n_llm_top}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
