import argparse
import time

import pandas as pd

from timeline_reconciliation_common import CONFIG_FILE, chat_json, connect_motherduck, parse_config_txt, remove_prefix


CREATED_BY = "4_judge_timeline_task_mdr_links.py"
LINK_METHOD = "embedding_topk_llm_judge"


def load_topk_for_judge(conn, db_name, embedding_model, timeline_name=None, top_k=30):
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
        )
        SELECT
            k.TimelineName,
            k.ProjectCode,
            k.TaskRowId,
            t.TaskCode,
            k.TaskName,
            k.WbsName,
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
        FROM ranked k
        JOIN {db_name}.timeline_reconciliation.TimelineTasksClassified t
          ON t.TimelineName = k.TimelineName
         AND t.TaskRowId = k.TaskRowId
        LEFT JOIN {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings c
          ON c.TimelineName = k.TimelineName
         AND c.MdrDocumentTitle = k.MdrDocumentTitle
         AND c.ConsolidatedTitleKey = k.ConsolidatedTitleKey
         AND c.EmbeddingModel = k.EmbeddingModel
        ORDER BY k.TimelineName, k.TaskRowId, k.Rank
        """,
        params,
    ).fetchdf()


def judge_task_links(task_group, cfg):
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
You judge whether a Primavera ENG_DOC task should be linked to one or more MDR documents.

Return ONLY valid JSON.

Select zero, one, or multiple candidates from the provided semantic Top-K retrieval set.
Do not force a match. Return an empty links array when no MDR candidate is credible.

Use MDR title as the primary link target. Use RACI title, description, and metadata only as
semantic context. A task may link to multiple MDR documents when it clearly covers a group
or package of document deliverables.

JSON schema:
{
  "links": [
    {
      "candidate_id": 1,
      "confidence": 0.0,
      "reason_short": "brief reason in Italian"
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
            "task_class_reason": str(first.get("TaskClassReason", "")),
        },
        "candidates": candidates,
    }
    try:
        parsed = chat_json(cfg, system, user, timeout=60)
    except Exception:
        return []
    links = parsed.get("links", [])
    if not isinstance(links, list):
        return []
    valid_ids = set(task_group["RetrievalRank"].astype(int).tolist())
    out = []
    for link in links:
        try:
            candidate_id = int(link.get("candidate_id"))
        except Exception:
            continue
        if candidate_id not in valid_ids:
            continue
        confidence = max(0.0, min(1.0, float(link.get("confidence", 0.0))))
        out.append(
            {
                "candidate_id": candidate_id,
                "confidence": confidence,
                "reason_short": str(link.get("reason_short", ""))[:300],
            }
        )
    return out


def build_final_links(topk, cfg, progress_every):
    rows = []
    groups = list(topk.groupby(["TimelineName", "TaskRowId"], sort=True))
    started = time.time()
    for idx, ((_, _), group) in enumerate(groups, 1):
        selected = judge_task_links(group, cfg)
        selected = sorted(selected, key=lambda x: x["confidence"], reverse=True)
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
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(groups)):
            print(f"Judged {idx}/{len(groups)} tasks (elapsed {round(time.time() - started, 1)}s)")
    return pd.DataFrame(rows)


def save_final_links(conn, db_name, rows):
    if rows.empty:
        return 0
    conn.register("final_links", rows)
    try:
        conn.execute(
            f"""
            INSERT INTO {db_name}.timeline_reconciliation.TimelineTaskToMdrLinks (
                TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                TaskClass, TaskClassConfidence, TaskClassReason,
                MdrDocumentTitle, MdrTitleKey, LinkRank, LinkScore, LinkMethod, LinkReason,
                ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource, CreatedBy
            )
            SELECT
                TimelineName, ProjectCode, TaskRowId, TaskCode, TaskName, WbsName,
                TaskClass, TaskClassConfidence, TaskClassReason,
                MdrDocumentTitle, MdrTitleKey, LinkRank, LinkScore, LinkMethod, LinkReason,
                ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource, CreatedBy
            FROM final_links
            """
        )
    finally:
        conn.unregister("final_links")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="4 LLM judge final timeline task -> MDR links")
    parser.add_argument("--timeline", default="", help="Processa una sola TimelineName.")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    embedding_model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    conn = connect_motherduck(cfg)
    try:
        topk = load_topk_for_judge(conn, db_name, embedding_model, timeline_name=args.timeline or None, top_k=args.top_k)
        print(f"Top-K rows for judge: {len(topk)}")
        final_links = build_final_links(topk, cfg, args.progress_every)
        print(f"Final links created: {len(final_links)}")
        print(f"Final links saved: {save_final_links(conn, db_name, final_links)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
