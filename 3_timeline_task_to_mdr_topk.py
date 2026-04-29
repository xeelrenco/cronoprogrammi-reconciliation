import argparse
import time

import numpy as np
import pandas as pd

from timeline_reconciliation_common import CONFIG_FILE, blob_to_float32, connect_motherduck, parse_config_txt


CREATED_BY = "3_timeline_task_to_mdr_topk.py"
RETRIEVAL_METHOD = "embedding_cosine_topk"


def load_task_embeddings(conn, db_name, embedding_model, timeline_name=None):
    where = "WHERE e.EmbeddingModel = ?"
    params = [embedding_model]
    if timeline_name:
        where += " AND e.TimelineName = ?"
        params.append(timeline_name)
    return conn.execute(
        f"""
        SELECT
            e.TimelineName,
            e.ProjectCode,
            e.TaskRowId,
            c.TaskName,
            c.WbsName,
            e.TaskText,
            e.TextHash AS TaskTextHash,
            e.EmbeddingModel,
            e.Embedding
        FROM {db_name}.timeline_reconciliation.TimelineTaskEmbeddings e
        JOIN {db_name}.timeline_reconciliation.TimelineTasksClassified c
          ON c.TimelineName = e.TimelineName
         AND c.TaskRowId = e.TaskRowId
        {where}
        ORDER BY e.TimelineName, e.TaskRowId
        """,
        params,
    ).fetchdf()


def load_candidate_embeddings(conn, db_name, embedding_model, timeline_name=None):
    where = "WHERE EmbeddingModel = ?"
    params = [embedding_model]
    if timeline_name:
        where += " AND TimelineName = ?"
        params.append(timeline_name)
    return conn.execute(
        f"""
        SELECT
            TimelineName,
            ProjectCode,
            MdrDocumentTitle,
            MdrTitleKey,
            ConsolidatedTitleKey,
            ConsolidatedRaciTitle,
            TextHash AS CandidateTextHash,
            EmbeddingModel,
            Embedding
        FROM {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings
        {where}
        ORDER BY TimelineName, MdrDocumentTitle
        """,
        params,
    ).fetchdf()


def compute_topk(tasks, candidates, top_k):
    rows = []
    started = time.time()
    for timeline, task_group in tasks.groupby("TimelineName"):
        candidate_group = candidates[candidates["TimelineName"] == timeline].reset_index(drop=True)
        if candidate_group.empty:
            print(f"[{timeline}] nessun candidato MDR embedding")
            continue
        candidate_matrix = np.vstack(candidate_group["Embedding"].apply(blob_to_float32).tolist())
        for _, task in task_group.iterrows():
            task_vector = blob_to_float32(task["Embedding"])
            similarities = candidate_matrix @ task_vector
            top_indices = np.argsort(-similarities)[:top_k]
            for rank, cand_idx in enumerate(top_indices, 1):
                cand = candidate_group.iloc[int(cand_idx)]
                rows.append(
                    {
                        "TimelineName": task["TimelineName"],
                        "ProjectCode": task["ProjectCode"],
                        "TaskRowId": int(task["TaskRowId"]),
                        "TaskName": task["TaskName"],
                        "WbsName": task["WbsName"],
                        "MdrDocumentTitle": cand["MdrDocumentTitle"],
                        "MdrTitleKey": cand["MdrTitleKey"],
                        "ConsolidatedTitleKey": cand["ConsolidatedTitleKey"],
                        "ConsolidatedRaciTitle": cand["ConsolidatedRaciTitle"],
                        "Similarity": float(similarities[int(cand_idx)]),
                        "Rank": rank,
                        "EmbeddingModel": task["EmbeddingModel"],
                        "RetrievalMethod": RETRIEVAL_METHOD,
                        "TaskTextHash": task["TaskTextHash"],
                        "CandidateTextHash": cand["CandidateTextHash"],
                        "CreatedBy": CREATED_BY,
                    }
                )
        print(
            f"[{timeline}] Top-{top_k} calcolati per {len(task_group)} task "
            f"(elapsed {round(time.time() - started, 1)}s)"
        )
    return pd.DataFrame(rows)


def save_candidates(conn, db_name, rows):
    if rows.empty:
        return 0
    conn.register("topk_rows", rows)
    try:
        conn.execute(
            f"""
            INSERT INTO {db_name}.timeline_reconciliation.TimelineTaskToMdrCandidates (
                TimelineName, ProjectCode, TaskRowId, TaskName, WbsName,
                MdrDocumentTitle, MdrTitleKey, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                Similarity, Rank, EmbeddingModel, RetrievalMethod, TaskTextHash,
                CandidateTextHash, CreatedBy
            )
            SELECT
                TimelineName, ProjectCode, TaskRowId, TaskName, WbsName,
                MdrDocumentTitle, MdrTitleKey, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                Similarity, Rank, EmbeddingModel, RetrievalMethod, TaskTextHash,
                CandidateTextHash, CreatedBy
            FROM topk_rows
            """
        )
    finally:
        conn.unregister("topk_rows")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="3 compute semantic Top-K timeline task -> MDR candidates")
    parser.add_argument("--timeline", default="", help="Processa una sola TimelineName.")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    embedding_model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    conn = connect_motherduck(cfg)
    try:
        tasks = load_task_embeddings(conn, db_name, embedding_model, timeline_name=args.timeline or None)
        candidates = load_candidate_embeddings(conn, db_name, embedding_model, timeline_name=args.timeline or None)
        print(f"Task embeddings: {len(tasks)}")
        print(f"MDR candidate embeddings: {len(candidates)}")
        topk = compute_topk(tasks, candidates, args.top_k)
        print(f"Top-K rows calcolate: {len(topk)}")
        print(f"Top-K rows salvate: {save_candidates(conn, db_name, topk)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
