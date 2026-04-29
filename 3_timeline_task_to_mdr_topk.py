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
        WITH e_latest AS (
            SELECT *
            FROM (
                SELECT
                    e.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.TimelineName, e.TaskRowId, e.EmbeddingModel
                        ORDER BY e.UpdatedAt DESC, e.CreatedAt DESC
                    ) AS rn
                FROM {db_name}.timeline_reconciliation.TimelineTaskEmbeddings e
                {where}
            ) x
            WHERE x.rn = 1
        ),
        c_latest AS (
            SELECT *
            FROM (
                SELECT
                    c.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.TimelineName, c.TaskRowId
                        ORDER BY c.UpdatedAt DESC, c.CreatedAt DESC
                    ) AS rn
                FROM {db_name}.timeline_reconciliation.TimelineTasksClassified c
            ) y
            WHERE y.rn = 1
        )
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
        FROM e_latest e
        JOIN c_latest c
          ON c.TimelineName = e.TimelineName
         AND c.TaskRowId = e.TaskRowId
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
        WITH latest AS (
            SELECT *
            FROM (
                SELECT
                    c.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.TimelineName, c.MdrDocumentTitle, c.ConsolidatedTitleKey, c.EmbeddingModel
                        ORDER BY c.UpdatedAt DESC, c.CreatedAt DESC
                    ) AS rn
                FROM {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings c
                {where}
            ) x
            WHERE x.rn = 1
        )
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
        FROM latest
        ORDER BY TimelineName, MdrDocumentTitle
        """,
        params,
    ).fetchdf()


def compute_topk(tasks, candidates, top_k):
    if tasks.empty or candidates.empty:
        return pd.DataFrame()
    rows = []
    started = time.time()
    for timeline, task_group in tasks.groupby("TimelineName"):
        candidate_group = candidates[candidates["TimelineName"] == timeline].reset_index(drop=True)
        if candidate_group.empty:
            print(f"[{timeline}] nessun candidato MDR embedding")
            continue
        task_matrix = np.vstack(task_group["Embedding"].apply(blob_to_float32).tolist()).astype(np.float32, copy=False)
        candidate_matrix = np.vstack(candidate_group["Embedding"].apply(blob_to_float32).tolist()).astype(
            np.float32, copy=False
        )
        if task_matrix.shape[1] != candidate_matrix.shape[1]:
            raise RuntimeError(
                f"[{timeline}] embedding dim mismatch: tasks {task_matrix.shape[1]} vs candidates {candidate_matrix.shape[1]}"
            )
        scores = (task_matrix @ candidate_matrix.T).astype(np.float32, copy=False)
        effective_k = min(top_k, candidate_matrix.shape[0])
        for task_pos, (_, task) in enumerate(task_group.iterrows()):
            similarities = scores[task_pos]
            if effective_k >= similarities.shape[0]:
                top_indices = np.argsort(-similarities)
            else:
                top_indices = np.argpartition(-similarities, effective_k)[:effective_k]
                top_indices = top_indices[np.argsort(-similarities[top_indices])]
            for rank, cand_idx in enumerate(top_indices[:effective_k], 1):
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


def clear_candidates_scope(conn, db_name, embedding_model, timeline_name=None):
    if timeline_name:
        conn.execute(
            f"""
            DELETE FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrCandidates
            WHERE EmbeddingModel = ? AND TimelineName = ?
            """,
            [embedding_model, timeline_name],
        )
    else:
        conn.execute(
            f"""
            DELETE FROM {db_name}.timeline_reconciliation.TimelineTaskToMdrCandidates
            WHERE EmbeddingModel = ?
            """,
            [embedding_model],
        )


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
        clear_candidates_scope(conn, db_name, embedding_model, timeline_name=args.timeline or None)
        print("Pulizia scope Top-K completata.")
        topk = compute_topk(tasks, candidates, args.top_k)
        print(f"Top-K rows calcolate: {len(topk)}")
        print(f"Top-K rows salvate: {save_candidates(conn, db_name, topk)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
