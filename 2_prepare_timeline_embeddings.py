import argparse
import time

import pandas as pd

from timeline_reconciliation_common import (
    CONFIG_FILE,
    build_mdr_candidate_text,
    connect_motherduck,
    embed_texts,
    extract_project_code,
    float32_to_blob,
    parse_config_txt,
    text_hash,
)


CREATED_BY = "2_prepare_timeline_embeddings.py"


def load_eng_doc_tasks(conn, db_name, timeline_name=None):
    where = "WHERE TaskClass = 'ENG_DOC'"
    params = []
    if timeline_name:
        where += " AND TimelineName = ?"
        params.append(timeline_name)
    return conn.execute(
        f"""
        SELECT DISTINCT
            TimelineName,
            ProjectCode,
            TaskRowId,
            TaskText
        FROM {db_name}.timeline_reconciliation.TimelineTasksClassified
        {where}
        ORDER BY TimelineName, TaskRowId
        """,
        params,
    ).fetchdf()


def load_mdr_candidates(conn, db_name, timeline_name=None):
    timeline_filter = ""
    params = []
    if timeline_name:
        timeline_filter = "AND Mdr_code_name_ref = ?"
        params.append(timeline_name)
    return conn.execute(
        f"""
        WITH mdr_docs AS (
            SELECT DISTINCT
                Document_title,
                Mdr_code_name_ref
            FROM {db_name}.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All
            WHERE Document_title IS NOT NULL
              AND TRIM(Document_title) <> ''
              {timeline_filter}
        )
        SELECT DISTINCT
            m.Mdr_code_name_ref AS TimelineName,
            c.Document_title AS MdrDocumentTitle,
            lower(trim(c.Document_title)) AS MdrTitleKey,
            c.ConsolidatedDecisionType,
            c.ConsolidatedTitleKey,
            c.ConsolidatedRaciTitle,
            c.ConsolidatedConfidence,
            c.ConsolidatedReason,
            c.ConsolidatedSource,
            d.EffectiveDescription,
            e.DisciplineName,
            e.TypeName,
            e.CategoryDescription,
            e.ChapterName
        FROM {db_name}.mdr_reconciliation.v_MdrReconciliationResults_Consolidated c
        JOIN mdr_docs m
          ON m.Document_title = c.Document_title
        LEFT JOIN {db_name}.mdr_reconciliation.v_DocumentTitleDescriptions_Final d
          ON d.TitleKey = c.ConsolidatedTitleKey
        LEFT JOIN {db_name}.mdr_reconciliation.v_DocumentsEnriched e
          ON e.TitleKey = c.ConsolidatedTitleKey
        WHERE c.ConsolidatedDecisionType = 'MATCH'
        ORDER BY TimelineName, MdrDocumentTitle
        """,
        params,
    ).fetchdf()


def get_existing_task_hashes(conn, db_name, embedding_model, timeline_name=None):
    where = "WHERE EmbeddingModel = ?"
    params = [embedding_model]
    if timeline_name:
        where += " AND TimelineName = ?"
        params.append(timeline_name)
    rows = conn.execute(
        f"""
        SELECT TimelineName, TaskRowId, TextHash
        FROM {db_name}.timeline_reconciliation.TimelineTaskEmbeddings
        {where}
        """,
        params,
    ).fetchall()
    return {(r[0], int(r[1])): r[2] for r in rows}


def get_existing_candidate_hashes(conn, db_name, embedding_model, timeline_name=None):
    where = "WHERE EmbeddingModel = ?"
    params = [embedding_model]
    if timeline_name:
        where += " AND TimelineName = ?"
        params.append(timeline_name)
    rows = conn.execute(
        f"""
        SELECT TimelineName, MdrDocumentTitle, ConsolidatedTitleKey, TextHash
        FROM {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings
        {where}
        """,
        params,
    ).fetchall()
    return {(r[0], r[1], r[2]): r[3] for r in rows}


def filter_rows_to_refresh(rows, key_cols, text_col, existing_hashes, force_refresh):
    out = rows.copy()
    out["TextHash"] = out[text_col].apply(text_hash)
    if force_refresh:
        return out
    keep_mask = []
    for _, row in out.iterrows():
        key = tuple(row[c] for c in key_cols)
        keep_mask.append(existing_hashes.get(key) != row["TextHash"])
    return out[pd.Series(keep_mask, index=out.index)].copy()


def embed_rows(rows, text_col, cfg, embedding_model, progress_label, progress_every, batch_size):
    out = rows.copy()
    started = time.time()
    total = len(out)
    texts = out[text_col].astype(str).tolist()
    vectors = embed_texts(cfg, texts, batch_size=batch_size)
    embeddings = []
    dims = []
    for idx, vector in enumerate(vectors, 1):
        embeddings.append(float32_to_blob(vector))
        dims.append(len(vector))
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[{progress_label}] embedded {idx}/{total} rows (elapsed {round(time.time() - started, 1)}s)")
    out["EmbeddingModel"] = embedding_model
    out["Embedding"] = embeddings
    out["Dim"] = dims
    out["CreatedBy"] = CREATED_BY
    return out


def save_task_embeddings(conn, db_name, rows):
    if rows.empty:
        return 0
    conn.register("task_embeddings", rows)
    try:
        conn.execute("BEGIN;")
        conn.execute(
            f"""
            DELETE FROM {db_name}.timeline_reconciliation.TimelineTaskEmbeddings t
            USING task_embeddings r
            WHERE t.TimelineName = r.TimelineName
              AND t.TaskRowId = r.TaskRowId
              AND t.EmbeddingModel = r.EmbeddingModel
            """
        )
        conn.execute(
            f"""
            INSERT INTO {db_name}.timeline_reconciliation.TimelineTaskEmbeddings (
                TimelineName, ProjectCode, TaskRowId, TaskText, EmbeddingModel,
                TextHash, Embedding, Dim, CreatedBy
            )
            SELECT
                TimelineName, ProjectCode, TaskRowId, TaskText, EmbeddingModel,
                TextHash, Embedding, Dim, CreatedBy
            FROM task_embeddings
            """
        )
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.unregister("task_embeddings")
    return len(rows)


def save_candidate_embeddings(conn, db_name, rows):
    if rows.empty:
        return 0
    conn.register("candidate_embeddings", rows)
    try:
        conn.execute("BEGIN;")
        conn.execute(
            f"""
            DELETE FROM {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings t
            USING candidate_embeddings r
            WHERE t.TimelineName = r.TimelineName
              AND t.MdrDocumentTitle = r.MdrDocumentTitle
              AND t.ConsolidatedTitleKey = r.ConsolidatedTitleKey
              AND t.EmbeddingModel = r.EmbeddingModel
            """
        )
        conn.execute(
            f"""
            INSERT INTO {db_name}.timeline_reconciliation.TimelineMdrCandidateEmbeddings (
                TimelineName, ProjectCode, MdrDocumentTitle, MdrTitleKey,
                ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource,
                EffectiveDescription, DisciplineName, TypeName, CategoryDescription,
                ChapterName, CandidateText, EmbeddingModel, TextHash, Embedding, Dim, CreatedBy
            )
            SELECT
                TimelineName, ProjectCode, MdrDocumentTitle, MdrTitleKey,
                ConsolidatedDecisionType, ConsolidatedTitleKey, ConsolidatedRaciTitle,
                ConsolidatedConfidence, ConsolidatedReason, ConsolidatedSource,
                EffectiveDescription, DisciplineName, TypeName, CategoryDescription,
                ChapterName, CandidateText, EmbeddingModel, TextHash, Embedding, Dim, CreatedBy
            FROM candidate_embeddings
            """
        )
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.unregister("candidate_embeddings")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="2 prepare embeddings for timeline task -> MDR retrieval")
    parser.add_argument("--timeline", default="", help="Processa una sola TimelineName.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--embed-batch-size", type=int, default=256, help="Batch size chiamate embeddings API.")
    parser.add_argument("--force-refresh", action="store_true", help="Ricalcola embeddings anche se TextHash invariato.")
    args = parser.parse_args()

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    embedding_model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    conn = connect_motherduck(cfg)
    try:
        tasks = load_eng_doc_tasks(conn, db_name, timeline_name=args.timeline or None)
        print(f"Task ENG_DOC da embeddare: {len(tasks)}")
        if not tasks.empty:
            existing_task_hashes = get_existing_task_hashes(conn, db_name, embedding_model, timeline_name=args.timeline or None)
            task_refresh = filter_rows_to_refresh(
                tasks,
                key_cols=["TimelineName", "TaskRowId"],
                text_col="TaskText",
                existing_hashes=existing_task_hashes,
                force_refresh=args.force_refresh,
            )
            print(f"Task embeddings da aggiornare: {len(task_refresh)}")
            if not task_refresh.empty:
                task_embeddings = embed_rows(
                    task_refresh,
                    "TaskText",
                    cfg,
                    embedding_model,
                    "tasks",
                    args.progress_every,
                    args.embed_batch_size,
                )
                print(f"Salvate task embeddings: {save_task_embeddings(conn, db_name, task_embeddings)}")

        candidates = load_mdr_candidates(conn, db_name, timeline_name=args.timeline or None)
        print(f"Candidati MDR MATCH da embeddare: {len(candidates)}")
        if not candidates.empty:
            candidates["ProjectCode"] = candidates["TimelineName"].apply(extract_project_code)
            candidates["CandidateText"] = candidates.apply(build_mdr_candidate_text, axis=1)
            existing_candidate_hashes = get_existing_candidate_hashes(
                conn, db_name, embedding_model, timeline_name=args.timeline or None
            )
            candidate_refresh = filter_rows_to_refresh(
                candidates,
                key_cols=["TimelineName", "MdrDocumentTitle", "ConsolidatedTitleKey"],
                text_col="CandidateText",
                existing_hashes=existing_candidate_hashes,
                force_refresh=args.force_refresh,
            )
            print(f"Candidate embeddings da aggiornare: {len(candidate_refresh)}")
            if not candidate_refresh.empty:
                candidate_embeddings = embed_rows(
                    candidate_refresh,
                    "CandidateText",
                    cfg,
                    embedding_model,
                    "mdr_candidates",
                    args.progress_every,
                    args.embed_batch_size,
                )
                print(f"Salvate candidate embeddings: {save_candidate_embeddings(conn, db_name, candidate_embeddings)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
