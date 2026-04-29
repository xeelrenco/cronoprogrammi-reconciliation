import argparse
import time

import pandas as pd

from timeline_reconciliation_common import (
    CONFIG_FILE,
    build_mdr_candidate_text,
    connect_motherduck,
    embed_text,
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
        ),
        agent_context AS (
            SELECT
                Document_title,
                TitleKey,
                any_value(EffectiveDescription) AS EffectiveDescription,
                any_value(DisciplineName) AS DisciplineName,
                any_value(TypeName) AS TypeName,
                any_value(CategoryDescription) AS CategoryDescription,
                any_value(ChapterName) AS ChapterName
            FROM {db_name}.mdr_reconciliation.v_MdrReconciliationAgentInput
            GROUP BY Document_title, TitleKey
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
            a.EffectiveDescription,
            a.DisciplineName,
            a.TypeName,
            a.CategoryDescription,
            a.ChapterName
        FROM {db_name}.mdr_reconciliation.v_MdrReconciliationResults_Consolidated c
        JOIN mdr_docs m
          ON m.Document_title = c.Document_title
        LEFT JOIN agent_context a
          ON a.Document_title = c.Document_title
         AND a.TitleKey = c.ConsolidatedTitleKey
        WHERE c.ConsolidatedDecisionType = 'MATCH'
        ORDER BY TimelineName, MdrDocumentTitle
        """,
        params,
    ).fetchdf()


def embed_rows(rows, text_col, cfg, embedding_model, progress_label, progress_every):
    out = rows.copy()
    embeddings = []
    dims = []
    started = time.time()
    total = len(out)
    for idx, (_, row) in enumerate(out.iterrows(), 1):
        vector = embed_text(cfg, row[text_col])
        embeddings.append(float32_to_blob(vector))
        dims.append(len(vector))
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[{progress_label}] embedded {idx}/{total} rows (elapsed {round(time.time() - started, 1)}s)")
    out["EmbeddingModel"] = embedding_model
    out["TextHash"] = out[text_col].apply(text_hash)
    out["Embedding"] = embeddings
    out["Dim"] = dims
    out["CreatedBy"] = CREATED_BY
    return out


def save_task_embeddings(conn, db_name, rows):
    if rows.empty:
        return 0
    conn.register("task_embeddings", rows)
    try:
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
    finally:
        conn.unregister("task_embeddings")
    return len(rows)


def save_candidate_embeddings(conn, db_name, rows):
    if rows.empty:
        return 0
    conn.register("candidate_embeddings", rows)
    try:
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
    finally:
        conn.unregister("candidate_embeddings")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="2 prepare embeddings for timeline task -> MDR retrieval")
    parser.add_argument("--timeline", default="", help="Processa una sola TimelineName.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    cfg = parse_config_txt(CONFIG_FILE)
    db_name = cfg.get("MOTHERDUCK_DB", "my_db").strip() or "my_db"
    embedding_model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")
    conn = connect_motherduck(cfg)
    try:
        tasks = load_eng_doc_tasks(conn, db_name, timeline_name=args.timeline or None)
        print(f"Task ENG_DOC da embeddare: {len(tasks)}")
        if not tasks.empty:
            task_embeddings = embed_rows(tasks, "TaskText", cfg, embedding_model, "tasks", args.progress_every)
            print(f"Salvate task embeddings: {save_task_embeddings(conn, db_name, task_embeddings)}")

        candidates = load_mdr_candidates(conn, db_name, timeline_name=args.timeline or None)
        print(f"Candidati MDR MATCH da embeddare: {len(candidates)}")
        if not candidates.empty:
            candidates["ProjectCode"] = candidates["TimelineName"].apply(extract_project_code)
            candidates["CandidateText"] = candidates.apply(build_mdr_candidate_text, axis=1)
            candidate_embeddings = embed_rows(candidates, "CandidateText", cfg, embedding_model, "mdr_candidates", args.progress_every)
            print(f"Salvate candidate embeddings: {save_candidate_embeddings(conn, db_name, candidate_embeddings)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
