-- =============================================================================
-- Timeline reconciliation — greenfield DDL (MotherDuck / DuckDB)
-- =============================================================================
-- Database: my_db (replace if your config uses another name)
-- Schema: timeline_reconciliation
--
-- Python pipeline (write / read):
--   1_classify_timeline_tasks.py           → TimelineTasksClassified
--   2_prepare_timeline_embeddings.py     → TimelineTaskEmbeddings, TimelineMdrCandidateEmbeddings
--   3_timeline_task_to_mdr_topk.py         → TimelineTaskToMdrCandidates
--   4_resolve_timeline_task_mdr_links.py   → TimelineTaskToMdrLinks, TimelineTaskToMdrResolverLlmTopCandidates
--   5_generate_timeline_reconciliation_report.py → reads tables (Excel report; no date views)
--
-- Full rebuild on existing DB:
--   DROP SCHEMA IF EXISTS my_db.timeline_reconciliation CASCADE;
-- then execute this script.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS my_db.timeline_reconciliation;

COMMENT ON SCHEMA my_db.timeline_reconciliation IS
'Primavera schedule-to-MDR/RACI reconciliation. Pipeline: classify ENG_DOC tasks → embeddings → Top-K retrieval → LLM resolver links. Schedule dates v3: raw Early/Actual/Target + actualized Selected* on links. External MDR source: historical_mdr_normalization + mdr_reconciliation consolidated views.';

-- =============================================================================
-- 1. TimelineTasksClassified
-- =============================================================================

CREATE TABLE IF NOT EXISTS my_db.timeline_reconciliation.TimelineTasksClassified (
    TimelineName VARCHAR,
    ProjectCode VARCHAR,

    TaskRowId BIGINT,
    TaskCode VARCHAR,
    TaskName VARCHAR,
    WbsName VARCHAR,

    EarlyStartDate TIMESTAMP,
    EarlyEndDate TIMESTAMP,
    ActualStartDate TIMESTAMP,
    ActualEndDate TIMESTAMP,
    TargetStartDate TIMESTAMP,
    TargetEndDate TIMESTAMP,

    TaskText VARCHAR,

    TaskClass VARCHAR,
    TaskClassConfidence VARCHAR,
    TaskClassReason VARCHAR,
    ClassifierModel VARCHAR,
    ClassifierPromptVersion VARCHAR,

    CreatedAt TIMESTAMP DEFAULT NOW(),
    UpdatedAt TIMESTAMP DEFAULT NOW(),
    CreatedBy VARCHAR
);

COMMENT ON TABLE my_db.timeline_reconciliation.TimelineTasksClassified IS
'Grain: one row per Primavera TASK row. Role: canonical classified task staging. Writer: 1_classify_timeline_tasks.py. Readers: 2_prepare_timeline_embeddings.py, 3_timeline_task_to_mdr_topk.py, 4_resolve_timeline_task_mdr_links.py, 5_generate_timeline_reconciliation_report.py. Natural key: (TimelineName, TaskRowId). Filter ENG_DOC for MDR matching. Schedule: six raw timestamps feed actualized COALESCE Actual→Early→Target.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TimelineName IS
'[KEY] Primavera timeline / project schedule id; usually XER/Excel file stem. Join/filter: historical MDR Mdr_code_name_ref (textual). Partition all downstream tables by this column.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ProjectCode IS
'[DERIVED] Project code parsed from TimelineName (e.g. 7910). Reporting only; not a guaranteed MDR join key.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskRowId IS
'[KEY] Source row index in Primavera TASK sheet. With TimelineName forms task identity across pipeline.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskCode IS
'[ATTR] Primavera activity id/code when present in export.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskName IS
'[ATTR] Original Primavera activity name from schedule export.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.WbsName IS
'[ATTR] WBS name resolved from PROJWBS; context for classification and matching.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.EarlyStartDate IS
'[DATE_RAW] Primavera early start. Actualized start COALESCE priority 2 (after Actual, before Target).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.EarlyEndDate IS
'[DATE_RAW] Primavera early finish/end. Actualized finish COALESCE priority 2.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ActualStartDate IS
'[DATE_RAW] Primavera actual start when activity has started. Actualized start COALESCE priority 1.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ActualEndDate IS
'[DATE_RAW] Primavera actual finish/end when completed. Actualized finish COALESCE priority 1.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TargetStartDate IS
'[DATE_RAW] Primavera target start (common XER fallback). Actualized start COALESCE priority 3.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TargetEndDate IS
'[DATE_RAW] Primavera target end/finish. Actualized finish COALESCE priority 3.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskText IS
'[TEXT] Embedding/classification input: typically task name + WBS + class label.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskClass IS
'[CLASS] ENG_DOC = linkable engineering document task; OTHER = excluded from MDR retrieval/resolver.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskClassConfidence IS
'[CLASS] Classifier confidence: HIGH | MEDIUM | LOW.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskClassReason IS
'[CLASS] Short natural-language justification for TaskClass.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ClassifierModel IS
'[AUDIT] Model or rule id (e.g. gpt-4o-mini, doc_prefix_rule).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ClassifierPromptVersion IS
'[AUDIT] Classifier prompt/version identifier.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.CreatedAt IS
'[AUDIT] Row insert timestamp.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.UpdatedAt IS
'[AUDIT] Last refresh timestamp; use ROW_NUMBER ORDER BY UpdatedAt DESC for latest row per task.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.CreatedBy IS
'[AUDIT] Producing script name (expected: 1_classify_timeline_tasks.py).';

-- =============================================================================
-- 2. TimelineTaskEmbeddings
-- =============================================================================

CREATE TABLE IF NOT EXISTS my_db.timeline_reconciliation.TimelineTaskEmbeddings (
    TimelineName VARCHAR,
    ProjectCode VARCHAR,

    TaskRowId BIGINT,
    TaskText VARCHAR,
    EmbeddingModel VARCHAR,
    TextHash VARCHAR,
    Embedding BLOB,
    Dim INTEGER,

    CreatedAt TIMESTAMP DEFAULT NOW(),
    UpdatedAt TIMESTAMP DEFAULT NOW(),
    CreatedBy VARCHAR
);

COMMENT ON TABLE my_db.timeline_reconciliation.TimelineTaskEmbeddings IS
'Grain: one row per (TimelineName, TaskRowId, EmbeddingModel) ENG_DOC task. Role: L2-normalized task vectors for cosine retrieval. Writer: 2_prepare_timeline_embeddings.py. Reader: 3_timeline_task_to_mdr_topk.py. Join parent: TimelineTasksClassified ON (TimelineName, TaskRowId) WHERE TaskClass=ENG_DOC.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TimelineName IS
'[KEY] Same as TimelineTasksClassified.TimelineName.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.ProjectCode IS
'[DERIVED] Project code from timeline name.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TaskRowId IS
'[KEY] Task row id; join TimelineTasksClassified.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TaskText IS
'[TEXT] Exact string embedded; must match hash logic in TextHash.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.EmbeddingModel IS
'[KEY] Embedding model id (e.g. text-embedding-3-small); filter retrieval runs by model.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TextHash IS
'[AUDIT] SHA256 of normalized TaskText; skip re-embed when unchanged.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.Embedding IS
'[VECTOR] float32 L2-normalized embedding bytes (BLOB).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.Dim IS
'[VECTOR] Embedding dimension count.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.CreatedAt IS
'[AUDIT] Insert time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.UpdatedAt IS
'[AUDIT] Last embedding refresh.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.CreatedBy IS
'[AUDIT] Expected: 2_prepare_timeline_embeddings.py.';

-- =============================================================================
-- 3. TimelineMdrCandidateEmbeddings
-- =============================================================================

CREATE TABLE IF NOT EXISTS my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings (
    TimelineName VARCHAR,
    ProjectCode VARCHAR,

    MdrDocumentTitle VARCHAR,
    MdrTitleKey VARCHAR,

    ConsolidatedDecisionType VARCHAR,
    ConsolidatedTitleKey VARCHAR,
    ConsolidatedRaciTitle VARCHAR,
    ConsolidatedConfidence DOUBLE,
    ConsolidatedReason VARCHAR,
    ConsolidatedSource VARCHAR,

    EffectiveDescription VARCHAR,
    DisciplineName VARCHAR,
    TypeName VARCHAR,
    CategoryDescription VARCHAR,
    ChapterName VARCHAR,
    CandidateText VARCHAR,

    EmbeddingModel VARCHAR,
    TextHash VARCHAR,
    Embedding BLOB,
    Dim INTEGER,

    CreatedAt TIMESTAMP DEFAULT NOW(),
    UpdatedAt TIMESTAMP DEFAULT NOW(),
    CreatedBy VARCHAR
);

COMMENT ON TABLE my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings IS
'Grain: one row per MDR document candidate per timeline and embedding model. Role: MDR-side vectors for Top-K. Writer: 2_prepare_timeline_embeddings.py from mdr_reconciliation.v_MdrReconciliationResults_Consolidated (MATCH only). Reader: 3_timeline_task_to_mdr_topk.py. Join keys: TimelineName + MdrTitleKey / ConsolidatedTitleKey.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.TimelineName IS
'[KEY] Must match MDR Mdr_code_name_ref for same project schedule.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ProjectCode IS
'[DERIVED] Parsed project code.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.MdrDocumentTitle IS
'[ATTR] Historical MDR Document_title from normalized MDR.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.MdrTitleKey IS
'[KEY] Normalized MDR title key; join raci_matrix.Documents.TitleKey.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedDecisionType IS
'[MDR] Final MDR→RACI decision; table populated with MATCH rows only.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedTitleKey IS
'[KEY] Final RACI TitleKey from consolidated reconciliation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedRaciTitle IS
'[ATTR] Final RACI document title.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedConfidence IS
'[MDR] Consolidated match confidence score.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedReason IS
'[MDR] Consolidated decision rationale text.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedSource IS
'[MDR] Provenance layer: judge_3_3 | recovery_3_4 etc.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.EffectiveDescription IS
'[TEXT] RACI description used in CandidateText (manual override preferred).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.DisciplineName IS
'[RACI_META] Human-readable discipline.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.TypeName IS
'[RACI_META] Document type label.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CategoryDescription IS
'[RACI_META] Category description.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ChapterName IS
'[RACI_META] Chapter name.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CandidateText IS
'[TEXT] Full text embedded (MDR title + RACI context).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.EmbeddingModel IS
'[KEY] Embedding model; must match task embeddings for similarity.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.TextHash IS
'[AUDIT] Hash of CandidateText for incremental refresh.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.Embedding IS
'[VECTOR] float32 L2-normalized BLOB.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.Dim IS
'[VECTOR] Dimension.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CreatedAt IS
'[AUDIT] Insert time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.UpdatedAt IS
'[AUDIT] Last refresh.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CreatedBy IS
'[AUDIT] Expected: 2_prepare_timeline_embeddings.py.';

-- =============================================================================
-- 4. TimelineTaskToMdrCandidates
-- =============================================================================

CREATE TABLE IF NOT EXISTS my_db.timeline_reconciliation.TimelineTaskToMdrCandidates (
    TimelineName VARCHAR,
    ProjectCode VARCHAR,

    TaskRowId BIGINT,
    TaskName VARCHAR,
    WbsName VARCHAR,

    MdrDocumentTitle VARCHAR,
    MdrTitleKey VARCHAR,
    ConsolidatedTitleKey VARCHAR,
    ConsolidatedRaciTitle VARCHAR,

    Similarity DOUBLE,
    Rank INTEGER,
    EmbeddingModel VARCHAR,
    RetrievalMethod VARCHAR,
    TaskTextHash VARCHAR,
    CandidateTextHash VARCHAR,

    CreatedAt TIMESTAMP DEFAULT NOW(),
    CreatedBy VARCHAR
);

COMMENT ON TABLE my_db.timeline_reconciliation.TimelineTaskToMdrCandidates IS
'Grain: one row per (TimelineName, TaskRowId, Rank) within Top-K retrieval. Role: retrieval evidence only — not final business truth (see TimelineTaskToMdrLinks). Writer: 3_timeline_task_to_mdr_topk.py. Reader: 4_resolve_timeline_task_mdr_links.py. No schedule date columns by design.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TimelineName IS
'[KEY] Timeline scope for retrieval.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.ProjectCode IS
'[DERIVED] Project code.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TaskRowId IS
'[KEY] ENG_DOC task id; join TimelineTasksClassified / TimelineTaskEmbeddings.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TaskName IS
'[ATTR] Task name snapshot for LLM/audit.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.WbsName IS
'[ATTR] WBS snapshot for audit.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.MdrDocumentTitle IS
'[ATTR] Retrieved MDR document title.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.MdrTitleKey IS
'[KEY] Normalized MDR title key.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.ConsolidatedTitleKey IS
'[KEY] RACI TitleKey for candidate.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.ConsolidatedRaciTitle IS
'[ATTR] RACI title for candidate.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.Similarity IS
'[SCORE] Cosine similarity in [0,1]; higher = closer embedding match.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.Rank IS
'[RANK] Retrieval order per task; Rank=1 is best similarity. Maps to resolver candidate_id.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.EmbeddingModel IS
'[KEY] Embedding model used for this retrieval run.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.RetrievalMethod IS
'[AUDIT] Method tag (e.g. embedding_cosine_topk).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TaskTextHash IS
'[AUDIT] Links to TimelineTaskEmbeddings.TextHash.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.CandidateTextHash IS
'[AUDIT] Links to TimelineMdrCandidateEmbeddings.TextHash.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.CreatedAt IS
'[AUDIT] Insert time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.CreatedBy IS
'[AUDIT] Expected: 3_timeline_task_to_mdr_topk.py.';

-- =============================================================================
-- 5. TimelineTaskToMdrLinks
-- =============================================================================

CREATE TABLE IF NOT EXISTS my_db.timeline_reconciliation.TimelineTaskToMdrLinks (
    TimelineName VARCHAR,
    ProjectCode VARCHAR,

    TaskRowId BIGINT,
    TaskCode VARCHAR,
    TaskName VARCHAR,
    WbsName VARCHAR,

    EarlyStartDate TIMESTAMP,
    EarlyEndDate TIMESTAMP,
    ActualStartDate TIMESTAMP,
    ActualEndDate TIMESTAMP,
    TargetStartDate TIMESTAMP,
    TargetEndDate TIMESTAMP,
    SelectedStartDate TIMESTAMP,
    SelectedFinishDate TIMESTAMP,

    TaskClass VARCHAR,
    TaskClassConfidence VARCHAR,
    TaskClassReason VARCHAR,

    MdrDocumentTitle VARCHAR,
    MdrTitleKey VARCHAR,

    LinkRank INTEGER,
    LinkScore DOUBLE,
    LinkMethod VARCHAR,
    LinkReason VARCHAR,

    ConsolidatedDecisionType VARCHAR,
    ConsolidatedTitleKey VARCHAR,
    ConsolidatedRaciTitle VARCHAR,
    ConsolidatedConfidence DOUBLE,
    ConsolidatedReason VARCHAR,
    ConsolidatedSource VARCHAR,

    CreatedAt TIMESTAMP DEFAULT NOW(),
    CreatedBy VARCHAR
);

COMMENT ON TABLE my_db.timeline_reconciliation.TimelineTaskToMdrLinks IS
'Grain: one row per accepted task-to-MDR link (multiple rows per task if multiple MDRs). Role: final business links after LLM resolver. Writer: 4_resolve_timeline_task_mdr_links.py. Readers: 5_generate_timeline_reconciliation_report.py, analytics. Operational schedule dates: SelectedStartDate/SelectedFinishDate (actualized). Raw schedule copies preserved for audit. Latest link per (TimelineName, TaskRowId, MdrTitleKey): ORDER BY CreatedAt DESC.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TimelineName IS
'[KEY] Timeline identifier.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ProjectCode IS
'[DERIVED] Project code.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskRowId IS
'[KEY] Primavera task row.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskCode IS
'[ATTR] Activity code at link time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskName IS
'[ATTR] Activity name at link time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.WbsName IS
'[ATTR] WBS at link time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.EarlyStartDate IS
'[DATE_RAW] Snapshot from classified task at link creation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.EarlyEndDate IS
'[DATE_RAW] Snapshot early end at link creation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ActualStartDate IS
'[DATE_RAW] Snapshot actual start at link creation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ActualEndDate IS
'[DATE_RAW] Snapshot actual end at link creation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TargetStartDate IS
'[DATE_RAW] Snapshot target start at link creation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TargetEndDate IS
'[DATE_RAW] Snapshot target end at link creation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.SelectedStartDate IS
'[DATE_OPS] Business start date: actualized COALESCE(ActualStart, EarlyStart, TargetStart) frozen at link time. Use for document sequencing and man-day duration (with SelectedFinishDate).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.SelectedFinishDate IS
'[DATE_OPS] Business finish date: actualized COALESCE(ActualEnd, EarlyEnd, TargetEnd) frozen at link time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskClass IS
'[CLASS] Usually ENG_DOC for rows in this table.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskClassConfidence IS
'[CLASS] Classifier confidence copied at link time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskClassReason IS
'[CLASS] Classifier reason copied at link time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.MdrDocumentTitle IS
'[ATTR] Linked historical MDR title.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.MdrTitleKey IS
'[KEY] Linked MDR title key; with TimelineName+TaskRowId identifies link set.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkRank IS
'[RANK] Preference among links for same task; 1 = primary link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkScore IS
'[SCORE] LLM resolver confidence for this link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkMethod IS
'[AUDIT] Resolver method (e.g. embedding_topk_llm_resolver).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkReason IS
'[TEXT] Short LLM explanation for accepting this MDR association.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedDecisionType IS
'[MDR] Expected MATCH from MDR→RACI consolidation.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedTitleKey IS
'[KEY] RACI TitleKey for linked document.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedRaciTitle IS
'[ATTR] RACI title for linked document.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedConfidence IS
'[MDR] Consolidated reconciliation confidence.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedReason IS
'[MDR] Consolidated reconciliation reason.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedSource IS
'[MDR] Consolidated layer source (judge_3_3, recovery_3_4, …).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.CreatedAt IS
'[AUDIT] Link row insert time; use for latest-link queries.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.CreatedBy IS
'[AUDIT] Expected: 4_resolve_timeline_task_mdr_links.py.';

-- =============================================================================
-- 6. TimelineTaskToMdrResolverLlmTopCandidates
-- =============================================================================

CREATE TABLE IF NOT EXISTS my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates (
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

COMMENT ON TABLE my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates IS
'Grain: up to N LLM-ranked plausible MDR candidates per task per embedding model (PK includes CandidateRankWithinResolver). Role: resolver shortlist audit — not final links. Writer: 4_resolve_timeline_task_mdr_links.py. Reader: 5_generate_timeline_reconciliation_report.py. Join TimelineTaskToMdrCandidates via RetrievalRank = Rank.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.TimelineName IS
'[PK] Timeline scope.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ProjectCode IS
'[DERIVED] Project code.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.TaskRowId IS
'[PK] Task row id.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.EmbeddingModel IS
'[PK] Resolver run embedding model.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CandidateRankWithinResolver IS
'[PK][RANK] LLM preference order in top_candidates JSON (1 = best).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.RetrievalRank IS
'[FK] References TimelineTaskToMdrCandidates.Rank (resolver candidate_id).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.MdrDocumentTitle IS
'[ATTR] MDR title for shortlist row.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.MdrTitleKey IS
'[ATTR] MDR key.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ConsolidatedTitleKey IS
'[ATTR] RACI TitleKey.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ConsolidatedRaciTitle IS
'[ATTR] RACI title.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.RetrievalSimilarity IS
'[SCORE] Embedding similarity at retrieval time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.LlmConfidence IS
'[SCORE] LLM confidence for plausibility of this candidate.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.WhyPlausible IS
'[TEXT] LLM rationale (why_plausible).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.EffectiveDescription IS
'[TEXT] RACI description shown to resolver.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.DisciplineName IS
'[RACI_META] Discipline label.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.TypeName IS
'[RACI_META] Type label.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CategoryDescription IS
'[RACI_META] Category label.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ChapterName IS
'[RACI_META] Chapter label.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CreatedAt IS
'[AUDIT] Shortlist persist time.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CreatedBy IS
'[AUDIT] Expected: 4_resolve_timeline_task_mdr_links.py.';

-- =============================================================================
-- 7. Views — schedule actualized (optional SQL analytics)
-- =============================================================================

CREATE OR REPLACE VIEW my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates AS
SELECT
    c.*,
    COALESCE(c.ActualStartDate, c.EarlyStartDate, c.TargetStartDate) AS StartActualized,
    COALESCE(c.ActualEndDate, c.EarlyEndDate, c.TargetEndDate) AS FinishActualized
FROM my_db.timeline_reconciliation.TimelineTasksClassified AS c;

COMMENT ON VIEW my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates IS
'Extends TimelineTasksClassified with computed actualized dates. Formula: StartActualized=COALESCE(ActualStart,EarlyStart,TargetStart); FinishActualized=COALESCE(ActualEnd,EarlyEnd,TargetEnd). Refreshed by timeline_reconciliation_common.refresh_timeline_classified_dates_view(). Report script 5 reads base table directly, not this view.';

COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates.StartActualized IS
'[COMPUTED] Actualized start from raw columns on same row; same rule as SelectedStartDate on links.';
COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates.FinishActualized IS
'[COMPUTED] Actualized finish from raw columns on same row.';

CREATE OR REPLACE VIEW my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates AS
SELECT
    l.*,
    COALESCE(l.SelectedStartDate, COALESCE(l.ActualStartDate, l.EarlyStartDate, l.TargetStartDate)) AS StartActualized,
    COALESCE(l.SelectedFinishDate, COALESCE(l.ActualEndDate, l.EarlyEndDate, l.TargetEndDate)) AS FinishActualized
FROM my_db.timeline_reconciliation.TimelineTaskToMdrLinks AS l;

COMMENT ON VIEW my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates IS
'Extends TimelineTaskToMdrLinks with StartActualized/FinishActualized. Prefers persisted Selected*; else recomputes from raw snapshots. Refreshed by refresh_timeline_links_dates_view().';

COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates.StartActualized IS
'[COMPUTED] COALESCE(SelectedStartDate, ActualStart, EarlyStart, TargetStart).';
COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates.FinishActualized IS
'[COMPUTED] COALESCE(SelectedFinishDate, ActualEnd, EarlyEnd, TargetEnd).';
