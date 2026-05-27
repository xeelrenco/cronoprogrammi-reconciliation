-- =============================================================================
-- Timeline reconciliation — creazione da zero (MotherDuck / DuckDB)
-- =============================================================================
-- Database di riferimento: my_db  (sostituire se necessario)
-- Schema: timeline_reconciliation
--
-- Pipeline Python:
--   1_classify_timeline_tasks.py      → TimelineTasksClassified
--   2_prepare_timeline_embeddings.py  → TimelineTaskEmbeddings, TimelineMdrCandidateEmbeddings
--   3_timeline_task_to_mdr_topk.py      → TimelineTaskToMdrCandidates
--   4_resolve_timeline_task_mdr_links.py → TimelineTaskToMdrLinks, TimelineTaskToMdrResolverLlmTopCandidates
--   5_generate_timeline_reconciliation_report.py → lettura tabelle (report Excel)
--
-- Ricreazione completa su DB esistente:
--   DROP SCHEMA IF EXISTS my_db.timeline_reconciliation CASCADE;
-- poi eseguire di nuovo questo script.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS my_db.timeline_reconciliation;

COMMENT ON SCHEMA my_db.timeline_reconciliation IS
'Riconciliazione cronoprogramma Primavera ↔ MDR/RACI: classificazione task, embedding, Top-K, link finali LLM. Date v3 minimal (actualized).';

-- =============================================================================
-- 1. TimelineTasksClassified — script 1_classify_timeline_tasks.py
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
'Una riga per task del cronoprogramma (foglio TASK). Classificazione ENG_DOC / OTHER. Date core per COALESCE actualized (Actual→Early→Target). Input per step 2–4.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TimelineName IS
'Nome timeline (di solito nome file XER/Excel). Filtrare con MDR Mdr_code_name_ref.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ProjectCode IS
'Codice commessa estratto da TimelineName (es. 7910).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskRowId IS
'Indice riga nel foglio TASK. Chiave con TimelineName.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskCode IS
'Codice attività Primavera se presente.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskName IS
'Nome attività dal cronoprogramma.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.WbsName IS
'Nome WBS risolto da PROJWBS.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.EarlyStartDate IS
'Early start grezzo — 2° priorità in actualized.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.EarlyEndDate IS
'Early end/finish grezzo — 2° priorità in actualized (fine).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ActualStartDate IS
'Actual start grezzo — 1° priorità in actualized.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ActualEndDate IS
'Actual end grezzo — 1° priorità in actualized (fine).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TargetStartDate IS
'Target start grezzo — fallback se Actual e Early assenti.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TargetEndDate IS
'Target end grezzo — fallback se Actual e Early assenti (fine).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskText IS
'Testo per embedding/matching (nome task, WBS, classe).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskClass IS
'ENG_DOC = documento ingegneria; OTHER = escluso dal matching MDR.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskClassConfidence IS
'HIGH, MEDIUM, LOW.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.TaskClassReason IS
'Motivazione breve della classificazione.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ClassifierModel IS
'Modello LLM o regola (es. doc_prefix_rule).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.ClassifierPromptVersion IS
'Versione prompt classificatore.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.CreatedAt IS
'Inserimento riga.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.UpdatedAt IS
'Ultimo aggiornamento.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTasksClassified.CreatedBy IS
'Script creatore (es. 1_classify_timeline_tasks.py).';

-- =============================================================================
-- 2. TimelineTaskEmbeddings — script 2_prepare_timeline_embeddings.py
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
'Embedding vettoriale per task ENG_DOC. Usato dallo step 3 per similarità coseno con candidati MDR.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TimelineName IS
'Timeline del task.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.ProjectCode IS
'Codice commessa.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TaskRowId IS
'Join a TimelineTasksClassified (TimelineName, TaskRowId).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TaskText IS
'Testo effettivamente embeddato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.EmbeddingModel IS
'Modello embedding (es. text-embedding-3-small).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.TextHash IS
'Hash di TaskText per refresh selettivo.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.Embedding IS
'Vettore float32 normalizzato L2 (BLOB).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.Dim IS
'Dimensione vettore.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.CreatedAt IS
'Inserimento.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.UpdatedAt IS
'Ultimo refresh embedding.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskEmbeddings.CreatedBy IS
'Script creatore (es. 2_prepare_timeline_embeddings.py).';

-- =============================================================================
-- 3. TimelineMdrCandidateEmbeddings — script 2_prepare_timeline_embeddings.py
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
'Embedding per documenti MDR della stessa timeline con ConsolidatedDecisionType = MATCH (vista MDR consolidata). Pool candidati per Top-K.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.TimelineName IS
'Timeline = Mdr_code_name_ref del progetto.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ProjectCode IS
'Codice commessa.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.MdrDocumentTitle IS
'Titolo documento MDR storico.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.MdrTitleKey IS
'Chiave normalizzata titolo MDR.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedDecisionType IS
'Di solito MATCH.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedTitleKey IS
'TitleKey RACI finale.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedRaciTitle IS
'Titolo RACI finale.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedConfidence IS
'Confidenza consolidata MDR→RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedReason IS
'Motivo decisione consolidata.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ConsolidatedSource IS
'Origine (es. judge_3_3, recovery_3_4).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.EffectiveDescription IS
'Descrizione RACI effettiva per il retrieval.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.DisciplineName IS
'Disciplina RACI (testo).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.TypeName IS
'Tipo documento RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CategoryDescription IS
'Categoria RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.ChapterName IS
'Capitolo RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CandidateText IS
'Testo embeddato (MDR + contesto RACI).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.EmbeddingModel IS
'Modello embedding.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.TextHash IS
'Hash CandidateText.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.Embedding IS
'Vettore float32 (BLOB).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.Dim IS
'Dimensione vettore.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CreatedAt IS
'Inserimento.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.UpdatedAt IS
'Ultimo refresh.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineMdrCandidateEmbeddings.CreatedBy IS
'Script creatore.';

-- =============================================================================
-- 4. TimelineTaskToMdrCandidates — script 3_timeline_task_to_mdr_topk.py
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
'Top-K retrieval semantico task ENG_DOC → MDR. Solo evidenza; non è la verità di business (vedi Links).';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TimelineName IS
'Timeline del task.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.ProjectCode IS
'Codice commessa.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TaskRowId IS
'Task sorgente.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TaskName IS
'Nome task (audit).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.WbsName IS
'WBS (audit).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.MdrDocumentTitle IS
'Titolo MDR candidato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.MdrTitleKey IS
'Chiave MDR.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.ConsolidatedTitleKey IS
'TitleKey RACI del candidato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.ConsolidatedRaciTitle IS
'Titolo RACI del candidato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.Similarity IS
'Similarità coseno (0–1).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.Rank IS
'Rank nel Top-K per (TimelineName, TaskRowId); 1 = migliore.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.EmbeddingModel IS
'Modello embedding usato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.RetrievalMethod IS
'Metodo (es. embedding_cosine_topk).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.TaskTextHash IS
'Tracciabilità verso TimelineTaskEmbeddings.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.CandidateTextHash IS
'Tracciabilità verso TimelineMdrCandidateEmbeddings.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.CreatedAt IS
'Inserimento.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrCandidates.CreatedBy IS
'Script creatore (es. 3_timeline_task_to_mdr_topk.py).';

-- =============================================================================
-- 5. TimelineTaskToMdrLinks — script 4_resolve_timeline_task_mdr_links.py
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
'Link finali task ENG_DOC ↔ documento MDR. Più righe per task se più documenti. Date operative in Selected* (actualized).';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TimelineName IS
'Timeline sorgente.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ProjectCode IS
'Codice commessa.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskRowId IS
'Task cronoprogramma.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskCode IS
'Codice attività.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskName IS
'Nome attività.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.WbsName IS
'WBS.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.EarlyStartDate IS
'Snapshot early start al momento del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.EarlyEndDate IS
'Snapshot early end al momento del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ActualStartDate IS
'Snapshot actual start al momento del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ActualEndDate IS
'Snapshot actual end al momento del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TargetStartDate IS
'Snapshot target start al momento del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TargetEndDate IS
'Snapshot target end al momento del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.SelectedStartDate IS
'Data inizio actualized (Actual→Early→Target) — uso operativo: ordine documenti, giorni/uomo.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.SelectedFinishDate IS
'Data fine actualized (Actual→Early→Target).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskClass IS
'Classe task (di solito ENG_DOC).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskClassConfidence IS
'Confidenza classificazione.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.TaskClassReason IS
'Motivo classificazione.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.MdrDocumentTitle IS
'Titolo MDR collegato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.MdrTitleKey IS
'Chiave MDR collegato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkRank IS
'Priorità tra link dello stesso task (1 = preferito).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkScore IS
'Confidenza LLM del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkMethod IS
'Metodo (es. embedding_topk_llm_resolver).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.LinkReason IS
'Motivazione breve del link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedDecisionType IS
'Decisione MDR→RACI (MATCH atteso).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedTitleKey IS
'TitleKey RACI finale.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedRaciTitle IS
'Titolo RACI finale.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedConfidence IS
'Confidenza consolidata.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedReason IS
'Motivo consolidato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.ConsolidatedSource IS
'Layer consolidamento (judge_3_3 / recovery_3_4).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.CreatedAt IS
'Inserimento link.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrLinks.CreatedBy IS
'Script creatore (es. 4_resolve_timeline_task_mdr_links.py).';

-- =============================================================================
-- 6. TimelineTaskToMdrResolverLlmTopCandidates — script 4 (shortlist LLM)
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
'Shortlist ordinata dal resolver LLM (top_candidates), distinta dai link finali e dal rank di retrieval.';

COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.TimelineName IS
'Timeline del task.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ProjectCode IS
'Codice commessa.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.TaskRowId IS
'Task sorgente.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.EmbeddingModel IS
'Modello embedding della run resolver.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CandidateRankWithinResolver IS
'Ordine preferenza LLM (1 = migliore) nella shortlist.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.RetrievalRank IS
'Rank nel Top-K retrieval (candidate_id verso TimelineTaskToMdrCandidates).';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.MdrDocumentTitle IS
'Titolo MDR candidato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.MdrTitleKey IS
'Chiave MDR.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ConsolidatedTitleKey IS
'TitleKey RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ConsolidatedRaciTitle IS
'Titolo RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.RetrievalSimilarity IS
'Similarità embedding al momento del retrieval.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.LlmConfidence IS
'Confidenza LLM sulla plausibilità del candidato.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.WhyPlausible IS
'Testo LLM: perché il candidato è plausibile.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.EffectiveDescription IS
'Descrizione RACI usata nel contesto.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.DisciplineName IS
'Disciplina RACI.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.TypeName IS
'Tipo documento.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CategoryDescription IS
'Categoria.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.ChapterName IS
'Capitolo.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CreatedAt IS
'Timestamp inserimento shortlist.';
COMMENT ON COLUMN my_db.timeline_reconciliation.TimelineTaskToMdrResolverLlmTopCandidates.CreatedBy IS
'Script creatore.';

-- =============================================================================
-- 7. View date — timeline_reconciliation_common.refresh_* (e analisi SQL)
-- =============================================================================

CREATE OR REPLACE VIEW my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates AS
SELECT
    c.*,
    COALESCE(c.ActualStartDate, c.EarlyStartDate, c.TargetStartDate) AS StartActualized,
    COALESCE(c.ActualEndDate, c.EarlyEndDate, c.TargetEndDate) AS FinishActualized
FROM my_db.timeline_reconciliation.TimelineTasksClassified AS c;

COMMENT ON VIEW my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates IS
'Classified + StartActualized/FinishActualized (Actual→Early→Target).';

COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates.StartActualized IS
'Inizio actualized calcolato da grezze.';
COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTasksClassified_Dates.FinishActualized IS
'Fine actualized calcolata da grezze.';

CREATE OR REPLACE VIEW my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates AS
SELECT
    l.*,
    COALESCE(l.SelectedStartDate, COALESCE(l.ActualStartDate, l.EarlyStartDate, l.TargetStartDate)) AS StartActualized,
    COALESCE(l.SelectedFinishDate, COALESCE(l.ActualEndDate, l.EarlyEndDate, l.TargetEndDate)) AS FinishActualized
FROM my_db.timeline_reconciliation.TimelineTaskToMdrLinks AS l;

COMMENT ON VIEW my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates IS
'Link finali + date actualized (Selected* o COALESCE grezze).';

COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates.StartActualized IS
'Inizio actualized: SelectedStartDate se presente, altrimenti COALESCE grezze.';
COMMENT ON COLUMN my_db.timeline_reconciliation.v_TimelineTaskToMdrLinks_Dates.FinishActualized IS
'Fine actualized: SelectedFinishDate se presente, altrimenti COALESCE grezze.';
