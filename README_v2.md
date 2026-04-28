# match_mdr_primavera_v2

Questo documento descrive la logica del nuovo script `match_mdr_primavera_v2.py`.

## Obiettivo

Il cronoprogramma contiene attivita eterogenee e non e una rappresentazione 1:1 dei documenti MDR.
Per questo motivo il processo e diviso in tre step:

1. classificazione progetto (`STEP 0`: tipo WBS/caso commessa)
2. classificazione righe task (`ENG_DOC` vs `OTHER`)
3. matching MDR solo sulle righe `ENG_DOC`

## Principi applicati

- solo una parte del cronoprogramma e collegabile all'MDR
- le attivita di procurement non vanno matchate ai documenti
- il matching puo essere 1:N o N:N
- il primo passo fondamentale e classificare correttamente le righe

## Input attesi

- cartella cronoprogrammi: `cronoprogrammi/*.xlsx`
- MDR da MotherDuck (config in `config.txt`) con query fissa su `my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All`.
- Per ogni cronoprogramma, il subset MDR viene filtrato su `Mdr_code_name_ref` usando il codice progetto estratto dal nome file (`LIKE '%<codice>%'`).
- sheet task atteso nei cronoprogrammi: `TASK`
### Config sorgente MDR

Nel file `config.txt`:

- `MOTHERDUCK_DB`: es. `my_db`
- `MOTHERDUCK_TOKEN`: token MotherDuck
- cache MDR locale fissata nel codice: `mdr_cache.csv` (non configurabile da `config.txt`)
- la query MDR in modalità MotherDuck è fissata nel codice:
  - `SELECT DISTINCT Document_title AS "Doc. Title", Mdr_code_name_ref AS "Mdr_code_name_ref" ... ORDER BY Document_title`

- colonne MDR attese:
  - `MDR Ref`
  - `Doc. Number` (se manca, viene usata la seconda colonna)
  - `Doc. Title`

## Flusso del processo

### 0) Classificazione progetto (adattiva)

Lo script sceglie una strategia per commessa in base al caso progetto:

- `CASE_1_WBS_USEFUL_STRUCTURED` (es. 8001) -> `TITLE_PLUS_WBS_STRONG`
- `CASE_1B_WBS_DISCIPLINE` (es. 8080/8189) -> `TITLE_PLUS_WBS_DISCIPLINE`
- `CASE_2_WBS_CHAPTER_ITEM` (es. 7910/7920) -> `CHAPTER_TO_ACTIVITY`
- `CASE_3_WBS_INCOMPLETE_MDR_COVERAGE` (es. 8816/8540) -> `PARTIAL_MATCH_ACCEPT_GAPS`
- `CASE_4_WBS_WRONG_MISALIGNED` (es. 7350) -> `TITLE_ONLY_IGNORE_WBS`
- `CASE_5_WBS_USELESS_SINGLE` (es. 7090) -> `TITLE_ACTIVITY_ONLY`
- `CASE_6_MISSING_EXPORT` (es. 6060) -> `NOT_PROCESSABLE`

La classificazione e data-driven:

- il `TASK` viene arricchito con `PROJWBS` (`wbs_id -> wbs_name`, `wbs_short_name`, `parent_wbs_id`)
- le metriche WBS reali guidano la scelta caso/strategia:
  - numero WBS uniche
  - concentrazione sulla WBS principale (`wbs_top_share`)
  - presenza keyword disciplina/chapter nei `wbs_name`

Gli override per commesse note restano disponibili solo come correzione mirata.

### 1) Caricamento e normalizzazione

- normalizza i nomi colonna task (`Task Name` -> `task_name`, `Task Code` -> `task_code`)
- normalizza testo (`lower`, pulizia caratteri speciali, spazi multipli)
- rimuove prefissi brevi tipo `AA-...` dal task name

### 2) Classificazione task

Ogni task riceve:

- `task_class`: `ENG_DOC` / `OTHER`
- `task_subclass_real`: sottoclasse presa dai dati reali Primavera
  - priorita: `wbs_name` (da `PROJWBS`)
  - fallback: `wbs_short_name`
  - ulteriore fallback: `task_type`
- `classification_confidence`: `HIGH` / `MEDIUM` / `LOW`
- `classification_reason`: motivo della regola applicata

Regole principali:

- classificazione `ENG_DOC/OTHER` ibrida: `task_name` + contesto WBS reale (`wbs_name`/`wbs_short_name`)
- keyword procurement hanno priorita assoluta e restano `OTHER` anche se compaiono parole documentali
- segnali WBS engineering aumentano probabilita `ENG_DOC`
- segnali WBS procurement/construction/testing/logistics aumentano probabilita `OTHER`
- se non ci sono indicatori chiari: default conservativo `OTHER`

### 3) Matching MDR (solo ENG_DOC)

Per ogni documento MDR della commessa:

- confronto fuzzy tra `Doc. Title` e `task_name` normalizzati delle sole righe `ENG_DOC`
- associazione bidirezionale supportata:
  - una task puo essere associata a piu documenti
  - un documento puo essere associato a piu task (configurabile)
- parametri in `config.txt`:
  - `MATCH_ALLOW_MULTIPLE_TASKS_PER_DOC`
  - `MATCH_MAX_TASKS_PER_DOC` (`0` = nessun limite, prende tutti quelli effettivi)
  - `MATCH_MIN_SCORE_FOR_ADDITIONAL_TASKS`

### 4) LLM Judge (opzionale, a cascata)

Per ridurre falsi positivi nei casi non chiari:

- `PERFECT` resta deterministico
- casi `MEDIUM` / `LOW` (e alcuni `GOOD`) possono passare al giudizio LLM
- output strutturato: `ACCEPT_MATCH` / `REJECT_MATCH` / `REVIEW`

Attivazione via file di configurazione:

- file: `config.txt`
- template: `config.example.txt`
- se `LLM_ENABLED=true`, lo script usa il judge LLM
- campi principali:
  - `LLM_API_KEY`
  - `LLM_MODEL` (default `gpt-4o-mini`)
  - `LLM_BASE_URL` (default `https://api.openai.com/v1`)
  - `LLM_TOP_K_CANDIDATES` (default `5`)
  - `LLM_GOOD_SCORE_GAP_THRESHOLD` (default `5`)
- se piu task hanno lo stesso score massimo, vengono mantenuti tutti (`match_N = 1 of n`, `2 of n`, ...)
- categorie score:
  - `PERFECT` >= 95
  - `GOOD` >= 85
  - `MEDIUM` >= 70
  - `LOW` < 70

## Output

Gli output sono scritti in `output_v2`.

Per ogni cronoprogramma: `matching_<commessa>.xlsx` con fogli:

- `Summary`: soli indicatori chiave (task totali, task `ENG_DOC`, righe MDR, caso progetto, strategia)
  - include anche una colonna `explanation` per descrivere ogni metrica
- `Task_ENG_DOC_flag`: colonne essenziali task (`task_code`, `task_name`, `wbs_name`, `task_class`, `task_subclass_real`, `classification_confidence`, `classification_reason`, `classification_reason_explanation`)
- `Match_ENG_DOC_only`: risultato matching essenziale (`Doc. Number`, `Document Title`, task migliore/i, score, category)
- `Match_category`: distribuzione categoria match con percentuali
- `LLM_review` (se presente): decisioni LLM sui casi incerti

Aggregato globale:

- `statistiche_aggregate_v2.xlsx` con soli fogli essenziali:
  - `Match_category`
  - `Project_case`
  - `Task_class`

## Limiti attuali

- classificazione basata su keyword (rule-based), non ML
- sensibilita ai testi reali dei task: keyword e priorita possono richiedere tuning
- se una commessa non ha task `ENG_DOC` classificati, i documenti restano con `NO_ENG_DOC_TASKS`

## Suggerimenti di tuning

- aggiornare le keyword sulle naming convention reali di Primavera
- introdurre dizionario sinonimi ITA/ENG per dominio progetto
- validare manualmente un campione e ritarare priorita tra classi

