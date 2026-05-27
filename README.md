# Timeline Reconciliation Pipeline

Pipeline LLM-first per collegare i cronoprogrammi Primavera ai documenti MDR gia riconciliati con RACI.

## Obiettivo

1. Classificare ogni task in due classi:

- `ENG_DOC` (progresso documentale ingegneria)
- `OTHER`

2. Per i soli task `ENG_DOC`, associare zero, uno o piu documenti MDR gia riconciliati con RACI.

3. Conservare le date Primavera dei task collegati, cosi da poter stimare in seguito le milestone dei documenti MDR/RACI.

La fase MDR usa solo candidati da `my_db.mdr_reconciliation.v_MdrReconciliationResults_Consolidated`
con `ConsolidatedDecisionType = 'MATCH'`, filtrati sulla stessa timeline tramite
`Mdr_code_name_ref = TimelineName`.

## Input

- `config.txt`:
  - `MOTHERDUCK_DB`
  - `MOTHERDUCK_TOKEN`
  - `LLM_API_KEY`
  - `LLM_MODEL`
  - `LLM_BASE_URL`
  - `EMBEDDING_MODEL`
- `cronoprogrammi/*.xlsx`

## Pipeline

Il flusso e diviso in script:

- `sql/00_create_timeline_reconciliation.sql` — schema MotherDuck (creazione da zero, con commenti)
  - crea le tabelle `timeline_reconciliation` e i commenti DB
  - non viene eseguito automaticamente dagli script
- `1_classify_timeline_tasks.py`
  - legge `cronoprogrammi/*.xlsx`
  - classifica task in `ENG_DOC` / `OTHER`
  - estrae e salva in DB solo le date core: `Early*`, `Actual*`, `Target*` (input al actualized)
  - scrive `TimelineTasksClassified`
- `2_prepare_timeline_embeddings.py`
  - genera embeddings dei task `ENG_DOC`
  - genera embeddings dei candidati MDR gia `MATCH` verso RACI
  - usa batching API e refresh incrementale su `TextHash` (ricalcola solo i record cambiati)
  - scrive `TimelineTaskEmbeddings` e `TimelineMdrCandidateEmbeddings`
- `3_timeline_task_to_mdr_topk.py`
  - calcola cosine similarity task -> MDR candidate
  - scrive `TimelineTaskToMdrCandidates` (senza colonne date)
- `4_resolve_timeline_task_mdr_links.py`
  - usa LLM sui Top-K semantici
  - copia le date core da Classified e calcola `SelectedStartDate` / `SelectedFinishDate` (actualized: Actual→Early→Target)
  - valida l'output LLM, applica retry, soglia minima e massimo link per task
  - scrive i link finali in `TimelineTaskToMdrLinks` e la diagnostica locale in `output/resolver_diagnostics_<timestamp>.csv`

## Esecuzione

Esempio ordine:

```powershell
python .\1_classify_timeline_tasks.py --limit 100 --save-db
python .\2_prepare_timeline_embeddings.py
python .\3_timeline_task_to_mdr_topk.py --top-k 30
python .\4_resolve_timeline_task_mdr_links.py --top-k 30 --min-link-confidence 0.35 --max-links-per-task 3
```

Per una sola timeline:

```powershell
python .\2_prepare_timeline_embeddings.py --timeline "8001 - YEREVAN COMBINED CYCLE POWER PLANT 2"
python .\3_timeline_task_to_mdr_topk.py --timeline "8001 - YEREVAN COMBINED CYCLE POWER PLANT 2"
python .\4_resolve_timeline_task_mdr_links.py --timeline "8001 - YEREVAN COMBINED CYCLE POWER PLANT 2" --min-link-confidence 0.35 --max-links-per-task 3
```

Opzioni utili step 2:

```powershell
python .\2_prepare_timeline_embeddings.py --embed-batch-size 256
python .\2_prepare_timeline_embeddings.py --force-refresh
```

Opzioni utili step 4:

```powershell
python .\4_resolve_timeline_task_mdr_links.py --min-link-confidence 0.35
python .\4_resolve_timeline_task_mdr_links.py --max-links-per-task 3
python .\4_resolve_timeline_task_mdr_links.py --workers 4
python .\4_resolve_timeline_task_mdr_links.py --retry-max 2 --retry-backoff-sec 2 --llm-timeout-sec 60
python .\5_generate_timeline_reconciliation_report.py
```

Modalita Batch API per step 4:

```powershell
python .\4_resolve_timeline_task_mdr_links.py --batch-submit --top-k 30 --min-link-confidence 0.35 --max-links-per-task 3
python .\4_resolve_timeline_task_mdr_links.py --batch-collect
```

Oppure submit e collect nello stesso run, attendendo il completamento del batch:

```powershell
python .\4_resolve_timeline_task_mdr_links.py --batch-and-collect --top-k 30 --batch-poll-interval 60
```

La modalita batch salva gli id in `.timeline_resolver_last_batch_ids.json` e il manifest task in `.timeline_resolver_last_batch_manifest.json`.

## Output

- `output/classification_<commessa>.xlsx`
  - report umano opzionale generato da `1_classify_timeline_tasks.py`
- MotherDuck:
  - `TimelineTasksClassified`
  - `TimelineTaskEmbeddings`
  - `TimelineMdrCandidateEmbeddings`
  - `TimelineTaskToMdrCandidates`
  - `TimelineTaskToMdrLinks`
- `output/resolver_diagnostics_<timestamp>.csv`
  - diagnostica locale per task prodotta da `4_resolve_timeline_task_mdr_links.py`

La tabella finale `TimelineTaskToMdrLinks` puo contenere piu righe per lo stesso task quando un task `ENG_DOC` viene associato a piu documenti MDR.

**Date (v3 minimal):** in DB solo `Early*`, `Actual*`, `Target*` + `Selected*` sui link. Schema SQL: cartella `sql/`. Ricreazione: `DROP SCHEMA … CASCADE` + `sql/00_create_timeline_reconciliation.sql`, poi pipeline `1` → `4`.

## Modalita Test

`1_classify_timeline_tasks.py` supporta `--limit` per classificare un campione proporzionale tra i file input.

Esempi:

```powershell
python .\1_classify_timeline_tasks.py --limit 100
python .\1_classify_timeline_tasks.py --limit 100 --save-db
python .\1_classify_timeline_tasks.py --limit 10 --progress-every 5
```
