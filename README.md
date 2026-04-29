# Timeline Reconciliation Pipeline

Pipeline LLM-first per collegare i cronoprogrammi Primavera ai documenti MDR gia riconciliati con RACI.

## Obiettivo

1. Classificare ogni task in due classi:

- `ENG_DOC` (progresso documentale ingegneria)
- `OTHER`

2. Per i soli task `ENG_DOC`, associare zero, uno o piu documenti MDR gia riconciliati con RACI.

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

- `create_timeline_reconciliation_tables.sql`
  - crea le tabelle `timeline_reconciliation` e i commenti DB
  - non viene eseguito automaticamente dagli script
- `1_classify_timeline_tasks.py`
  - legge `cronoprogrammi/*.xlsx`
  - classifica task in `ENG_DOC` / `OTHER`
  - scrive `TimelineTasksClassified`
- `2_prepare_timeline_embeddings.py`
  - genera embeddings dei task `ENG_DOC`
  - genera embeddings dei candidati MDR gia `MATCH` verso RACI
  - scrive `TimelineTaskEmbeddings` e `TimelineMdrCandidateEmbeddings`
- `3_timeline_task_to_mdr_topk.py`
  - calcola cosine similarity task -> MDR candidate
  - scrive `TimelineTaskToMdrCandidates`
- `4_judge_timeline_task_mdr_links.py`
  - usa LLM sui Top-K semantici
  - scrive i link finali in `TimelineTaskToMdrLinks`

## Esecuzione

Esempio ordine:

```powershell
python .\1_classify_timeline_tasks.py --limit 100 --save-db
python .\2_prepare_timeline_embeddings.py
python .\3_timeline_task_to_mdr_topk.py --top-k 30
python .\4_judge_timeline_task_mdr_links.py --top-k 30
```

Per una sola timeline:

```powershell
python .\2_prepare_timeline_embeddings.py --timeline "8001 - YEREVAN COMBINED CYCLE POWER PLANT 2"
python .\3_timeline_task_to_mdr_topk.py --timeline "8001 - YEREVAN COMBINED CYCLE POWER PLANT 2"
python .\4_judge_timeline_task_mdr_links.py --timeline "8001 - YEREVAN COMBINED CYCLE POWER PLANT 2"
```

## Output

- `output/classification_<commessa>.xlsx`
  - report umano opzionale generato da `1_classify_timeline_tasks.py`
- MotherDuck:
  - `TimelineTasksClassified`
  - `TimelineTaskEmbeddings`
  - `TimelineMdrCandidateEmbeddings`
  - `TimelineTaskToMdrCandidates`
  - `TimelineTaskToMdrLinks`

La tabella finale `TimelineTaskToMdrLinks` puo contenere piu righe per lo stesso task quando un task `ENG_DOC` viene associato a piu documenti MDR.

## Modalita Test

`1_classify_timeline_tasks.py` supporta `--limit` per classificare un campione proporzionale tra i file input.

Esempi:

```powershell
python .\1_classify_timeline_tasks.py --limit 100
python .\1_classify_timeline_tasks.py --limit 100 --save-db
python .\1_classify_timeline_tasks.py --limit 10 --progress-every 5
```
