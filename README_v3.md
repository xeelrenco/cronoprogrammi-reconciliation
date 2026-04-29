# match_mdr_primavera_v3

Versione sperimentale LLM-first.

## Obiettivo fase 1

Classificare ogni task in due classi:

- `ENG_DOC` (progresso documentale ingegneria)
- `OTHER`

Con input LLM:

- `task_name`
- `wbs_name` (hint aggiuntivo)

## Differenza rispetto v2

- niente prefiltraggio keyword per la classificazione
- classificazione affidata direttamente all'LLM
- nessun fuzzy matching
- nessun matching MDR in questa fase

## Input

- `config.txt`:
  - `LLM_API_KEY`
  - `LLM_MODEL`
  - `LLM_BASE_URL`
- `cronoprogrammi/*.xlsx`

Parametri runtime solo da CLI:

- `--limit 100` -> classifica un campione totale di 100 task (proporzionale tra file)
- `--progress-every 25` -> log avanzamento ogni 25 task

## Output

- `output_v3/classify_match_<commessa>.xlsx`
 - `output_v3/classification_<commessa>.xlsx`
  - `Summary`
  - `Task_LLM_classification`
  - `Task_class_stats`
- `output_v3/classification_aggregate_v3.xlsx`
  - include `Sample_distribution` se attivo test campione

## Nota

Questa v3 e focalizzata solo sulla classificazione.
La fase di associazione MDR verra gestita successivamente.

## Modalita test campione (100 task)

Se lanci `--limit 100`:

- lo script distribuisce il campione in modo proporzionale tra i file input
- la prima volta sceglie task random e salva il file campione (`v3_task_sample.csv`)
- ai run successivi riusa lo stesso file, così il test è ripetibile

Esempi:

- full run: `python .\match_mdr_primavera_v3.py`
- test 100 task: `python .\match_mdr_primavera_v3.py --limit 100`
- test 10 task con log frequente: `python .\match_mdr_primavera_v3.py --limit 10 --progress-every 5`
