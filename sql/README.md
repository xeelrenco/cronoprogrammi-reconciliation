# SQL — schema `timeline_reconciliation`

Creazione **da zero** del database usato dalla pipeline cronoprogramma (script `1` … `5`).

## File

| File | Contenuto |
|------|-----------|
| `00_create_timeline_reconciliation.sql` | Schema, 6 tabelle, 2 view, `COMMENT ON` per tabella/vista/colonna |

## Esecuzione (MotherDuck)

1. Sostituisci `my_db` nel file se il database in `config.txt` è diverso.
2. Su database **vuoto** (prima installazione): esegui l’intero script una volta.
3. Per **ricreare da zero** su DB già popolato:

```sql
DROP SCHEMA IF EXISTS my_db.timeline_reconciliation CASCADE;
```

Poi riesegui `00_create_timeline_reconciliation.sql` e rilancia la pipeline Python.

## Mapping script → oggetti

| Script | Tabelle / view |
|--------|----------------|
| `1_classify_timeline_tasks.py` | `TimelineTasksClassified` |
| `2_prepare_timeline_embeddings.py` | `TimelineTaskEmbeddings`, `TimelineMdrCandidateEmbeddings` |
| `3_timeline_task_to_mdr_topk.py` | `TimelineTaskToMdrCandidates` |
| `4_resolve_timeline_task_mdr_links.py` | `TimelineTaskToMdrLinks`, `TimelineTaskToMdrResolverLlmTopCandidates` (+ refresh view date via common) |
| `5_generate_timeline_reconciliation_report.py` | legge tabelle (non le view date) |
| `timeline_reconciliation_common.py` | `v_TimelineTasksClassified_Dates`, `v_TimelineTaskToMdrLinks_Dates` |

## Date (modello v3 minimal)

- **Classified / Links (grezze):** `Early*`, `Actual*`, `Target*`
- **Links (operative):** `SelectedStartDate`, `SelectedFinishDate` — actualized = `Actual → Early → Target`
- **View:** `StartActualized`, `FinishActualized` calcolate con la stessa regola
