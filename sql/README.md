# SQL — `timeline_reconciliation` schema

Greenfield DDL for the Primavera timeline ↔ MDR reconciliation pipeline (scripts `1`–`5`).

All `COMMENT ON` metadata is in **English**, tagged for MCP/SQL assistants (`[KEY]`, `[DATE_OPS]`, writers/readers, join hints).

## File

| File | Purpose |
|------|---------|
| `00_create_timeline_reconciliation.sql` | `CREATE SCHEMA`, 6 tables, 2 views, full `COMMENT ON` |

## Run (MotherDuck)

1. Replace `my_db` if your `config.txt` uses another database name.
2. **Empty install:** execute the whole script once.
3. **Rebuild:**

```sql
DROP SCHEMA IF EXISTS my_db.timeline_reconciliation CASCADE;
```

Then re-run `00_create_timeline_reconciliation.sql` and the Python pipeline.

## Object map (for MCP)

| Object | Writer script | Primary key / grain |
|--------|---------------|---------------------|
| `TimelineTasksClassified` | `1_classify` | `(TimelineName, TaskRowId)` |
| `TimelineTaskEmbeddings` | `2_prepare` | `(TimelineName, TaskRowId, EmbeddingModel)` |
| `TimelineMdrCandidateEmbeddings` | `2_prepare` | per timeline + MDR + model |
| `TimelineTaskToMdrCandidates` | `3_topk` | `(TimelineName, TaskRowId, Rank)` |
| `TimelineTaskToMdrLinks` | `4_resolve` | link row; latest by `CreatedAt` |
| `TimelineTaskToMdrResolverLlmTopCandidates` | `4_resolve` | PK on table |
| `v_TimelineTasksClassified_Dates` | SQL view | classified + actualized |
| `v_TimelineTaskToMdrLinks_Dates` | SQL view | links + actualized |

## Schedule model (v3)

- **Raw (stored):** `Early*`, `Actual*`, `Target*`
- **Operational (links):** `SelectedStartDate`, `SelectedFinishDate` = actualized `Actual → Early → Target`
