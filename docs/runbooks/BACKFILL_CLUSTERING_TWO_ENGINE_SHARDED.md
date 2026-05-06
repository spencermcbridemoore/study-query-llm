# Clustering analysis backfill — two engines, sharded workers

Status: living  
Owner: ops-maintainers  

This runbook describes running **all bundled clustering methods** (`iter_algorithm_specs()`) across snapshots for **two embedding engines** using **N parallel processes** (default **8**) without overlapping snapshot assignment.

## Prerequisites

- DB URL via `CANONICAL_DATABASE_URL` or `DATABASE_URL` (see [`README.md`](README.md) contract).
- Embedding batches exist for each target `(source_dataframe_group_id, entry_max)` and engine string (matches `embedding_batch.metadata_json.embedding_engine`).
- Resolve **preflight blockers** before execute: no lineage collisions (`>1` batch per snapshot/engine), no duplicate manifest keys, at least one eligible snapshot pair.

## Locked defaults

Orchestrator [`scripts/living/run_two_engine_backfill_sharded.py`](../../scripts/living/run_two_engine_backfill_sharded.py) defaults:

- Engine A: `text-embedding-3-large`
- Engine B: `embed-v-4-0`

Override with `--engines ...`.

## One-shot workflow

From repo root (PowerShell):

```powershell
python scripts/living/run_two_engine_backfill_sharded.py all --shard-count 8
```

Artifacts land under `experimental_results/backfill_manifests/two_engine_<UTC>/` (gitignored). Copy `verify_report.json` and `ACCEPTANCE_SUMMARY.md` elsewhere if you need them in version control.

### Dry rehearsal (no `analyze` writes)

```powershell
python scripts/living/run_two_engine_backfill_sharded.py all --shard-count 8 --dry-run
```

## Phased workflow (restart-friendly)

Use the same `--work-dir` path across phases.

1. **Preflight** — manifests only; aborts on blockers  
   `python scripts/living/run_two_engine_backfill_sharded.py preflight --work-dir <DIR>`  

2. **Shard** — writes `<slug>.shard{i}.snapshot_ids.json`  
   `python scripts/living/run_two_engine_backfill_sharded.py shard --work-dir <DIR> --shard-count 8`  

3. **Execute** — up to 8 concurrent child processes **per engine** (engines run sequentially)  
   `python scripts/living/run_two_engine_backfill_sharded.py execute --work-dir <DIR>`  

4. **Verify** — full-engine manifest + DB-refreshed coverage  
   `python scripts/living/run_two_engine_backfill_sharded.py verify --work-dir <DIR>`  

5. **Acceptance** — markdown summary from saved verify report  
   `python scripts/living/run_two_engine_backfill_sharded.py acceptance --work-dir <DIR>`  

## Acceptance criteria

- `verify_report.json` shows `coverage.coverage_complete: true` and `missing_keys_remaining: 0` per engine.
- `ACCEPTANCE_SUMMARY.md` written under `--work-dir`.

## Related code

- [`scripts/living/backfill_all_variant_clustering_analysis.py`](../../scripts/living/backfill_all_variant_clustering_analysis.py) — single-engine planner/executor; `--snapshot-ids-file` for shard inputs.
- [`src/study_query_llm/experiments/clustering_analysis_backfill.py`](../../src/study_query_llm/experiments/clustering_analysis_backfill.py) — manifest math, sharding helpers, resume state.
