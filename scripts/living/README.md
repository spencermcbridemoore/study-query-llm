# Scripts Living Lane

This lane is reserved for actively maintained script entrypoints.

- `backfill_all_variant_clustering_analysis.py` — plan/execute registry-wide `pipeline.analyze` runs for a single `embedding_engine` (see `study_query_llm.experiments.clustering_analysis_backfill`). Use `--snapshot-ids-file path.json` (`{"snapshot_ids":[uuid,...]}`) for large shard lists.
- `run_two_engine_backfill_sharded.py` — operator orchestrator: `preflight` → deterministic round-robin snapshot shards (default 8) → parallel child backfills per engine → `verify` + `ACCEPTANCE_SUMMARY.md`. Defaults: `text-embedding-3-large`, `embed-v-4-0`. Full procedure: [`docs/runbooks/BACKFILL_CLUSTERING_TWO_ENGINE_SHARDED.md`](../../docs/runbooks/BACKFILL_CLUSTERING_TWO_ENGINE_SHARDED.md).
- `export_sweep_selection_curve_metrics_csv.py` — CSV export of per-`k` metrics from ``*_selection_curve.json`` artifacts for registry `sweep_select` methods (`study_query_llm.experiments.selection_curve_export`).

During migration, many active scripts remain at the root `scripts/` path for command
stability. A later pass can relocate them here after wrapper coverage and doc parity
are validated.
