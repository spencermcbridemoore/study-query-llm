# Scripts Living Lane

This lane is reserved for actively maintained script entrypoints.

- `backfill_all_variant_clustering_analysis.py` — plan/execute registry-wide `pipeline.analyze` runs for a single `embedding_engine` (see `study_query_llm.experiments.clustering_analysis_backfill`). Use `--snapshot-ids-file path.json` (`{"snapshot_ids":[...]}`) for large shard lists.
- `export_sweep_selection_curve_metrics_csv.py` — CSV export of per-`k` metrics from ``*_selection_curve.json`` artifacts for registry `sweep_select` methods (`study_query_llm.experiments.selection_curve_export`).

During migration, many active scripts remain at the root `scripts/` path for command
stability. A later pass can relocate them here after wrapper coverage and doc parity
are validated.
