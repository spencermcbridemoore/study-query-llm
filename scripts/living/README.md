# Scripts Living Lane

This lane is reserved for actively maintained script entrypoints.

- `backfill_all_variant_clustering_analysis.py` — plan/execute registry-wide `pipeline.analyze` runs for a single `embedding_engine` (see `study_query_llm.experiments.clustering_analysis_backfill`). Use `--snapshot-ids-file path.json` (`{"snapshot_ids":[...]}`) for large shard lists.
- `run_two_engine_backfill_sharded.py` — operator orchestrator: `preflight` → deterministic round-robin snapshot shards (default 8) → parallel child backfills per engine → `verify` + `ACCEPTANCE_SUMMARY.md`. Defaults: `text-embedding-3-large`, `embed-v-4-0`. Optional `--snapshot-ids-file` scopes preflight/final verify manifests to a subset of snapshots. Full procedure: [`docs/runbooks/BACKFILL_CLUSTERING_TWO_ENGINE_SHARDED.md`](../../docs/runbooks/BACKFILL_CLUSTERING_TWO_ENGINE_SHARDED.md).
- `run_overnight_fivesnap_backfill.py` — unattended/resumable sequencing for fixed snapshots **6, 9, 10, 21, 22**: intersection `(provider, embedding_engine)` roster → per-pair preflight + `queue.json` (runnable vs `preflight_manifest_blocking_issues`) → timeboxed sequential sharded runs (`work/<pair_id>/`) → `verify-package` (`CONSOLIDATED_SUMMARY.json`, `RESUME_COMMANDS.md`).
- `export_sweep_selection_curve_metrics_csv.py` — CSV export of per-`k` metrics from ``*_selection_curve.json`` artifacts for registry `sweep_select` methods (`study_query_llm.experiments.selection_curve_export`).

During migration, many active scripts remain at the root `scripts/` path for command
stability. A later pass can relocate them here after wrapper coverage and doc parity
are validated.
