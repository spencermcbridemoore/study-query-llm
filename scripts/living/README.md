# Scripts Living Lane

This lane is reserved for actively maintained script entrypoints.

- `backfill_all_variant_clustering_analysis.py` — plan/execute registry-wide `pipeline.analyze` runs for a single `embedding_engine` (see `study_query_llm.experiments.clustering_analysis_backfill`). Use `--snapshot-ids-file path.json` (`{"snapshot_ids":[...]}`) for large shard lists.
- `run_per_pair_backfill_orchestrator.py` — strict per-pair `(snapshot_id, embedding_engine)` orchestrator for snapshots `6/9/10/17/18`: UNION pair inventory, per-pair classification (`clustering-ready` / `embed-then-cluster` / `infeasible`), phase0 embedding workers, phase1 clustering workers, bounded retries, and acceptance artifacts (`orchestrator_summary.json`, `infeasible_pairs.json`, `verification_summary.json`, `ACCEPTANCE_SUMMARY.md`).
- `run_per_pair_backfill_worker.py` — one worker process for one pair with isolated per-pair work layout; supports `phase0`, `phase1`, and `all`.
- `export_sweep_selection_curve_metrics_csv.py` — CSV export of per-`k` metrics from ``*_selection_curve.json`` artifacts for registry `sweep_select` methods (`study_query_llm.experiments.selection_curve_export`).

During migration, many active scripts remain at the root `scripts/` path for command
stability. A later pass can relocate them here after wrapper coverage and doc parity
are validated.
