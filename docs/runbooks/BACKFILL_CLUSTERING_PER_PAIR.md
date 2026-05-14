# Clustering analysis backfill - per-pair workers

Status: living  
Owner: ops-maintainers

This runbook documents the strict per-pair clustering backfill flow:

- One worker process per `(snapshot_id, embedding_engine)` pair
- Pair universe built as `snapshot_ids x engine_union` (UNION semantics)
- Phase 0 embed precompute for `no_embedding_batch` pairs
- Phase 1 clustering execution for `clustering-ready` pairs
- Infeasible-modulo final verification against `scratch/readonly_full_coverage_export.py`

## Prerequisites

- DB URL via `CANONICAL_DATABASE_URL` or `DATABASE_URL`.
- Canonical lane writes available (`SQLLM_WRITE_INTENT=canonical` or explicit constructor intent).
- For OpenRouter pricing budgets, configured OpenRouter credentials are recommended.

## Entry points

- Orchestrator: `scripts/living/run_per_pair_backfill_orchestrator.py`
- Per-pair worker: `scripts/living/run_per_pair_backfill_worker.py`
- Single-engine planner/runner reused by worker logic: `scripts/living/backfill_all_variant_clustering_analysis.py`

## Dry-run inventory and budget

```powershell
python scripts/living/run_per_pair_backfill_orchestrator.py `
  --phase dry-run `
  --snapshot-ids 6 9 10 17 18 `
  --max-workers 8
```

Dry-run writes:

- `pair_universe.json`
- `pair_classification.json`
- `classification_summary.json`
- `infeasible_pairs.json`
- `budget_estimate.json`
- `budget_estimate.md`

under `experimental_results/backfill_per_pair/<UTC-stamp>/`.

## Execute full flow

```powershell
python scripts/living/run_per_pair_backfill_orchestrator.py `
  --phase all `
  --snapshot-ids 6 9 10 17 18 `
  --max-workers 8 `
  --pair-max-retries 2 `
  --accept-budget
```

## Same-Key Embedding Batch Remediation (One-Off)

When per-pair dry-run infeasible output shows `collision_multiple_batches` for
embedding-backed lineages, run the approved one-off cleanup workflow:

1. Backup gate:
   - `python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream --output "<campaign>/pre_cleanup/db/jetstream_pre_cleanup.dump"`
   - `python scripts/backup_jetstream_full_state.py --skip-artifact-backup --dump-path "<campaign>/pre_cleanup/db/jetstream_pre_cleanup.dump" --receipt-path "<campaign>/pre_cleanup/db/backup_receipt.json"`
2. Inventory:
   - `python scratch/readonly_embedding_batch_duplicate_inventory.py --run-stamp <run-stamp>`
3. Dry-run cleanup plan:
   - `python scratch/oneoff_fix_embedding_batch_same_key_duplicates.py --dry-run --run-stamp <run-stamp> --target-lineage <sdf:entry_max> ...`
4. Apply cleanup after operator approval:
   - `python scratch/oneoff_fix_embedding_batch_same_key_duplicates.py --apply --run-stamp <run-stamp> --target-lineage <sdf:entry_max> ... --receipt-path "<campaign>/pre_cleanup/db/backup_receipt.json" --confirm-token APPLY_EMBEDDING_BATCH_SAME_KEY_DUPLICATES`

Expected apply behavior for an approved dry-run: same in-scope set count,
delete-candidate count, and `0` conflict/blocked sets.

Optional controls:

- `--halt-on-first-permanent`
- `--halt-on-failure-rate <fraction>`
- `--provider-price-json <path.json>` where JSON shape is `{ "<engine>": <usd_per_1k_tokens> }`
- `--run-stamp <UTC stamp>`

## Artifacts and acceptance

Final orchestrator artifacts:

- `orchestrator_summary.json`
- `infeasible_pairs.json` (with `missing_method_count` per infeasible pair)
- `verification_summary.json`
- `ACCEPTANCE_SUMMARY.md`

Per-pair artifacts (under `pairs/<pair_id>/`):

- `pair_spec.json`
- `preflight_manifest.json`
- `post_phase0_manifest.json` (when applicable)
- `phase0/phase0_result.json`
- `phase1/preflight_manifest.json`
- `phase1/run_state.json`
- `phase1/completed_keys_cache.json`
- `phase1/execute_stats.json`
- `phase1/final_manifest.json`
- `status.json`
- `logs/*.log`

Acceptance logic:

- Re-runs `scratch/readonly_full_coverage_export.py`
- Requires `grand_completed_cells == grand_expected_cells - sum(infeasible missing_method_count)`
- Requires `zero_coverage_pairs` to be limited to explicit infeasible pairs
- Uses `26` as the embedding-level missing-method expectation for unresolved embedding-level infeasible pairs; partial clustering failures compute actual residual missing methods from final per-pair manifests.
