# Per-Pair Collision Implementation Checklist (No Runtime Code Changes Yet)

Date: 2026-05-14  
Related proposal: `.cursor/plans/per_pair_collision_fix_proposal_20260514.md`  
Related root-cause report: `scratch/root_cause_collision_multiple_batches_20260514.md`

## Singleflight Verification (Requested)

- [x] Verified `pipeline/embed.py` around the pre-hit and matrix write path (`find_embedding_matrix_artifact` + `store_embedding_matrix`).
- [x] Verified `services/embeddings/service.py` singleflight implementation.
- [x] Conclusion:
  - `singleflight_lease_seconds` is passed from `pipeline/embed.py` to `fetch_embeddings_async` and into `EmbeddingService`.
  - The lease is used in `EmbeddingService._wait_for_cache_or_lease(...)` keyed by **request hash** inside `get_embedding(...)` (per-text request layer).
  - The matrix-level path (`pipeline/embed.py` stage pre-check and final `embedding_batch` group/artifact creation) is **not** protected by that lease.
  - In chunked batch mode (`get_embeddings_batch(..., chunk_size=...)`), the per-request lease path is not the primary path, which further reinforces the need for matrix-level dedupe.

## Phase 0: Baseline Freeze

- [ ] Preserve current dry-run artifacts as baseline:
  - `experimental_results/backfill_per_pair/20260514T085641Z/`
- [ ] Record current expected baseline metrics in operator notes:
  - `63` collision pairs total
  - `21` collisions each on snapshots `6/9/10`
  - `0` collisions on snapshots `17/18`

## Phase 1: Backup Gate (Must Pass Before Any Apply)

- [ ] Create DB dump to explicit campaign path:
  - `python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream --output "experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/jetstream_pre_cleanup.dump"`
- [ ] Run backup orchestration (DB-only) with explicit receipt:
  - `python scripts/backup_jetstream_full_state.py --skip-artifact-backup --dump-path "experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/jetstream_pre_cleanup.dump" --receipt-path "experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/backup_receipt.json"`
- [ ] Verify receipt exists and reports success before continuing:
  - `experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/backup_receipt.json`

## Phase 2: Duplicate Inventory Export (Read-Only)

- [ ] Add read-only inventory script:
  - `scratch/readonly_embedding_batch_duplicate_inventory.py`
- [ ] Inventory output (under campaign path):
  - `experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/inventory/duplicate_sets.json`
  - `experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/inventory/duplicate_sets.csv`
- [ ] Group key for inventory rows:
  - `source_dataframe_group_id`, `entry_max`, `embedding_engine`, `provider`, `representation`, `dataset_key`, `key_version`
- [ ] Include per-candidate metadata:
  - `batch_id`, `created_at`, reference counts from `provenanced_runs` / `group_links`

## Phase 3: Cleanup Tool (One-Off, Read/Write Scratch Script)

- [ ] Add one-off tool:
  - `scratch/oneoff_fix_embedding_batch_same_key_duplicates.py`
- [ ] CLI contract:
  - `--dry-run` (required first)
  - `--apply`
  - `--receipt-path` (must point to backup receipt)
  - explicit confirmation token for apply
- [ ] Survivor selection policy implementation:
  - if referenced by prior completed analysis runs in a duplicate set, keep referenced ID
  - if multiple referenced IDs in same set, halt set as conflict
  - if no references, keep newest ID (highest batch ID)
- [ ] Hard safety guards before delete:
  - block delete if unresolved references exist in:
    - `provenanced_runs.metadata_json.embedding_batch_group_id`
    - `provenanced_runs.config_json.embedding_batch_group_id`
    - `provenanced_runs.source_group_id` (when embedding_batch)
    - required `group_links`

## Phase 4: Dry-Run Review Gate

- [ ] Run cleanup tool in `--dry-run` mode only.
- [ ] Validate dry-run report includes per-batch rationale:
  - duplicate-set key
  - chosen survivor
  - delete candidates
  - reference migration/skip reason
- [ ] Obtain operator sign-off on dry-run output before any `--apply`.

## Phase 5: Recurrence Prevention (Matrix-Level Dedupe Integration)

- [ ] Implement matrix-level dedupe guard in `pipeline/embed.py` (separate from per-text request singleflight).
- [ ] Add matrix lease key:
  - `embed_matrix:{dataset_key}:{embedding_engine}:{provider}:{entry_max}:{key_version}`
- [ ] Flow requirements:
  - acquire matrix lease
  - second lookup after lease acquisition
  - only lease holder may create new `embedding_batch` group/artifact
  - followers wait/poll and reuse winner result
  - timeout path fails loudly with actionable diagnostics
- [ ] Ensure behavior applies regardless of chunked/non-chunked embedding fetch mode.
- [ ] Phase 5 gate (must pass before proceeding to cleanup apply):
  - [ ] Sequential idempotency test: repeated embed call returns same batch/artifact identity.
  - [ ] Concurrency test: parallel embed invocations on same matrix key produce one batch.
  - [ ] Chunked-mode test: matrix-level dedupe still prevents duplicates when `chunk_size` is set.
  - [ ] Distinct-key test: differing engine/provider/entry_max/key_version still produce distinct batches.

## Phase 6: Apply Cleanup

- [ ] Execute apply mode with explicit confirmation.
- [ ] Persist apply report under campaign artifacts:
  - `experimental_results/backfill_per_pair/20260514T085641Z/post_cleanup/apply_report.json`
- [ ] Ensure tool reports no unresolved blocked sets in target scope (`6/9/10` lineage keys).

## Phase 7: Post-Cleanup Verification

- [ ] Re-run duplicate inventory read-only script; expect no same-key duplicate sets for affected lineages.
- [ ] Re-run per-pair orchestrator dry-run (no `--phase all`):
  - expect research collisions to drop from `63` to `0`
- [ ] Re-check infeasible output:
  - no `collision_multiple_batches` for snapshots `6/9/10`
- [ ] Spot-check sampled pairs from root-cause report for unique candidate resolution.

## Phase 8: Documentation + Evidence Sync

- [ ] Update living docs to capture matrix-level idempotency contract and one-off remediation pattern:
  - `docs/DATA_PIPELINE.md`
  - relevant runbook sections in `docs/runbooks/`
- [ ] Attach final evidence links in operator notes:
  - backup receipt
  - dry-run and apply reports
  - post-cleanup dry-run summary

## Final Exit Criteria

- [ ] Backup receipt present and valid.
- [ ] Duplicate same-key `embedding_batch` sets removed for research lineages.
- [ ] Per-pair dry-run for snapshots `6/9/10/17/18` has `0` `collision_multiple_batches` on research snapshots.
- [ ] Matrix-level dedupe prevention in place with tests passing.
