# Canonical DB inventory - 2026-06-04 (read-only)

Point-in-time, read-only inventory of the canonical Jetstream Postgres
(`study_query_jetstream`), captured over the SSH tunnel. **No writes were issued.**
Captured for plan item 0.3 (`.cursor/plans/post-audit-hardening-v2.md`) and to
settle the "is the race-fix applied to canonical?" question.

Regenerate (tunnel up): `python scripts/probe_postgres_inventory.py --env-var JETSTREAM_DATABASE_URL`.

## Headline

- Canonical is ACTIVE and healthy (8 GB, 438,261 raw_calls). The earlier
  "dormant" framing in `JETSTREAM_STATE_TIMELINE.md` was stale (fixed in plan 1.1).
- The `analysis_request` get-or-create race fix (commit `dd38ddc`, ledger C079) is
  **confirmed applied** on canonical, and canonical is clean: the partial UNIQUE
  index exists, zero duplicate identities, zero half-identity rows. No remediation
  or migration is pending (plan 0.4 / 0.5 are moot for this fix).

## Inventory

- `pg_database_size`: 8003 MB
- public tables (15): `analysis_results`, `call_artifacts`,
  `embedding_cache_entries`, `embedding_cache_leases`, `embedding_vectors`,
  `group_graph_audit_log`, `group_links`, `group_members`, `groups`,
  `method_definitions`, `orchestration_job_dependencies`, `orchestration_jobs`,
  `provenanced_runs`, `raw_calls`, `sweep_run_claims`
- `raw_calls`: 438,261
- groups by type:

  | group_type | count |
  |---|---:|
  | analysis_run | 15,942 |
  | analysis_request | 3,313 |
  | embedding_batch | 523 |
  | summarization_batch | 110 |
  | dataset_snapshot | 26 |
  | custom | 25 |
  | clustering_step | 21 |
  | clustering_run | 21 |
  | dataset_dataframe | 20 |
  | dataset | 20 |
  | clustering_sweep | 19 |
  | clustering_sweep_request | 14 |

## analysis_request integrity (race-fix surface)

- unique index `uq_groups_analysis_request_identity`: **PRESENT** (partial
  functional index over `metadata_json ->> {method_name,input_id,run_key}` WHERE
  `group_type='analysis_request'`).
- duplicate identities (`build_duplicate_probe_sql`): **0**
- identity shape (`num_nulls` of the three fields):
  - `nn=0` (identity-bearing): 3,299
  - `nn=3` (container / empty-metadata): 14
  - `nn=1` or `2` (partial / half-row): **0**  <- plan 2.2 detection: clean
- Conclusion: race-fix fully enforced; the two documented shapes
  (identity-bearing + container) hold; no half-rows exist.

## analysis_run uniqueness (plan 2.3)

- `analysis_run` uses a DISTINCT identity shape, not the `analysis_request`
  triple. metadata keys: `method_name`, `method_version`, `run_key`,
  `snapshot_group_id`, `embedding_batch_group_id`, `representation_type`,
  `parameters`, `request_group_id`, `source_run_key` (15,357 / 15,942).
  `input_id` is NULL in all 15,942 rows (the bulk come from the pipeline
  `analyze` path, not `create_analysis_run_group`).
- Correctly NOT covered by `uq_groups_analysis_request_identity` (different shape;
  the index is scoped to `analysis_request` only).
- `(method_name, run_key)` multiplicity: 15,886 distinct pairs over 15,942 rows;
  53 pairs duplicated; max multiplicity 3.
- **Decision:** duplicate `analysis_run` groups are ACCEPTABLE / INTENDED.
  `analysis_run` is an execution-event record (no insert-or-get); the deduped
  identity lives in `analysis_request`. The rare `(method_name, run_key)`
  collisions reflect re-runs or input/variant differences (the
  `snapshot_group_id` / `embedding_batch_group_id` differ). No index or
  insert-or-get change is warranted.

## Pending cleanup: legacy `embedding_vectors` table (deferred)

`embedding_vectors` appears in the table list above but is **code-retired**: the ORM
model is gone (the CI `rg "EmbeddingVector"` gate passes) and
`RawCallRepository.get_embedding_vectors_by_request_hashes` now serves from
`embedding_cache_entries`. A ready, idempotent drop migration exists
(`db/migrations/drop_embedding_vectors.py`), but it was never run on canonical, so the
physical table lingers.

Dropping it is a DESTRUCTIVE canonical write (`DROP TABLE ... CASCADE`; the migration
drops unconditionally), so it is **deferred** to the 0.4 gated-write discipline:
read-only `SELECT COUNT(*)` + table size on `embedding_vectors` first; if non-trivial,
confirm the data is superseded (per ledger C032 embeddings moved to
`embedding_cache_entries` / blob artifacts); then full-state backup
(`scripts/backup_jetstream_full_state.py`) + explicit approval, then run the drop.
