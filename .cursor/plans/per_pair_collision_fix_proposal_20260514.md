# Goal

Eliminate `collision_multiple_batches` for research snapshot backfill by fixing the underlying identity ambiguity in `embedding_batch` lineage, so snapshots `6/9/10` become backfillable against one unambiguous, correct batch per engine.

# Context

- Current detector (`clustering_analysis_backfill.embedding_batches_for_lineage_and_engine`) keys by:
  - `source_dataframe_group_id`
  - `entry_max` (`source_dataframe_row_count`)
  - `embedding_engine`
  - optional provider filter
- Investigation result (`scratch/root_cause_collision_multiple_batches_20260514.md`):
  - `63/63` collisions are **bucket (a) re-embed with same key**.
  - No provider/representation/dimension/key-version drift in collision set.
  - Duplicate rows are true same-key duplicates on research dataframes:
    - snapshot `6` lineage (`source_dataframe_group_id=5`, `entry_max=2086`) -> **2 candidates** per colliding engine
    - snapshots `9/10` lineage (`source_dataframe_group_id=8`, `entry_max=324`) -> **5 candidates** per colliding engine
  - For `60/63` pairs, prior successful runs consistently used exactly one candidate batch (the newest by id).
  - Remaining `3/63` are qwen pairs with no prior run history (still same-key duplicate batches).
- Constraint from request:
  - Do not hide ambiguity by “pick latest/first”.
  - Fix root cause; no flag-gated workaround.

# Approach

Use a **two-track remediation**:

1. **Primary fix now: data cleanup for same-key duplicate `embedding_batch` rows (bucket a)**  
   Make lineage identity unique again by removing provably redundant duplicate rows after backup + dry-run auditing.

2. **Prevent recurrence: choose (a) extend existing singleflight with persistence-time dedupe**  
   Keep caller behavior unchanged and make embed-stage writes idempotent by reusing an existing same-key batch when one appears during/after compute.

Rationale (decision between requested options):

- Because all collisions are bucket (a), lineage-key expansion (provider/representation/dimension/key_version) does not solve this dataset: those fields are equal across duplicate candidates.
- Existing singleflight in embedding services dedupes **per-text embedding API calls** (request-hash / cache lease), but does **not** protect `embedding_batch` group creation in `pipeline/embed.py`.
- We choose **(a)** over **(b)** for this phase: graceful idempotent convergence at stage persistence avoids introducing new hard-failure surfaces for current callers while still preventing duplicate same-key batches.
- DB-level UNIQUE INDEX remains a possible later hardening step after metadata normalization, but is not the primary fix for this incident.

# Steps

1. **Pre-cleanup safety checkpoint**
   - Use the canonical DB backup flow from `docs/runbooks/README.md` (section: one-command full-state backup) with explicit, local campaign artifacts:
     1. `python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream --output "experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/jetstream_pre_cleanup.dump"`
     2. `python scripts/backup_jetstream_full_state.py --skip-artifact-backup --dump-path "experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/jetstream_pre_cleanup.dump" --receipt-path "experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/backup_receipt.json"`
   - Treat `experimental_results/backfill_per_pair/20260514T085641Z/pre_cleanup/db/backup_receipt.json` as the mandatory backup receipt before any delete/apply step.
   - Export duplicate inventory (read-only): grouped by
     - `source_dataframe_group_id`
     - `entry_max`
     - `embedding_engine`
     - `provider`
     - `representation`
     - `dataset_key`
     - `key_version`
   - For each duplicate set, annotate:
     - candidate batch ids sorted by created time
     - batch ids referenced by completed legacy analysis runs and/or current `provenanced_runs`.

2. **Create cleanup tool as read/write scratch script (not `scripts/living`)**
   - Path pattern: `scratch/oneoff_fix_embedding_batch_same_key_duplicates.py` (use `oneoff_` to distinguish from `readonly_` probes).
   - Required modes:
     - `--dry-run`: emit JSON/markdown plan only, no writes.
     - `--apply`: perform deletions only after explicit confirmation argument.
   - Required output per candidate deletion:
     - batch id
     - duplicate-set key
     - evidence that it is redundant (same-key twin exists)
     - evidence that it is not chosen canonical survivor.

3. **Canonical survivor selection policy (deterministic and evidence-backed)**
   - For each duplicate set:
     1. If any batch id is referenced by completed analysis runs for that `(snapshot, engine)` lineage, keep that referenced id as survivor.
     2. If multiple referenced ids exist (unexpected), halt that set and require operator adjudication.
     3. If no referenced id exists, keep newest batch id (highest id) as default survivor and mark as “no-prior-run lineage”.
   - Delete only non-survivor duplicates for sets passing safety checks.

4. **Write safeguards before delete**
   - Hard fail if a candidate-to-delete batch is referenced by:
     - `provenanced_runs.metadata_json.embedding_batch_group_id`
     - `provenanced_runs.config_json.embedding_batch_group_id`
     - `provenanced_runs.source_group_id` when it points to `embedding_batch`
     - `group_links` where removing would orphan required lineage semantics
   - If references exist, either:
     - migrate those references to survivor batch (in same transaction), or
     - skip deletion and report blocker.

5. **Post-cleanup verification**
   - Re-run read-only collision inventory query:
     - expected `collision_multiple_batches = 0` for snapshots `6/9/10`.
   - Re-run per-pair dry-run orchestrator:
     - expected formerly-colliding pairs become `clustering-ready` (or `embed-then-cluster` if genuinely missing).
   - Spot-check a representative subset of previously-colliding pairs to confirm unique selected batch equals intended survivor.

6. **Idempotency guard follow-up (integrate with existing singleflight)**
   - Extend the existing singleflight pattern to **matrix-level/stage-level identity** in `pipeline/embed.py` write path:
     - introduce a lease key derived from canonical batch identity  
       `embed_matrix:{dataset_key}:{embedding_engine}:{provider}:{entry_max}:{key_version}`.
     - acquire lease before creating new `embedding_batch` group/artifact.
     - perform **second lookup after lease acquisition** (`find_embedding_matrix_artifact`) so followers can reuse a winner that finished first.
   - Persistence-time dedupe rule:
     - if same-key matrix artifact already exists at write time, return/reuse its `group_id` and URI instead of creating a new group.
     - only lease holder proceeds to create group+artifact when lookup remains empty.
     - non-holders wait/poll via existing lease mechanism and then reuse discovered artifact; on timeout, fail loudly with actionable error.
   - Scope note:
     - existing singleflight in `services/embeddings/service.py` remains for per-text embedding request dedupe;
     - this step adds a separate guard for `embedding_batch` materialization, which is where collision rows are born.
   - Add tests covering:
     - repeated sequential embed calls
     - concurrent embed calls
     - same key with different key components (engine/provider/etc.) remains distinct.

7. **Policy/documentation update**
   - Document embedding idempotency contract in living docs (`docs/DATA_PIPELINE.md` and relevant runbook section) to codify duplicate prevention and operator remediation flow.

# Validation

- **Root-cause elimination**
  - Query result: no duplicate `embedding_batch` rows for same-key tuple on dataframe `5` and `8`.
  - Per-pair dry-run on snapshots `6/9/10/17/18` reports:
    - `collision_multiple_batches == 0` for research snapshots.

- **Backfill readiness**
  - All former 63 collision pairs become resolvable to exactly one batch and proceed to clustering phase.

- **Safety**
  - Dry-run output includes explicit per-batch deletion rationale and would-be survivor.
  - Apply mode logs exact changes and supports rollback via pre-captured backup.

- **Regression prevention**
  - Embed matrix-level singleflight + persistence dedupe tests prevent reintroduction of same-key duplicate batches.

# Non-Goals / Explicit Rejections

- Do **not** change orchestrator collision behavior to “pick first/most recent”.
- Do **not** add a feature flag workaround that preserves ambiguous lineage key semantics.
- Do **not** broaden lineage tuple for this incident as primary fix, because all 63 collisions are same-key duplicates where expanded fields still match.

# Rollout Notes

- Execute cleanup on a maintenance window because duplicate removal can touch provenance-linked records.
- Keep script in `scratch/` as a one-off operational fix; if recurring operational need emerges, promote a hardened tool in a later scoped change.
