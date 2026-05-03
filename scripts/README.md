# Scripts Directory

This directory contains standalone script entrypoints and script helper modules.

## What Belongs Here

This directory is a compatibility/operations surface, not the canonical runtime layer.

- **Tier B (`scripts/`)**: command-stable wrappers, one-off ops utilities, and lane-governed historical tooling.
- **Tier A (`src/study_query_llm/`)**: canonical orchestration, pipeline stages, job runners, and execution logic.

Decision tree:
- If code changes job orchestration, worker behavior, sweep planning, or pipeline-stage method behavior -> put it in `src/study_query_llm/**`.
- If code is a standalone operator utility (DB checks, migration helper, inventory probe) or a wrapper entrypoint -> `scripts/**` is appropriate.

Concrete examples:
- `scripts/register_clustering_methods.py` belongs here (operator/setup utility).
- `src/study_query_llm/pipeline/clustering/kmeans_fixed_k_runner.py` belongs in Tier A runtime (method execution logic).

Worked counter-example:
- Adding a new clustering method should update `src/study_query_llm/pipeline/clustering/` (registry + runner + tests), not introduce a new `scripts/run_*.py` orchestration entrypoint.

Point-of-temptation rationale:
- Keep boundary guidance in wrapper docstrings and this README so humans/agents see the rule where edits are likely, instead of relying on distant architecture docs only.

Known transitional note:
- `scripts/run_local_300_2datasets_worker.py` is currently a thin compatibility wrapper; canonical worker logic lives in `src/study_query_llm/experiments/sweep_worker_main.py`.

Source-of-truth guidance:
- See `AGENTS.md` for full tier vocabulary, terminology lock, and transitional-violation review cadence.

## Framework Lanes

This repository applies the same governance framework used by docs:

- **living**: current operator/automation entrypoints
- **history**: experiment/chronology scripts kept for reproducibility
- **deprecated**: retired compatibility surface with explicit replacements
- **internal**: helper modules not intended as direct operator entrypoints

Lane directories:

- [`scripts/living/`](living/README.md)
- [`scripts/history/`](history/README.md)
- [`scripts/deprecated/`](deprecated/README.md)
- [`scripts/internal/`](internal/README.md)

## Current Lane Assignments (Cleanup Passes)

The following no-PCA/experimental scripts were moved into `history`:

- `scripts/history/analysis/analyze_dataset_lengths.py`
- `scripts/history/analysis/analyze_dbpedia_character_length_grid.py`
- `scripts/history/analysis/analyze_estela_lengths.py`
- `scripts/history/analysis/plot_no_pca_50runs.py`
- `scripts/history/analysis/plot_no_pca_multi_embedding.py`
- `scripts/history/experiments/run_custom_full_categories_sweep.py`
- `scripts/history/experiments/run_experimental_sweep.py`
- `scripts/history/experiments/run_no_pca_50runs_sweep.py`
- `scripts/history/experiments/run_no_pca_multi_embedding_sweep.py`
- `scripts/history/experiments/test_no_pca_sweep.py`

Move set v1.2 moved one-off probes/utilities into `scripts/history/one_offs/`
(for example: `check_db_empty.py`, `check_all_data.py`, `check_embedding_calls.py`,
`check_rate_limits.py`, TEI/OpenRouter probes, and export/migration one-offs).

Thin compatibility wrappers remain at their original root paths under `scripts/` and
print a deprecation notice when executed. **Move set v1.1:** no-PCA/history forwarding
scripts delegate to [`scripts/deprecated/`](deprecated/) (canonical relocated surface);
root -> `scripts/deprecated/*` -> `scripts/history/*` (or `scripts/run_pca_kllmeans_sweep.py`
for the legacy PCA name). Incident recovery scripts live under
[`scripts/history/sweep_recovery/`](history/sweep_recovery/); root `archive_pre_fix_runs.py`
/ `label_pre_fix_runs.py` forward there.

Historical compatibility stubs for names that still appear in historical docs
(`scripts/pca_kllmeans_sweep.py`, `scripts/migrate_v1_to_v2.py`) have implementations
under `scripts/deprecated/` with root wrappers preserved.

## Canonical DB Ops Matrix

Source-of-truth policy and URL contract live in [`docs/runbooks/README.md`](../docs/runbooks/README.md).

| Script | Primary intent | Source assumptions | Target assumptions | Risk level |
|---|---|---|---|---|
| `dump_postgres_for_jetstream_migration.py` | Create custom-format `pg_dump` snapshot (`-Fc`) | Uses explicit source via `--source-url`, `--from-local`, `--from-jetstream`, or env fallback | Writes `.dump` file under `pg_migration_dumps/` | Low (read-only DB, sensitive artifact output) |
| `restore_pg_dump_to_local_docker.py` | Restore `.dump` into local clone DB | `.dump` file created by `pg_dump -Fc` | `LOCAL_DATABASE_URL` by default (or explicit URL) | High (drops/recreates target DB unless `--skip-recreate`) |
| `sync_from_online.py` | Incrementally copy v2 rows from source DB to local clone | `SOURCE_DATABASE_URL`/`DATABASE_URL` or `--online-url` is source | `LOCAL_DATABASE_URL`/`--local-url` is target | Medium-High (writes target rows, no schema-level clone) |
| `probe_postgres_inventory.py` | Quick inventory probe (size/tables/counts) | URL from selected env var | No writes | Low |
| `verify_db_backup_inventory.py` | Compare local vs Jetstream table counts + backup manifests/blob listing | `JETSTREAM_DATABASE_URL`, `LOCAL_DATABASE_URL` | No writes | Low |
| `verify_call_artifact_blob_lanes.py` | Read-only: classify `call_artifacts.uri` by Azure blob container (and optional key prefix) | `DATABASE_URL` or `--env-var` / `--database-url` | No writes | Low |
| `check_call_artifacts_uri_constraint.py` | Inspect `call_artifacts_uri_must_be_blob` state and optional reject probe | `CANONICAL_DATABASE_URL` or `DATABASE_URL` | Probe insert rolls back; no persistent writes | Low |
| `remediate_call_artifacts_to_blob.py` | Re-upload local-path `call_artifacts.uri` payloads to Azure blob and relink rows | `CANONICAL_DATABASE_URL` + Azure blob auth vars | Updates canonical `call_artifacts` rows; optional `VALIDATE CONSTRAINT` | High |
| `check_raw_calls_uri_sentinel.py` | Report non-blob `raw_calls.response_json['uri']` anomalies and sentinel index status | `CANONICAL_DATABASE_URL` or `DATABASE_URL` | No writes | Low |
| `upload_jetstream_pg_dump_to_blob.py` | Upload `jetstream_for_local_*.dump` to Azure `db-backups` + write manifest | `AZURE_STORAGE_CONNECTION_STRING`; optional `JETSTREAM_DATABASE_URL` for manifest `table_counts` | Writes blobs + `backup_pg_dumps/*.manifest.json` | Medium (blob writes; sensitive dump contents) |
| `backup_jetstream_full_state.py` | One-command full-state backup (`pg_dump` orchestration + manifest verify + artifact container mirror + local receipt) | Jetstream DB/tunnel env and Azure blob auth (`AZURE_STORAGE_CONNECTION_STRING`; optional destination env override) | Writes `db-backups` blobs/manifests + destination artifact backup prefix + local receipt JSON | Medium-High (copies potentially large artifact corpus) |
| `start_jetstream_postgres_tunnel.py` | SSH local-forward to Jetstream Postgres | Requires Jetstream SSH host/auth env | No DB writes; network tunnel only | Low |
| `purge_dataset_acquisition.py` | Remove dataset acquisition artifacts for a dataset group | Selected DB URL + artifact storage backend | Deletes blob artifacts + matching DB rows | High (destructive by design) |
| `backup_mcq_db_to_json.py` | Export MCQ lineage rows (`groups`, `provenanced_runs`, `analysis_results`, `call_artifacts`) to JSON + `.manifest.json` | `LOCAL_DATABASE_URL` or `DATABASE_URL` | No writes | Low (can contain sensitive prompts/artifacts) |
| `check_active_workers.py` | Guardrail check for non-terminal `orchestration_jobs` before destructive ops | `LOCAL_DATABASE_URL` or `DATABASE_URL` | No writes | Low |
| `archive_mcq_artifact_blobs.py` | Copy MCQ-linked artifact blobs to frozen `mcq-archive/<date>/` prefix and emit URI remap receipt | Export JSON from `backup_mcq_db_to_json.py`; local artifact root or Azure blob creds | Writes archived blobs + remap JSON | Medium |

## Full-Copy vs Incremental Copy

- **Full-copy replace/clone:** `dump_postgres_for_jetstream_migration.py` + restore runbooks/scripts (`pg_dump`/`pg_restore`); optional off-VM archival: `upload_jetstream_pg_dump_to_blob.py` -> Azure `db-backups` + manifest for `verify_db_backup_inventory.py`
- **Incremental sync:** `sync_from_online.py` for additive v2-row transfer into a local clone, not full replacement
- **Compatibility note:** `dump_postgres_for_jetstream_migration.py` keeps its filename for backward compatibility

## High-Signal Living Entrypoints

These entrypoints are currently referenced in runbooks/living docs/CI and should remain
stable unless accompanied by wrappers and doc updates:

- `docker_smoke.py`
- `run_bank77_pipeline.py`
- `check_persistence_contract.py`
- `check_db_lane_policy.py`
- `check_living_docs_drift.py` (CI hard check for `.cursor/rules/living-docs-only.mdc`)
- `warn_restricted_doc_edits.py` (pre-commit warning; called from `scripts/git-hooks/pre-commit`)
- `validate_and_backfill_run_snapshots.py`
- `run_300_bigrun_sweep.py`
- `run_langgraph_job_worker.py`
- `run_cached_job_supervisor.py`
- `run_local_300_2datasets_engine_supervisor.py`
- `run_local_300_2datasets_worker.py`
- `ingest_sweep_to_db.py`
- `audit_last_partial_sweep.py`
- `reconcile_last_partial_sweep.py`
- `check_sweep_requests.py`
- `check_orchestration_jobs.py`
- `check_run_groups.py`
- `check_azure_blob_storage.py`
- `check_active_workers.py`
- `check_call_artifacts_uri_constraint.py`
- `check_raw_calls_uri_sentinel.py`
- `remediate_call_artifacts_to_blob.py`
- `backup_mcq_db_to_json.py`
- `archive_mcq_artifact_blobs.py`
- `backup_jetstream_full_state.py`

## Living-Docs-Only Governance

The binding doc-to-code map and restricted-reading set live in
[`.cursor/rules/living-docs-only.mdc`](../.cursor/rules/living-docs-only.mdc).
Two enforcement entrypoints live in this directory:

- `check_living_docs_drift.py` -- CI hard check. Fails when a diff range edits
  restricted paths without `[restricted-doc-edit-ok]` in any commit message in
  the range. Wired into `.github/workflows/living-docs-drift.yml`. Shared
  restricted-set source: `scripts/internal/living_docs_governance.py`.
- `warn_restricted_doc_edits.py` -- optional pre-commit warning that lists
  staged restricted-path edits to stderr and exits 0 (heads-up, not a block).
  Install the hook with `git config core.hooksPath scripts/git-hooks` (see
  [`scripts/git-hooks/README.md`](git-hooks/README.md)).

When membership changes, update both
`scripts/internal/living_docs_governance.py` and the table in the rule
together.

## Usage

Run scripts from repo root:

```bash
python scripts/<entrypoint>.py
```

Or pass explicit environment values inline:

```bash
DATABASE_URL=postgresql://... python scripts/probe_postgres_inventory.py
```

## Maintenance Rules

- Prefer `python -m study_query_llm.cli ...` for canonical CLI workflows where available.
- Keep `scripts/README.md` synchronized with lane assignments and high-signal entrypoints.
- Run `python scripts/verify_script_path_references.py` after doc/script path changes.
- Use `encoding='utf-8'` for Python file operations.
- When moving a referenced entrypoint, keep a compatibility wrapper and update runbooks/docs/parity evidence in the same change.



