# Scripts Directory

This directory contains standalone script entrypoints and script helper modules.

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

## Current Lane Assignments (First Cleanup Pass)

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

Compatibility wrappers remain at their original root paths under `scripts/` and print
a deprecation notice when executed.

Historical compatibility stubs also exist for legacy names that still appear in
historical docs (`scripts/pca_kllmeans_sweep.py`, `scripts/migrate_v1_to_v2.py`).

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
| `upload_jetstream_pg_dump_to_blob.py` | Upload `jetstream_for_local_*.dump` to Azure `db-backups` + write manifest | `AZURE_STORAGE_CONNECTION_STRING`; optional `JETSTREAM_DATABASE_URL` for manifest `table_counts` | Writes blobs + `backup_pg_dumps/*.manifest.json` | Medium (blob writes; sensitive dump contents) |
| `start_jetstream_postgres_tunnel.py` | SSH local-forward to Jetstream Postgres | Requires Jetstream SSH host/auth env | No DB writes; network tunnel only | Low |
| `purge_dataset_acquisition.py` | Remove Layer-0 acquisition artifacts for a dataset group | Selected DB URL + artifact storage backend | Deletes blob artifacts + matching DB rows | High (destructive by design) |
| `record_dataset_download.py --persist-db` | Persist acquisition manifest/files as DB + blob artifacts | Dataset slug + active `DATABASE_URL` + Azure config | Creates/updates `dataset` group, artifacts, placeholders | Medium-High (writes canonical dataset artifacts) |
| `backup_mcq_db_to_json.py` | Export MCQ-related rows to JSON backup | `LOCAL_DATABASE_URL` or `DATABASE_URL` | No writes | Low (can contain sensitive prompts/artifacts) |

## Full-Copy vs Incremental Copy

- **Full-copy replace/clone:** `dump_postgres_for_jetstream_migration.py` + restore runbooks/scripts (`pg_dump`/`pg_restore`); optional off-VM archival: `upload_jetstream_pg_dump_to_blob.py` → Azure `db-backups` + manifest for `verify_db_backup_inventory.py`
- **Incremental sync:** `sync_from_online.py` for additive v2-row transfer into a local clone, not full replacement
- **Compatibility note:** `dump_postgres_for_jetstream_migration.py` keeps its filename for backward compatibility

## High-Signal Living Entrypoints

These entrypoints are currently referenced in runbooks/living docs/CI and should remain
stable unless accompanied by wrappers and doc updates:

- `docker_smoke.py`
- `create_bank77_snapshot_and_embeddings.py`
- `create_dataset_snapshots_286.py`
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

## Usage

Run scripts from repo root:

```bash
python scripts/<entrypoint>.py
```

Or pass explicit environment values inline:

```bash
DATABASE_URL=postgresql://... python scripts/check_all_data.py
```

## Maintenance Rules

- Prefer `python -m study_query_llm.cli ...` for canonical CLI workflows where available.
- Keep `scripts/README.md` synchronized with lane assignments and high-signal entrypoints.
- Run `python scripts/verify_script_path_references.py` after doc/script path changes.
- Use `encoding='utf-8'` for Python file operations.
- When moving a referenced entrypoint, keep a compatibility wrapper and update runbooks/docs/parity evidence in the same change.
