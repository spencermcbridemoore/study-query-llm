# Runbooks Index

Status: living  
Owner: ops-maintainers  
Last reviewed: 2026-04-14

Use this page as the **single procedural entrypoint** for operator workflows.

## Source-Of-Truth Policy

- **Jetstream Postgres is the writable source of truth.**
- **Local Docker Postgres is clone/sandbox only** and may be dropped/recreated.
- Full-copy migrations use `pg_dump` / `pg_restore`; incremental copy is a separate workflow.

## Database URL Contract

| Variable | Canonical meaning | Typical use |
|---|---|---|
| `DATABASE_URL` | Active runtime write target for app/scripts in the current shell | Should point to Jetstream target for production-intent writes |
| `JETSTREAM_DATABASE_URL` | Jetstream DB endpoint (often tunnel URL on laptop) | Probing, verification, Jetstream dump source |
| `LOCAL_DATABASE_URL` | Local Docker Postgres endpoint | Clone target/sandbox and local restores |
| `SOURCE_DATABASE_URL` | Optional explicit dump source | One-off dump workflows (avoids overloading `DATABASE_URL`) |

## Canonical DB Workflows

### 1) Operate Jetstream On VM

- Runbook: [`deploy/jetstream/RUNBOOK.md`](../../deploy/jetstream/RUNBOOK.md)
- Bootstrap/boundaries: [`deploy/jetstream/README.md`](../../deploy/jetstream/README.md)

### 2) Full-Copy Migration/Replace

- Neon/any source to Jetstream compose DB: [`deploy/jetstream/MIGRATION_FROM_NEON.md`](../../deploy/jetstream/MIGRATION_FROM_NEON.md)
- Jetstream clone into local Docker: [`docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`](../LOCAL_DB_CLONE_FROM_JETSTREAM.md)

### 3) Local Tunnel + Cross-Env Verification

- Tunnel setup: [`deploy/jetstream/LOCAL_DEV_TUNNEL.md`](../../deploy/jetstream/LOCAL_DEV_TUNNEL.md)
- DB inventory probe: `python scripts/probe_postgres_inventory.py`
- Local-vs-Jetstream inventory + backup manifests: `python scripts/verify_db_backup_inventory.py`
- Upload a Jetstream-sourced `pg_dump -Fc` to Azure `db-backups` (same account as artifacts): after `python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream`, run `python scripts/upload_jetstream_pg_dump_to_blob.py` (or `--dump-path …`); then re-run `verify_db_backup_inventory.py`

## Application / Infrastructure Runbooks

- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md) (repo-root Docker stack; not Jetstream VM ops)
- [`docs/COLAB_SETUP.md`](../COLAB_SETUP.md)

## Sweep / Orchestration

- [`docs/SWEEP_MIGRATION_RUNBOOK.md`](../SWEEP_MIGRATION_RUNBOOK.md)
- [`docs/LANGGRAPH_JOB_EXECUTION.md`](../LANGGRAPH_JOB_EXECUTION.md)
- [`docs/TESTING_CHECKLIST.md`](../TESTING_CHECKLIST.md)
- Script lane note: active operational entrypoints remain under `scripts/`; historical experiment drivers are being moved under `scripts/history/` with compatibility wrappers.
- Dataset snapshot flow:
  - contract: [`docs/DATASET_SNAPSHOT_PROVENANCE.md`](../DATASET_SNAPSHOT_PROVENANCE.md)
  - create snapshots: `scripts/create_dataset_snapshots_286.py`
  - validate/backfill run linkage: `scripts/validate_and_backfill_run_snapshots.py`
  - BANK77 bootstrap:
    - full bootstrap: `python scripts/create_bank77_snapshot_and_embeddings.py --provider azure --embedding-engine text-embedding-3-large --require-azure-blob`
    - verify only: `python scripts/create_bank77_snapshot_and_embeddings.py --provider azure --embedding-engine text-embedding-3-large --verify-only --require-azure-blob`
- Layer-0 acquisition:
  - contract: [`docs/DATASET_ACQUISITION_LAYER0.md`](../DATASET_ACQUISITION_LAYER0.md)
  - command runbook: [`docs/runbooks/record_dataset_download.md`](record_dataset_download.md)

## Policy

- [`docs/BLOB_OPS_HARDENING_POLICY.md`](../BLOB_OPS_HARDENING_POLICY.md)
