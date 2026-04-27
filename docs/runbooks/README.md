# Runbooks Index

Status: living  
Owner: ops-maintainers  
Last reviewed: 2026-04-26

Use this page as the **single procedural entrypoint** for operator workflows.

## Source-Of-Truth Policy

- **Jetstream Postgres is the writable source of truth.**
- **Local Docker Postgres is clone/sandbox only** and may be dropped/recreated.
- Full-copy migrations use `pg_dump` / `pg_restore`; incremental copy is a separate workflow.

## Database URL Contract

| Variable | Canonical meaning | Typical use |
|---|---|---|
| `CANONICAL_DATABASE_URL` | Canonical source-of-truth DB endpoint | Identity/role binding for canonical-lane workflows |
| `DATABASE_URL` | Active runtime DB target for app/scripts in the current shell | Canonical writes, local mirror reads, or sandbox runs depending on `SQLLM_WRITE_INTENT` |
| `JETSTREAM_DATABASE_URL` | Jetstream DB endpoint (often tunnel URL on laptop) | Probing, verification, Jetstream dump source |
| `LOCAL_DATABASE_URL` | Local Docker Postgres endpoint | Clone target/sandbox and local restores |
| `SQLLM_WRITE_INTENT` | Declared DB lane intent (`canonical`, `read_mirror`, `sandbox`) | Optional shell default when constructors do not pass `write_intent` |
| `SOURCE_DATABASE_URL` | Optional explicit dump source | One-off dump workflows (avoids overloading `DATABASE_URL`) |

### Destructive DDL Override Note

- `SQLLM_ALLOW_DESTRUCTIVE_DDL=1` is intended for known throwaway targets only (for example local scratch/test clones).
- This override does not permit destructive operations against the target identified by `JETSTREAM_DATABASE_URL`.
- Clear the override after one-off maintenance (`Remove-Item Env:SQLLM_ALLOW_DESTRUCTIVE_DDL` in PowerShell).

## Canonical DB Workflows

### 1) Operate Jetstream On VM

- Runbook: [`deploy/jetstream/RUNBOOK.md`](../../deploy/jetstream/RUNBOOK.md)
- Bootstrap/boundaries: [`deploy/jetstream/README.md`](../../deploy/jetstream/README.md)

### 2) Full-Copy Migration/Replace

- Neon/any source to Jetstream compose DB: [`deploy/jetstream/MIGRATION_FROM_NEON.md`](../../deploy/jetstream/MIGRATION_FROM_NEON.md)
- Jetstream clone into local Docker: [`docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`](../LOCAL_DB_CLONE_FROM_JETSTREAM.md)

### 3) Local Tunnel + Cross-Env Verification

- Tunnel setup: [`deploy/jetstream/LOCAL_DEV_TUNNEL.md`](../../deploy/jetstream/LOCAL_DEV_TUNNEL.md)
- Jetstream lifecycle context (Apr 22 v5 cutover → dormant; restore options): [`docs/runbooks/JETSTREAM_STATE_TIMELINE.md`](JETSTREAM_STATE_TIMELINE.md)
- DB inventory probe: `python scripts/probe_postgres_inventory.py`
- Local-vs-Jetstream inventory + backup manifests: `python scripts/verify_db_backup_inventory.py`
- Azure `call_artifacts` lane sanity (read-only): `python scripts/verify_call_artifact_blob_lanes.py` (defaults: `DATABASE_URL`, expected container `artifacts-dev`; add `--expected-prefix dev` to require `dev/` blob keys)
- Canonical `call_artifacts.uri` constraint check/probe: `python scripts/check_call_artifacts_uri_constraint.py --env-var CANONICAL_DATABASE_URL --probe-insert`
- Canonical remediation + constraint validation: `python scripts/remediate_call_artifacts_to_blob.py --apply --validate-constraint`
- Raw-call URI sentinel monitor: `python scripts/check_raw_calls_uri_sentinel.py --env-var CANONICAL_DATABASE_URL --require-zero`
- Upload a Jetstream-sourced `pg_dump -Fc` to Azure `db-backups` (same account as artifacts): after `python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream`, run `python scripts/upload_jetstream_pg_dump_to_blob.py` (or `--dump-path …`); then re-run `verify_db_backup_inventory.py`
- One-command full-state backup (DB dump/upload/verify + artifact container mirror + receipt): `python scripts/backup_jetstream_full_state.py` (defaults source container from `ARTIFACT_RUNTIME_ENV`/`AZURE_STORAGE_CONTAINER`; destination defaults to `<source>-backups`; optional cross-account destination via `AZURE_BACKUP_STORAGE_CONNECTION_STRING`)

## Application / Infrastructure Runbooks

- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md) (repo-root Docker stack; not Jetstream VM ops)
- [`docs/COLAB_SETUP.md`](../COLAB_SETUP.md)

## Sweep / Orchestration

- [`docs/SWEEP_MIGRATION_RUNBOOK.md`](../SWEEP_MIGRATION_RUNBOOK.md)
- [`docs/LANGGRAPH_JOB_EXECUTION.md`](../LANGGRAPH_JOB_EXECUTION.md)
- [`docs/TESTING_CHECKLIST.md`](../TESTING_CHECKLIST.md)
- Script lane note: active operational entrypoints remain under `scripts/`; historical experiment drivers are being moved under `scripts/history/` with compatibility wrappers.
- Data pipeline flow:
  - contract: [`docs/DATA_PIPELINE.md`](../DATA_PIPELINE.md)
  - run BANK77 end-to-end (acquire -> parse -> snapshot -> embed -> analyze):
    - `python scripts/run_bank77_pipeline.py --embedding-provider azure --embedding-deployment text-embedding-3-small --embedding-representation full --embedding-chunk-size 128 --analysis-method cosine_kllmeans_no_pca --analysis-run-key bank77_full_run_seed7`
  - idempotency check: rerun the exact same command and confirm stage metadata reports `reused: true`
  - validate/backfill legacy snapshot linkage: `scripts/validate_and_backfill_run_snapshots.py`

## Policy

- [`docs/BLOB_OPS_HARDENING_POLICY.md`](../BLOB_OPS_HARDENING_POLICY.md)
