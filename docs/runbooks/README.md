# Runbooks Index

Status: living  
Owner: ops-maintainers  
Last reviewed: 2026-04-07

Use runbooks for procedures that are executed step-by-step.

## Application and Infrastructure

- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)
- [`docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`](../LOCAL_DB_CLONE_FROM_JETSTREAM.md)
- [`docs/COLAB_SETUP.md`](../COLAB_SETUP.md)

## Sweep / Orchestration

- [`docs/SWEEP_MIGRATION_RUNBOOK.md`](../SWEEP_MIGRATION_RUNBOOK.md)
- [`docs/LANGGRAPH_JOB_EXECUTION.md`](../LANGGRAPH_JOB_EXECUTION.md)
- [`docs/TESTING_CHECKLIST.md`](../TESTING_CHECKLIST.md)
- Dataset snapshot flow (lightweight):
  - contract: [`docs/DATASET_SNAPSHOT_PROVENANCE.md`](../DATASET_SNAPSHOT_PROVENANCE.md)
  - create snapshots: `scripts/create_dataset_snapshots_286.py`
  - validate/backfill run linkage: `scripts/validate_and_backfill_run_snapshots.py`
  - BANK77 bootstrap (snapshot + full embeddings + means):
    - full bootstrap: `python scripts/create_bank77_snapshot_and_embeddings.py --provider azure --embedding-engine text-embedding-3-large --require-azure-blob`
    - verify only: `python scripts/create_bank77_snapshot_and_embeddings.py --provider azure --embedding-engine text-embedding-3-large --verify-only --require-azure-blob`
- Layer 0 download provenance (checksums + `acquisition.json`, optional Azure persist):
  - contract: [`docs/DATASET_ACQUISITION_LAYER0.md`](../DATASET_ACQUISITION_LAYER0.md)
  - runbook: [`docs/runbooks/record_dataset_download.md`](record_dataset_download.md)

## Policy Runbook

- [`docs/BLOB_OPS_HARDENING_POLICY.md`](../BLOB_OPS_HARDENING_POLICY.md)
