# Documentation Index

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-13

## Quick Route

- Current product/runtime truth: [`docs/living/CURRENT_STATE.md`](living/CURRENT_STATE.md)
- Current architecture (v2-first): [`docs/living/ARCHITECTURE_CURRENT.md`](living/ARCHITECTURE_CURRENT.md)
- Current API entrypoints: [`docs/living/API_CURRENT.md`](living/API_CURRENT.md)
- Canonical DB ops entrypoint: [`docs/runbooks/README.md`](runbooks/README.md)
- Current user workflow (v2-first): [`docs/USER_GUIDE.md`](USER_GUIDE.md)
- Design flaws register: [`docs/DESIGN_FLAWS.md`](DESIGN_FLAWS.md)
- Parity evidence ledger: [`docs/review/DOC_PARITY_LEDGER.md`](review/DOC_PARITY_LEDGER.md)

## Taxonomy

### Living

- [`docs/living/CURRENT_STATE.md`](living/CURRENT_STATE.md)
- [`docs/living/ARCHITECTURE_CURRENT.md`](living/ARCHITECTURE_CURRENT.md)
- [`docs/living/API_CURRENT.md`](living/API_CURRENT.md)
- [`docs/living/PLOT_CONVENTIONS.md`](living/PLOT_CONVENTIONS.md)
- [`docs/USER_GUIDE.md`](USER_GUIDE.md)
- [`docs/STANDING_ORDERS.md`](STANDING_ORDERS.md)
- [`docs/DATASET_SNAPSHOT_PROVENANCE.md`](DATASET_SNAPSHOT_PROVENANCE.md)
- [`docs/DESIGN_FLAWS.md`](DESIGN_FLAWS.md)

### Runbooks

- [`docs/runbooks/README.md`](runbooks/README.md) (**start here** for DB/tunnel/backup/restore ops)
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md)
- [`docs/SWEEP_MIGRATION_RUNBOOK.md`](SWEEP_MIGRATION_RUNBOOK.md)
- [`docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`](LOCAL_DB_CLONE_FROM_JETSTREAM.md)
- [`docs/TESTING_CHECKLIST.md`](TESTING_CHECKLIST.md)
- [`docs/LANGGRAPH_JOB_EXECUTION.md`](LANGGRAPH_JOB_EXECUTION.md)
- [`docs/BLOB_OPS_HARDENING_POLICY.md`](BLOB_OPS_HARDENING_POLICY.md)
- [`docs/COLAB_SETUP.md`](COLAB_SETUP.md)

### History

- [`docs/history/README.md`](history/README.md)
- [`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- [`docs/PHASE1_5_VERIFICATION.md`](PHASE1_5_VERIFICATION.md)
- [`docs/PLOT_ORGANIZATION.md`](PLOT_ORGANIZATION.md)
- [`docs/history/USER_GUIDE_V1_LEGACY.md`](history/USER_GUIDE_V1_LEGACY.md)
- [`docs/experiments/CUSTOM_SWEEP_README.md`](experiments/CUSTOM_SWEEP_README.md)
- [`docs/plans/README.md`](plans/README.md) (archived planning framework)

### Deprecated

- [`docs/deprecated/README.md`](deprecated/README.md)
- [`docs/API.md`](API.md) (replaced by `docs/living/API_CURRENT.md`)
- [`docs/MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md)

### Review Artifacts

- [`docs/review/DOC_PARITY_LEDGER.md`](review/DOC_PARITY_LEDGER.md)

## Navigation Policy

- Treat only `living` docs as current implementation truth.
- Use `runbooks` for procedures and operator workflows.
- For DB operations, treat `docs/runbooks/README.md` as the top-level workflow index and URL contract source.
- Keep `history` for chronology and migration context.
- Keep `deprecated` for compatibility context; do not use for new implementation decisions.
