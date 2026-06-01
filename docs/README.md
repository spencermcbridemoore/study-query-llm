# Documentation Index

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-18

## Quick Route

- Current product/runtime truth: [`docs/living/CURRENT_STATE.md`](living/CURRENT_STATE.md)
- Current architecture (v2-first): [`docs/living/ARCHITECTURE_CURRENT.md`](living/ARCHITECTURE_CURRENT.md)
- Canonical data pipeline contract: [`docs/DATA_PIPELINE.md`](DATA_PIPELINE.md)
- Current API entrypoints: [`docs/living/API_CURRENT.md`](living/API_CURRENT.md)
- Scheduling/provenance boundary + execution terminology: [`docs/living/SCHEDULING_PROVENANCE_BOUNDARY.md`](living/SCHEDULING_PROVENANCE_BOUNDARY.md)
- Method recipes (composite pipeline spec): [`docs/living/METHOD_RECIPES.md`](living/METHOD_RECIPES.md)
- Canonical DB ops entrypoint: [`docs/runbooks/README.md`](runbooks/README.md)
- Current user workflow (v2-first): [`docs/USER_GUIDE.md`](USER_GUIDE.md)
- Design flaws register: [`docs/DESIGN_FLAWS.md`](DESIGN_FLAWS.md)
- Parity evidence ledger: [`docs/review/DOC_PARITY_LEDGER.md`](review/DOC_PARITY_LEDGER.md)

## Taxonomy

### Living

- [`docs/living/CURRENT_STATE.md`](living/CURRENT_STATE.md)
- [`docs/living/ARCHITECTURE_CURRENT.md`](living/ARCHITECTURE_CURRENT.md)
- [`docs/living/API_CURRENT.md`](living/API_CURRENT.md)
- [`docs/living/SCHEDULING_PROVENANCE_BOUNDARY.md`](living/SCHEDULING_PROVENANCE_BOUNDARY.md)
- [`docs/living/METHOD_RECIPES.md`](living/METHOD_RECIPES.md)
- [`docs/living/PLOT_CONVENTIONS.md`](living/PLOT_CONVENTIONS.md)
- [`docs/USER_GUIDE.md`](USER_GUIDE.md)
- [`docs/STANDING_ORDERS.md`](STANDING_ORDERS.md)
- [`docs/DATA_PIPELINE.md`](DATA_PIPELINE.md)
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

- [`docs/history/README.md`](https://github.com/spencermcbridemoore/study-query-llm/tree/archive/pre-context-hygiene-cleanup/docs/history/README.md)
- [`docs/IMPLEMENTATION_PLAN.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/IMPLEMENTATION_PLAN.md)
- [`docs/ARCHITECTURE.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/ARCHITECTURE.md)
- [`docs/PHASE1_5_VERIFICATION.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/PHASE1_5_VERIFICATION.md)
- [`docs/PLOT_ORGANIZATION.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/PLOT_ORGANIZATION.md)
- [`docs/history/USER_GUIDE_V1_LEGACY.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/history/USER_GUIDE_V1_LEGACY.md)
- [`docs/experiments/CUSTOM_SWEEP_README.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/experiments/CUSTOM_SWEEP_README.md)
- [`docs/plans/README.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/plans/README.md) (archived planning framework)
- [`docs/audit/`](https://github.com/spencermcbridemoore/study-query-llm/tree/archive/pre-context-hygiene-cleanup/docs/audit/)
- [`docs/design/`](https://github.com/spencermcbridemoore/study-query-llm/tree/archive/pre-context-hygiene-cleanup/docs/design/)

### Deprecated

- [`docs/deprecated/README.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/deprecated/README.md)
- [`docs/API.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/API.md) (replaced by `docs/living/API_CURRENT.md`)
- [`docs/MIGRATION_GUIDE.md`](https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/MIGRATION_GUIDE.md)

### Review Artifacts

- [`docs/review/DOC_PARITY_LEDGER.md`](review/DOC_PARITY_LEDGER.md)

## Navigation Policy

- Treat only `living` docs as current implementation truth.
- Use `runbooks` for procedures and operator workflows.
- For DB operations, treat `docs/runbooks/README.md` as the top-level workflow index and URL contract source.
- Keep `history` for chronology and migration context.
- Keep `deprecated` for compatibility context; do not use for new implementation decisions.

## Archived material

Historical, deprecated, plan, experiment, audit, and design material is
archived at tag `archive/pre-context-hygiene-cleanup`. Use:

- `https://github.com/spencermcbridemoore/study-query-llm/tree/archive/pre-context-hygiene-cleanup/docs/`
- `https://github.com/spencermcbridemoore/study-query-llm/blob/archive/pre-context-hygiene-cleanup/docs/<path>`
