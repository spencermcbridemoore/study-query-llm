# Documentation Parity Ledger

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-06

## Purpose

This ledger tracks concrete documentation claims against repository evidence.
It is the evidence base for documentation updates and taxonomy decisions.

Status values:

- `verified`: matches current code/tests behavior.
- `partial`: mostly true but incomplete/outdated scope.
- `stale`: formerly true but no longer current.
- `incorrect`: factually wrong for current code.
- `superseded`: replaced by a newer canonical path.
- `ambiguous`: cannot be statically verified from repository content alone.

## Claim Ledger

| claim_id | source_doc | claim_text | evidence_paths | status | impact | action |
|---|---|---|---|---|---|---|
| C001 | `README.md` | "Production Ready - All core features implemented and tested." | `README.md`, `docs/IMPLEMENTATION_PLAN.md` | partial | High: overstates confidence while plan still marks partial items. | Replace with evidence-based wording and point to current-state doc + backlog doc. |
| C002 | `README.md` | Completed phases include OpenAI + Hyperbolic provider implementation and full provider factory support. | `README.md`, `src/study_query_llm/providers/factory.py` | stale | High: onboarding confusion about available provider APIs. | Remove phased completion claims from README; direct to current-state and roadmap docs. |
| C003 | `README.md` | "Full test coverage (112 tests)." | `README.md`, `tests/**/*.py` (79 files), `test_e2e_verification.py` | incorrect | Medium: stale quality signal erodes trust. | Remove fixed numeric coverage claims; reference CI/test commands instead. |
| C004 | `docs/ARCHITECTURE.md` | Data access layer centers `InferenceRepository` for current analytics flows. | `docs/ARCHITECTURE.md`, `panel_app/views/analytics.py`, `src/study_query_llm/services/study_service.py` | stale | High: directs contributors to legacy path instead of v2 path. | Add/replace with v2 architecture (`RawCallRepository` + v2 tables). |
| C005 | `docs/ARCHITECTURE.md` | v2 database section lists `InferenceRun` fields under `models_v2.py`. | `docs/ARCHITECTURE.md`, `src/study_query_llm/db/models_v2.py` | incorrect | High: schema confusion and incorrect implementation assumptions. | Move `InferenceRun` to legacy context; document v2 models as canonical. |
| C006 | `docs/ARCHITECTURE.md` | Langfuse is integrated as observability layer in current app architecture. | `docs/ARCHITECTURE.md`, search `langfuse` (no source matches under `src/`) | stale | Medium: expectation mismatch for tracing/observability behavior. | Reword as planned/optional unless code integration is added. |
| C007 | `docs/API.md` | `ProviderFactory.get_available_providers()` returns `['azure','openai','hyperbolic']`. | `docs/API.md`, `src/study_query_llm/providers/factory.py` | incorrect | High: direct API misuse by developers. | Replace with current factory surface and provider lists per method. |
| C008 | `docs/API.md` | `StudyService` uses `InferenceRepository` in normal usage examples. | `docs/API.md`, `src/study_query_llm/services/study_service.py`, `panel_app/views/analytics.py` | incorrect | High: examples fail against current service contract. | Replace with `RawCallRepository` example path. |
| C009 | `docs/API.md` | `ProviderResponse.metadata` defaults to `None`. | `docs/API.md`, `src/study_query_llm/providers/base.py` | incorrect | Low: minor but concrete API mismatch. | Update example to match dataclass defaults/factory usage. |
| C010 | `docs/IMPLEMENTATION_PLAN.md` | Phase 1.3/1.4 openai/hyperbolic provider implementation is not implemented. | `docs/IMPLEMENTATION_PLAN.md`, `src/study_query_llm/providers/openai_compatible_chat_provider.py`, `src/study_query_llm/providers/factory.py`, `src/study_query_llm/config.py` | stale | Medium: roadmap narrative diverges from real provider strategy. | Mark early phase narrative as historical and add current provider architecture summary. |
| C011 | `docs/IMPLEMENTATION_PLAN.md` | Phase 4.1 says StudyService wraps `InferenceRepository`. | `docs/IMPLEMENTATION_PLAN.md`, `src/study_query_llm/services/study_service.py` | stale | Medium: analytics code references can be implemented incorrectly. | Update to `RawCallRepository` for current state. |
| C012 | `docs/DEPLOYMENT.md` + `Dockerfile` | Build-time tests run `pytest tests/test_e2e_verification.py`. | `docs/DEPLOYMENT.md`, `Dockerfile`, `test_e2e_verification.py` | incorrect | Medium: test command path mismatch can hide failures. | Correct command paths and align deployment docs with repository layout. |
| C013 | `docs/TESTING_CHECKLIST.md` | E2E verification script reflects current database model path. | `docs/TESTING_CHECKLIST.md`, `test_e2e_verification.py`, `src/study_query_llm/db/raw_call_repository.py` | stale | Medium: checklist validates v1 while app runs v2 analytics path. | Add v2 verification checklist and mark v1 script usage as legacy. |
| C014 | `docs/STANDING_ORDERS.md` | `IMPLEMENTATION_PLAN.md` and `ARCHITECTURE.md` are sole authoritative truth for status/design. | `docs/STANDING_ORDERS.md`, claims C004-C011 | partial | High: process guidance points at drifting sources. | Introduce explicit living current-state docs and relabel legacy/historical docs. |
| C015 | `docs/USER_GUIDE.md` | Contains both v1 and v2 setup in one path, presented as equivalent current flow. | `docs/USER_GUIDE.md` | partial | Medium: mixed guidance slows onboarding and causes wrong defaults. | Split or clearly route users to v2-first path; isolate v1 notes into history/deprecated. |
| C016 | `docs/ARCHITECTURE.md` | Standalone sweep worker + package CLI routes are canonical operational path. | `docs/ARCHITECTURE.md`, `src/study_query_llm/cli/__main__.py`, `src/study_query_llm/services/jobs/runtime_workers.py`, `src/study_query_llm/experiments/runtime_sweeps.py` | verified | Medium positive: valid anchor for operations docs. | Keep as canonical in living architecture summary. |
| C017 | `docs/SWEEP_MIGRATION_RUNBOOK.md` | CLI-first recommendation with script wrappers as compatibility layer. | `docs/SWEEP_MIGRATION_RUNBOOK.md`, `scripts/run_300_bigrun_sweep.py`, `scripts/run_langgraph_job_worker.py` | verified | Medium positive: correct operator guidance. | Keep as runbook; tighten cross-links from docs index. |
| C018 | `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md` | Jetstream clone runbook uses explicit backup + restore + verify phases. | `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`, referenced scripts under `scripts/` | verified | Low positive: clear and operationally useful runbook. | Keep as runbook lane with minimal edits. |
| C019 | `docs/BLOB_OPS_HARDENING_POLICY.md` | Blob policy defines lane/auth/strict-mode governance contract. | `docs/BLOB_OPS_HARDENING_POLICY.md`, `src/study_query_llm/storage/azure_blob.py`, `src/study_query_llm/services/artifact_service.py` | partial | Medium: policy is strong but enforcement coverage should be explicit. | Keep as living policy; add explicit "policy vs implementation" notes where necessary. |
| C020 | `docs/MIGRATION_GUIDE.md` | Migration from ad-hoc scripts to pytest is current primary guidance. | `docs/MIGRATION_GUIDE.md`, `CONTRIBUTING.md` | superseded | Low: duplicate guidance and old migration framing. | Mark deprecated and route to modern testing docs. |

## Severity-Ranked Mismatch Summary

### High

- C001, C002, C004, C005, C007, C008, C014

### Medium

- C003, C006, C010, C011, C012, C013, C015, C019

### Low

- C009, C020

## Aligned and Sampled Areas

These areas are sampled and considered sufficiently aligned for this review cycle:

- Sweep worker and package CLI operational architecture (`docs/ARCHITECTURE.md` + `src/study_query_llm/cli` + `runtime_*` modules).
- Sweep migration runbook CLI-first direction and wrapper model (`docs/SWEEP_MIGRATION_RUNBOOK.md` + `scripts/run_*.py` wrappers).
- Local DB clone runbook structure and script references (`docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md` + dump/restore scripts).

## Notes

- This ledger is intentionally claim-centric and does not replace architecture docs.
- Hypotheses that need runtime/load validation are tracked in `docs/DESIGN_FLAWS.md`.
