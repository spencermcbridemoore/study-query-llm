# Documentation Parity Ledger

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-10

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
| C001 | `README.md` | "Production Ready - All core features implemented and tested." | `README.md`, `docs/IMPLEMENTATION_PLAN.md` | superseded | High: previously overstated readiness. | Completed: replaced with current-state references and historical roadmap links. |
| C002 | `README.md` | Completed phases include OpenAI + Hyperbolic provider implementation and full provider factory support. | `README.md`, `src/study_query_llm/providers/factory.py` | superseded | High: previously implied outdated provider model. | Completed: phased completion claims removed from README. |
| C003 | `README.md` | "Full test coverage (112 tests)." | `README.md`, `tests/**/*.py` | superseded | Medium: previously stale quality signal. | Completed: hardcoded test-count claims removed. |
| C004 | `docs/ARCHITECTURE.md` | Data access layer centers `InferenceRepository` for current analytics flows. | `docs/ARCHITECTURE.md`, `panel_app/views/analytics.py`, `src/study_query_llm/services/study_service.py` | superseded | High: legacy architecture narrative can mislead if treated as current. | Mitigated by routing: current architecture moved to `docs/living/ARCHITECTURE_CURRENT.md`; source doc marked historical. |
| C005 | `docs/ARCHITECTURE.md` | v2 database section lists `InferenceRun` fields under `models_v2.py`. | `docs/ARCHITECTURE.md`, `src/study_query_llm/db/models_v2.py` | superseded | High: schema confusion risk in legacy narrative. | Mitigated by routing and historical labeling; use living architecture doc for implementation decisions. |
| C006 | `docs/ARCHITECTURE.md` | Langfuse is integrated as observability layer in current app architecture. | `docs/ARCHITECTURE.md`, search `langfuse` (no source matches under `src/`) | superseded | Medium: outdated expectation in historical narrative. | Mitigated by living-doc routing; retain as historical context only. |
| C007 | `docs/API.md` | `ProviderFactory.get_available_providers()` returns `['azure','openai','hyperbolic']`. | `docs/API.md`, `src/study_query_llm/providers/factory.py` | superseded | High: stale API in deprecated doc. | Completed: `docs/API.md` marked deprecated; canonical API moved to `docs/living/API_CURRENT.md`. |
| C008 | `docs/API.md` | `StudyService` uses `InferenceRepository` in normal usage examples. | `docs/API.md`, `src/study_query_llm/services/study_service.py`, `panel_app/views/analytics.py` | superseded | High: stale API usage in deprecated doc. | Completed: canonical v2 usage documented in `docs/living/API_CURRENT.md`. |
| C009 | `docs/API.md` | `ProviderResponse.metadata` defaults to `None`. | `docs/API.md`, `src/study_query_llm/providers/base.py` | superseded | Low: minor stale API detail in deprecated doc. | Accepted as deprecated-doc drift; canonical reference is `docs/living/API_CURRENT.md`. |
| C010 | `docs/IMPLEMENTATION_PLAN.md` | Phase 1.3/1.4 openai/hyperbolic provider implementation is not implemented. | `docs/IMPLEMENTATION_PLAN.md`, `src/study_query_llm/providers/openai_compatible_chat_provider.py`, `src/study_query_llm/providers/factory.py`, `src/study_query_llm/config.py` | superseded | Medium: historical phase chronology diverges from current implementation. | Completed: plan explicitly labeled historical-roadmap; current status moved to `docs/living/CURRENT_STATE.md`. |
| C011 | `docs/IMPLEMENTATION_PLAN.md` | Phase 4.1 says StudyService wraps `InferenceRepository`. | `docs/IMPLEMENTATION_PLAN.md`, `src/study_query_llm/services/study_service.py` | superseded | Medium: legacy phase narrative not aligned with v2 defaults. | Mitigated by historical labeling + living current-state docs. |
| C012 | `docs/DEPLOYMENT.md` + `Dockerfile` | Build-time tests run `pytest tests/test_e2e_verification.py`. | `docs/DEPLOYMENT.md`, `Dockerfile`, `test_e2e_verification.py` | superseded | Medium: previously mismatched command path. | Completed: command paths aligned to repository layout (`pytest test_e2e_verification.py`, fallback `pytest test_phase_1_1.py`). |
| C013 | `docs/TESTING_CHECKLIST.md` | E2E verification script reflects current database model path. | `docs/TESTING_CHECKLIST.md`, `test_e2e_verification.py`, `src/study_query_llm/db/raw_call_repository.py` | superseded | Medium: previously implied v1 script as current. | Completed: checklist now explicitly flags `test_e2e_verification.py` as legacy-v1 validation. |
| C014 | `docs/STANDING_ORDERS.md` | `IMPLEMENTATION_PLAN.md` and `ARCHITECTURE.md` are sole authoritative truth for status/design. | `docs/STANDING_ORDERS.md` | superseded | High: previously pointed process at drifting docs. | Completed: standing orders now point to living docs + parity ledger. |
| C015 | `docs/USER_GUIDE.md` | Contains both v1 and v2 setup in one path, presented as equivalent current flow. | `docs/USER_GUIDE.md`, `docs/history/USER_GUIDE_V1_LEGACY.md` | superseded | Medium: previously mixed onboarding path. | Completed: v2-first guide retained; legacy v1 content split to historical doc and v2 analytics examples updated. |
| C016 | `docs/ARCHITECTURE.md` | Standalone sweep worker + package CLI routes are canonical operational path. | `docs/ARCHITECTURE.md`, `src/study_query_llm/cli/__main__.py`, `src/study_query_llm/services/jobs/runtime_workers.py`, `src/study_query_llm/experiments/runtime_sweeps.py` | verified | Medium positive: valid anchor for operations docs. | Keep as canonical in living architecture summary. |
| C017 | `docs/SWEEP_MIGRATION_RUNBOOK.md` | CLI-first recommendation with script wrappers as compatibility layer. | `docs/SWEEP_MIGRATION_RUNBOOK.md`, `scripts/run_300_bigrun_sweep.py`, `scripts/run_langgraph_job_worker.py` | verified | Medium positive: correct operator guidance. | Keep as runbook; tighten cross-links from docs index. |
| C018 | `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md` | Jetstream clone runbook uses explicit backup + restore + verify phases. | `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`, referenced scripts under `scripts/` | verified | Low positive: clear and operationally useful runbook. | Keep as runbook lane with minimal edits. |
| C019 | `docs/BLOB_OPS_HARDENING_POLICY.md` | Blob policy defines lane/auth/strict-mode governance contract. | `docs/BLOB_OPS_HARDENING_POLICY.md`, `src/study_query_llm/storage/azure_blob.py`, `src/study_query_llm/services/artifact_service.py` | verified | Medium positive: hard quota and backend governance now enforced in write path. | Keep as living policy; monitor operational exceptions. |
| C020 | `docs/MIGRATION_GUIDE.md` | Migration from ad-hoc scripts to pytest is current primary guidance. | `docs/MIGRATION_GUIDE.md`, `CONTRIBUTING.md` | superseded | Low: duplicate guidance and old migration framing. | Completed: marked deprecated and routed to current testing/contributing guidance. |
| C021 | `docs/living/CURRENT_STATE.md` + `docs/living/ARCHITECTURE_CURRENT.md` | OrchestrationJob-first clustering path and unified `provenanced_runs` execution model are current behavior. | `src/study_query_llm/services/sweep_request_service.py`, `src/study_query_llm/services/provenanced_run_service.py`, `src/study_query_llm/db/models_v2.py`, `src/study_query_llm/experiments/sweep_worker_main.py` | verified | High positive: aligns docs with control-plane and provenance contracts used by runtime code. | Keep as canonical architecture contract for ongoing migration/cutover work. |
| C022 | `docs/living/CURRENT_STATE.md` + `docs/living/ARCHITECTURE_CURRENT.md` | MCQ runs are OrchestrationJob-first (single `mcq_run` job per run_key), and new MCQ method executions persist explicit `provenanced_runs` rows while legacy MCQ rows remain queryable via compatibility mapping. | `src/study_query_llm/services/sweep_request_service.py`, `src/study_query_llm/services/jobs/job_payload_models.py`, `src/study_query_llm/services/jobs/job_runner_factory.py`, `src/study_query_llm/experiments/sweep_worker_main.py`, `src/study_query_llm/experiments/mcq_run_persistence.py`, `src/study_query_llm/services/provenanced_run_service.py` | verified | High positive: aligns MCQ execution/provenance contracts with the same control-plane/query model used by clustering. | Keep as canonical until dedicated MCQ backfill policy changes. |
| C023 | `docs/DATASET_SNAPSHOT_PROVENANCE.md` + `docs/living/CURRENT_STATE.md` | Clustering snapshot provenance uses normalized `dataset_snapshot_ids` + `depends_on` links, with primary pointer persisted in `provenanced_runs.input_snapshot_group_id` and compatibility fallback for legacy runs. | `src/study_query_llm/experiments/ingestion.py`, `src/study_query_llm/services/provenanced_run_service.py`, `src/study_query_llm/services/sweep_query_service.py`, `scripts/validate_and_backfill_run_snapshots.py` | verified | High positive: makes snapshot lineage queryable in both execution and sweep read paths without schema expansion. | Keep as lightweight canonical contract until/if dedicated input-materialization schema is introduced. |
| C024 | `docs/DATASET_SNAPSHOT_PROVENANCE.md` + `docs/runbooks/README.md` + `docs/living/CURRENT_STATE.md` | BANK77 bootstrap flow is implemented as deterministic snapshot + full embedding matrix + 77xD intent means artifacts, with mapping metadata and verify/readback path. | `scripts/create_bank77_snapshot_and_embeddings.py`, `tests/test_scripts/test_create_bank77_snapshot_and_embeddings.py`, `src/study_query_llm/services/artifact_service.py`, `src/study_query_llm/services/provenance_service.py` | verified | High positive: provides concrete, repeatable ingestion path for BANK77 with lineage and integrity checks using existing schema/services. | Keep as current runbook contract for BANK77 dataset setup and validate with `--verify-only` during operations. |
| C025 | `docs/living/CURRENT_STATE.md` + `docs/living/API_CURRENT.md` | OpenRouter is first-class in embedding provider surfaces, deployment discovery supports OpenRouter runtime catalogs, ModelRegistry persists deployment metadata, and embedding token validation can consume discovered context-length limits with fallback behavior. | `src/study_query_llm/providers/factory.py`, `src/study_query_llm/providers/base.py`, `src/study_query_llm/services/model_registry.py`, `src/study_query_llm/services/embeddings/service.py`, `tests/test_providers/test_factory.py`, `tests/test_services/test_model_registry.py`, `tests/test_services/test_embedding_service.py` | verified | High positive: closes chat-only/discovery gaps for OpenRouter embeddings and makes runtime characterization available through canonical surfaces. | Keep as canonical behavior; extend metadata/probing only if stronger provider guarantees are required. |
| C026 | `docs/living/CURRENT_STATE.md` + `docs/living/ARCHITECTURE_CURRENT.md` + `docs/living/API_CURRENT.md` | Run/analysis reunification is now execution-first: canonical `provenanced_runs.run_kind=execution`, MCQ `analysis_run` jobs share orchestration control plane, and request analysis state reads derive from execution/job evidence with parity diagnostics vs legacy metadata. | `src/study_query_llm/services/provenanced_run_service.py`, `src/study_query_llm/db/raw_call_repository.py`, `src/study_query_llm/db/models_v2.py`, `src/study_query_llm/services/sweep_request_service.py`, `src/study_query_llm/experiments/sweep_worker_main.py`, `src/study_query_llm/analysis/mcq_analyze_request.py`, `src/study_query_llm/services/jobs/job_payload_models.py` | verified | High positive: aligns living docs with the unified execution substrate and removes split control/read semantics from canonical behavior. | Keep as canonical contract; remove legacy mirrors after migration window + backfill validation. |
| C027 | `docs/living/CURRENT_STATE.md` | Panel includes a Storage / DB stats tab: Postgres catalog sizes and largest relations when `DATABASE_URL` is PostgreSQL; v2 counts including `CallArtifact`; artifact env summary (no secrets); optional capped Azure blob prefix listing; redacted `LOCAL_DATABASE_URL` and clone runbook pointers. | `panel_app/app.py`, `panel_app/views/storage_stats.py`, `src/study_query_llm/storage/azure_blob.py` | verified | Low positive: operator visibility without expanding write paths. | Keep tab behavior aligned with artifact env contract in `artifact_service.py` / blob policy docs. |
| C028 | `docs/DATASET_ACQUISITION_LAYER0.md` + `docs/living/CURRENT_STATE.md` | Layer 0 download provenance records pinned URLs, per-file SHA-256, and `acquisition.json`; optional `--persist-db` creates a `dataset` group and stores `dataset_acquisition_file` / `dataset_acquisition_manifest` blobs via `ArtifactService.store_group_blob_artifact` when Azure blob backend is configured. | `src/study_query_llm/datasets/acquisition.py`, `src/study_query_llm/datasets/source_specs/`, `scripts/record_dataset_download.py`, `src/study_query_llm/services/artifact_service.py`, `tests/test_datasets/test_acquisition.py`, `tests/test_services/test_artifact_service.py` | verified | Medium positive: reproducible upstream capture before normalized `dataset_snapshot` work. | Add new slugs via `ACQUIRE_REGISTRY`; keep pinned git refs intentional when upstream data changes. |

## Severity-Ranked Mismatch Summary

### Open High

- None

### Open Medium

- None

### Open Low

- None

## Aligned and Sampled Areas

These areas are sampled and considered sufficiently aligned for this review cycle:

- Sweep worker and package CLI operational architecture (`docs/ARCHITECTURE.md` + `src/study_query_llm/cli` + `runtime_*` modules).
- Sweep migration runbook CLI-first direction and wrapper model (`docs/SWEEP_MIGRATION_RUNBOOK.md` + `scripts/run_*.py` wrappers).
- Local DB clone runbook structure and script references (`docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md` + dump/restore scripts).

## Notes

- This ledger is intentionally claim-centric and does not replace architecture docs.
- Hypotheses that need runtime/load validation are tracked in `docs/DESIGN_FLAWS.md`.
