# Design Flaws Register

Status: living  
Owner: architecture-maintainers  
Last reviewed: 2026-05-02

## Purpose

This register captures design/process issues discovered during top-down documentation and code review.
It is intentionally evidence-driven and separate from roadmap chronology.

## Field Definitions

- `id`: stable flaw identifier.
- `type`: `architecture` | `implementation` | `documentation-process` | `operability`.
- `severity`: `high` | `medium` | `low`.
- `confirmation`: `confirmed` (repository evidence) or `hypothesis` (requires runtime validation).
- `status`: `open` | `accepted` | `mitigated`.
- `evidence`: concrete file paths.
- `remediation`: direction, not a full implementation plan.

## Flaws

| id | type | severity | confirmation | status | evidence | risk | remediation |
|---|---|---|---|---|---|---|---|
| DF-001 | documentation-process | high | confirmed | open | `README.md`, `docs/IMPLEMENTATION_PLAN.md`, `src/study_query_llm/providers/factory.py` | Conflicting status narratives mislead implementation decisions. | Treat `docs/living/CURRENT_STATE.md` as current SoT and keep phased chronology explicitly historical. |
| DF-002 | documentation-process | high | confirmed | open | `docs/ARCHITECTURE.md`, `panel_app/views/analytics.py`, `src/study_query_llm/services/study_service.py` | v1-centric architecture descriptions drive wrong repository usage. | Keep v2-first architecture in living docs; retain old architecture only as historical context. |
| DF-003 | documentation-process | medium | confirmed | open | `docs/ARCHITECTURE.md`, repository search for `langfuse` in `src/` | Docs imply active Langfuse integration where source integration is absent. | Reclassify Langfuse as planned/optional unless implemented. |
| DF-004 | implementation | medium | confirmed | open | `src/study_query_llm/providers/factory.py` | Split factory surfaces (`create` vs `create_chat_provider`) are easy to misuse. | Explicitly document canonical factory entrypoints and de-emphasize legacy `create` path. |
| DF-005 | architecture | medium | confirmed | open | `src/study_query_llm/db/__init__.py` | Mixed v1+v2 exports blur boundaries for new development. | Document/import v2 as default; keep v1 behind explicit legacy framing. |
| DF-006 | documentation-process | medium | confirmed | open | `docs/API.md`, `src/study_query_llm/providers/factory.py`, `src/study_query_llm/services/study_service.py` | Stale API examples cause copy/paste breakage. | Keep `docs/living/API_CURRENT.md` as canonical and mark legacy API doc deprecated. |
| DF-007 | operability | medium | hypothesis | open | `src/study_query_llm/db/raw_call_repository.py` (`claim_next_orchestration_job`) | Job-claim path iterates candidate lists in Python; may degrade at large queue sizes. | Validate with load tests; consider SQL-level lock/claim patterns if needed. |
| DF-008 | operability | medium | hypothesis | open | `src/study_query_llm/services/jobs/`, `src/study_query_llm/experiments/`, `docs/SWEEP_MIGRATION_RUNBOOK.md` | Multiple execution surfaces increase operator error risk under incident pressure. | Maintain one operator matrix in docs and keep wrapper-vs-canonical guidance explicit. |
| DF-009 | documentation-process | low | confirmed | open | `README.md`, `tests/` tree | Fixed test-count claims age quickly and become inaccurate. | Avoid hardcoded counts; reference CI and commands instead. |
| DF-010 | implementation | medium | confirmed | open | `Dockerfile`, `docs/DEPLOYMENT.md`, `test_e2e_verification.py` | Build-time test path mismatch can silently skip intended checks. | Keep Docker/test command paths aligned with repository layout. |
| DF-011 | implementation | low | confirmed | open | `src/study_query_llm/algorithms/recipes.py` (`COMPOSITE_RECIPES["cosine_kllmeans_no_pca"]`), `src/study_query_llm/pipeline/analyze.py` (TODO at runtime path + hardcoded `parameters_schema` fallback for `cosine_kllmeans_no_pca`), `scripts/register_clustering_methods.py` (`cosine_kllmeans_no_pca` branch), `docs/audit/coupling_acid_audit_2026-04-24/PROPOSALS.md` (T2.3) | `cosine_kllmeans_no_pca` is registered as a `MethodDefinition` and present in `COMPOSITE_RECIPES`, but has no `AlgorithmSpec`/runner in `pipeline/clustering/registry.py`; analyze keeps a hardcoded `parameters_schema` fallback and TODO that references the unresolved runtime mismatch. Several service-layer tests use the name as a stand-in algorithm, masking the absence of a real runner. Tracked here per Slice 2 Wave 1 reconciliation Item 9 (deferred from W1-PR3). | Pick one: (a) wire a dedicated runner under the bundled grammar (e.g. `cosine+kllmeans+fixed`) and add a registry spec + recipes entry, or (b) drop from `COMPOSITE_RECIPES` + remove the analyze fallback + retire the registration branch. Either way, remove the TODO in `analyze.py` and update `parity` claim C041 + `METHOD_RECIPES.md` recipe count. |

## Validation Queue (Hypotheses)

### VQ-001 - Orchestration claim scaling

- Related flaw: `DF-007`
- Needed validation:
  - Queue-size/load profiling on claim/lease loop.
  - Lock contention and fairness under concurrent workers.

### VQ-002 - Multi-surface operator complexity

- Related flaw: `DF-008`
- Needed validation:
  - On-call simulation for common failure scenarios.
  - Measure mean-time-to-recover with current runbook shape.

## Resolution Policy

- Keep resolved flaws in this file but mark `status=mitigated` and add the implementing PR/commit reference.
- Do not remove old entries; this file is a durable design-debt register.
