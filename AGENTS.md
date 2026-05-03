# AGENTS.md - Study Query LLM

## Project Context
Panel-based web application for running LLM inference experiments across multiple providers (Azure OpenAI, OpenAI, Hyperbolic) with PostgreSQL database and analytics.

## Source of Truth (binding)

The single source of truth for which docs an agent may consult is
`.cursor/rules/living-docs-only.mdc` (always-on). It enumerates the **living
set**, maps each living doc to the code surface it governs, and lists the
**restricted set** (history/deprecated/plans) that must not be opened without
explicit user instruction. Do not duplicate or override that table here.

## Current Work Status
- **Active Work**: Analysis/Experimentation workflows in `notebooks/`
- **Development Status (current)**: See `docs/living/CURRENT_STATE.md`

## Key Documentation (living-only quick links)
- **Docs Navigation**: `docs/README.md` - living/runbook/history/deprecated taxonomy
- **Current Status**: `docs/living/CURRENT_STATE.md`
- **Current Architecture**: `docs/living/ARCHITECTURE_CURRENT.md`
- **Current API Surface**: `docs/living/API_CURRENT.md`
- **Canonical Data Pipeline**: `docs/DATA_PIPELINE.md`
- **Five-stage pipeline contract**: `acquire -> parse -> snapshot -> embed -> analyze` (full-matrix embed; non-full representations derived in analyze)
- **Scheduling/provenance boundary**: `docs/living/SCHEDULING_PROVENANCE_BOUNDARY.md`
- **Method recipes**: `docs/living/METHOD_RECIPES.md`
- **DB ops entrypoint**: `docs/runbooks/README.md`
- **Jetstream PC -> Postgres SSH tunnel**: `deploy/jetstream/LOCAL_DEV_TUNNEL.md`
- **Clone Jetstream DB into local Docker**: `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`
- **Coding Rules**: `.cursorrules` - technical conventions (no source-of-truth duplication)
- **Living-docs gate**: `.cursor/rules/living-docs-only.mdc` - binding doc<->code map and restricted set
- **Git identity guardrail**: `.cursor/rules/git-identity-shell-guardrail.mdc`
- **Contributing**: `CONTRIBUTING.md`
- **Standing Orders**: `docs/STANDING_ORDERS.md`

## Local development
- **`.venv`:** `python -m venv .venv`, activate, then `pip install -e ".[dev]"`.
- **Miniconda:** dedicated env **`study-query-llm`** from repo root: `conda env create -f environment.yml` (first time) or `conda env update -f environment.yml -n study-query-llm --prune`, then `conda activate study-query-llm`.

## Essential Constraints
- Always use v2 database schema for new features (`models_v2.py`, `connection_v2.py`, `raw_call_repository.py`)
- Pipeline changes must preserve five-stage contracts in `docs/DATA_PIPELINE.md` (parser identity/version, deterministic snapshot sampling, analyze dual-input lineage).
- Follow bottom-up, incremental development approach
- Update `docs/living/CURRENT_STATE.md` and `docs/review/DOC_PARITY_LEDGER.md` when current behavior/claims change
- **MANDATORY: Stage and commit when completing features/tasks** - do not wait for user to ask
- Never commit `.env` files or API keys (see `SECURITY.md`)
- Use `encoding='utf-8'` for all Python file operations (Windows compatibility)

## Code Layer Boundary
- **Tier A (canonical runtime):** `src/study_query_llm/**` orchestration, pipeline, and runtime logic.
- **Tier B (compatibility surfaces):** root `scripts/run_*.py` wrappers and stable operator entrypoints.
- **Tier C (historical):** `scripts/history/**` and archived experiment artifacts.
- **Tier D (policy mirror):** `docs/living/**` and runbooks that document current behavior.

Boundary rule:
- New orchestration/job logic belongs in Tier A (`src/study_query_llm/services/jobs/` or `src/study_query_llm/experiments/`), not in root `scripts/`.
- Root `scripts/` are wrappers/utilities unless explicitly documented otherwise.

Worked counter-example:
- Adding a new clustering method belongs under `src/study_query_llm/pipeline/clustering/` (registry + runner + tests). Do **not** add a new `scripts/run_*.py` orchestration surface for it.

Terminology lock:
- `orchestration_job`: schedulable control-plane unit.
- `worker_runtime`: process that claims/executes orchestration jobs.
- `standalone_mode`: orchestration profile, not a separate legacy control plane.
- `summarizer`: current persisted clustering axis key (`"None"` is first-class).
- `compatibility_wrapper`: command/path-stable delegate into Tier A runtime.

Known transitional violations (phase-2 backlog; document honestly):
- `src/study_query_llm/services/jobs/runtime_supervisors.py` still spawns `scripts/run_local_300_2datasets_worker.py` (see current spawn callsites around lines 259 and 503).
- `src/study_query_llm/experiments/sweep_worker_main.py` still carries campaign-era naming defaults (`OUT_PREFIX = "local_300_2d_14e_4s"` and fixed dataset defaults).
- `src/study_query_llm/providers/managers/ollama.py` still contains sweep-era `LLM_SUMMARIZERS` wording/examples.

Transitional-violation review cadence:
- For every boundary-related PR, include a "Transitional Violations Review" checkpoint: each listed item must be either fixed and removed or explicitly re-accepted with date/reason.

## Before Starting Work
1. Check `docs/living/CURRENT_STATE.md` for current capability/status
2. Review `docs/living/ARCHITECTURE_CURRENT.md` for current design patterns
3. See `.cursorrules` for detailed coding conventions
