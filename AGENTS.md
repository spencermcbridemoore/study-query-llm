# AGENTS.md - Study Query LLM

## Project Context
Panel-based web application for running LLM inference experiments across multiple providers (Azure OpenAI, OpenAI, Hyperbolic) with PostgreSQL database and analytics.

## Current Work Status
- **Active Work**: Analysis/Experimentation workflows in `notebooks/` (including archived notebook variants)
- **Development Status (current)**: See `docs/living/CURRENT_STATE.md`
- **Historical roadmap context**: `docs/IMPLEMENTATION_PLAN.md` (phased chronology)

## Key Documentation
- **Docs Navigation**: `docs/README.md` - living/runbook/history/deprecated routing
- **Current Status**: `docs/living/CURRENT_STATE.md`
- **Current Architecture**: `docs/living/ARCHITECTURE_CURRENT.md`
- **Canonical Data Pipeline**: `docs/DATA_PIPELINE.md`
- **Five-stage pipeline contract**: `acquire -> parse -> snapshot -> embed -> analyze` (full-matrix embed; non-full representations derived in analyze)
- **Historical Architecture**: `docs/ARCHITECTURE.md`
- **Jetstream: PC → Postgres SSH tunnel** (local `DATABASE_URL` via forward): `deploy/jetstream/LOCAL_DEV_TUNNEL.md`
- **Clone Jetstream DB into local Docker** (backup local, `pg_dump` Jetstream, restore): `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`
- **Coding Rules**: `.cursorrules` - Detailed technical conventions (Cursor-specific)
- **Git identity guardrail**: `.cursor/rules/git-identity-shell-guardrail.mdc` - Windows cross-shell commit identity checks
- **Contributing**: `CONTRIBUTING.md` - Development process and guidelines
- **Standing Orders**: `docs/STANDING_ORDERS.md` - Consistency guidelines

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

## Before Starting Work
1. Check `docs/living/CURRENT_STATE.md` for current capability/status
2. Review `docs/living/ARCHITECTURE_CURRENT.md` for current design patterns
3. See `.cursorrules` for detailed coding conventions
