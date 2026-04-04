# AGENTS.md - Study Query LLM

## Project Context
Panel-based web application for running LLM inference experiments across multiple providers (Azure OpenAI, OpenAI, Hyperbolic) with PostgreSQL database and analytics.

## Current Work Status
- **Active Work**: Analysis/Experimentation - `notebooks/pca_kllmeans_sweep.ipynb` (PCA/KLLMeans sweep analysis using completed Phase 7 features)
- **Development Status**: See `docs/IMPLEMENTATION_PLAN.md` for feature implementation phase status
- **Source of Truth**: `docs/IMPLEMENTATION_PLAN.md` tracks all implementation phases (✅ ⚠️ ⬜)

## Key Documentation
- **Planning & Status**: `docs/IMPLEMENTATION_PLAN.md` - Phased roadmap and current status
- **Architecture**: `docs/ARCHITECTURE.md` - System design and patterns (includes **Standalone sweep worker**: `python -m study_query_llm.cli sweep-worker` / `analyze`, `experiments/sweep_worker_main.py`)
- **Jetstream: PC → Postgres SSH tunnel** (local `DATABASE_URL` via forward): `deploy/jetstream/LOCAL_DEV_TUNNEL.md`
- **Clone Jetstream DB into local Docker** (backup local, `pg_dump` Jetstream, restore): `docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md`
- **Coding Rules**: `.cursorrules` - Detailed technical conventions (Cursor-specific)
- **Contributing**: `CONTRIBUTING.md` - Development process and guidelines
- **Standing Orders**: `docs/STANDING_ORDERS.md` - Consistency guidelines

## Local development
- **`.venv`:** `python -m venv .venv`, activate, then `pip install -e ".[dev]"`.
- **Miniconda:** dedicated env **`study-query-llm`** from repo root: `conda env create -f environment.yml` (first time) or `conda env update -f environment.yml -n study-query-llm --prune`, then `conda activate study-query-llm`.

## Essential Constraints
- Always use v2 database schema for new features (`models_v2.py`, `connection_v2.py`, `raw_call_repository.py`)
- Follow bottom-up, incremental development approach
- Update `docs/IMPLEMENTATION_PLAN.md` status markers when completing work
- **MANDATORY: Stage and commit when completing features/tasks** - do not wait for user to ask
- Never commit `.env` files or API keys (see `SECURITY.md`)
- Use `encoding='utf-8'` for all Python file operations (Windows compatibility)

## Before Starting Work
1. Check `docs/IMPLEMENTATION_PLAN.md` for current phase status
2. Review `docs/ARCHITECTURE.md` for design patterns
3. See `.cursorrules` for detailed coding conventions
