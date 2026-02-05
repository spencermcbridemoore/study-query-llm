# AGENTS.md - Study Query LLM

## Project Context
Panel-based web application for running LLM inference experiments across multiple providers (Azure OpenAI, OpenAI, Hyperbolic) with PostgreSQL database and analytics.

## Current Work Status
- **Active Work**: Analysis/Experimentation - `notebooks/pca_kllmeans_sweep.ipynb` (PCA/KLLMeans sweep analysis using completed Phase 7 features)
- **Development Status**: See `docs/IMPLEMENTATION_PLAN.md` for feature implementation phase status
- **Source of Truth**: `docs/IMPLEMENTATION_PLAN.md` tracks all implementation phases (✅ ⚠️ ⬜)

## Key Documentation
- **Planning & Status**: `docs/IMPLEMENTATION_PLAN.md` - Phased roadmap and current status
- **Architecture**: `docs/ARCHITECTURE.md` - System design and patterns
- **Coding Rules**: `.cursorrules` - Detailed technical conventions (Cursor-specific)
- **Contributing**: `CONTRIBUTING.md` - Development process and guidelines
- **Standing Orders**: `docs/STANDING_ORDERS.md` - Consistency guidelines

## Essential Constraints
- Always use v2 database schema for new features (`models_v2.py`, `connection_v2.py`, `raw_call_repository.py`)
- Follow bottom-up, incremental development approach
- Update `docs/IMPLEMENTATION_PLAN.md` status markers when completing work
- Never commit `.env` files or API keys (see `SECURITY.md`)
- Use `encoding='utf-8'` for all Python file operations (Windows compatibility)

## Before Starting Work
1. Check `docs/IMPLEMENTATION_PLAN.md` for current phase status
2. Review `docs/ARCHITECTURE.md` for design patterns
3. See `.cursorrules` for detailed coding conventions
