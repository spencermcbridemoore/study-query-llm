# Current State (Authoritative)

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-06

## Scope

This document is the canonical "what exists and works now" summary for the repository.

## What Is Implemented

### Runtime Surface

- Panel application with inference + analytics views (`panel_app/`).
- Package CLI entrypoint: `python -m study_query_llm.cli`.
- Sweep/job workers and supervisors under `src/study_query_llm/services/jobs/` and `src/study_query_llm/experiments/`.

### Database Model

- Canonical path for new development: v2 schema (`models_v2.py`, `connection_v2.py`, `raw_call_repository.py`).
- Core v2 entities include `RawCall`, `Group`, `GroupMember`, `CallArtifact`, `EmbeddingVector`, and `GroupLink`.
- Legacy v1 schema (`models.py`, `connection.py`, `inference_repository.py`) is retained for compatibility and migration context.

### Providers and Factory Behavior

- Chat-provider creation surface:
  - `ProviderFactory.create_chat_provider(provider_name, model)`
  - Supported names: `azure`, `openrouter`, `local_llm`, `ollama`.
- Embedding-provider creation surface:
  - `ProviderFactory.create_embedding_provider(provider_name)`
  - Supported names: `azure`, `openai`, `huggingface`, `local`, `ollama`.
- `Config.get_available_providers()` exposes configured provider keys (including `openrouter`, `huggingface`, `local`, `local_llm`).

### Service Layer

- `InferenceService` handles inference flow, retries, and persistence integration.
- `StudyService` is v2-based and expects `RawCallRepository`.
- Provenance, artifacts, sweep request lifecycle, and job orchestration services are implemented in `src/study_query_llm/services/`.

## What To Use By Default

- For new DB-backed functionality: use v2 repository and models.
- For chat providers: use `create_chat_provider` path, not legacy assumptions about `create()`.
- For operational execution: prefer package CLI subcommands over legacy script wrappers.

## Known Legacy / Transitional Areas

- `docs/IMPLEMENTATION_PLAN.md` and `docs/ARCHITECTURE.md` contain historical phase narrative and legacy context.
- `docs/API.md` is deprecated in favor of `docs/living/API_CURRENT.md`.
- `docs/MIGRATION_GUIDE.md` is historical/deprecated migration context.

## Near-Term Documentation Priorities

- Keep parity with `docs/review/DOC_PARITY_LEDGER.md`.
- Update living docs first; append historical notes instead of mixing modes.
- Avoid fixed claims that go stale quickly (for example exact test counts).
