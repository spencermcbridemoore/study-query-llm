# Current State (Authoritative)

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-10

## Scope

This document is the canonical "what exists and works now" summary for the repository.

## What Is Implemented

### Runtime Surface

- Panel application with inference, analytics, embeddings, groups, sweep explorers, and a **Storage / DB stats** tab (`panel_app/views/storage_stats.py`) for PostgreSQL size metrics (when using Postgres), v2 row counts, artifact storage configuration summary, optional capped Azure blob prefix usage probe, and pointers to the local DB clone runbook.
- Package CLI entrypoint: `python -m study_query_llm.cli`.
- Sweep/job workers and supervisors under `src/study_query_llm/services/jobs/` and `src/study_query_llm/experiments/`.

### Database Model

- Canonical path for new development: v2 schema (`models_v2.py`, `connection_v2.py`, `raw_call_repository.py`).
- Core v2 entities include `RawCall`, `Group`, `GroupMember`, `CallArtifact`, `EmbeddingVector`, and `GroupLink`.
- Unified execution-provenance entity `ProvenancedRun` now uses canonical `run_kind=execution` for new writes; semantic role is stored in `metadata_json.execution_role` (`method_execution` / `analysis_execution`), with compatibility support for legacy row kinds during migration.
- Legacy v1 schema (`models.py`, `connection.py`, `inference_repository.py`) is retained for compatibility and migration context.

### Providers and Factory Behavior

- Chat-provider creation surface:
  - `ProviderFactory.create_chat_provider(provider_name, model)`
  - Supported names: `azure`, `openrouter`, `local_llm`, `ollama`.
- Embedding-provider creation surface:
  - `ProviderFactory.create_embedding_provider(provider_name)`
  - Supported names: `azure`, `openrouter`, `openai`, `huggingface`, `local`, `ollama`.
- Deployment/model discovery surface:
  - `ProviderFactory.list_provider_deployments(provider_name, modality=...)`
  - Current runtime-supported providers: `azure`, `openrouter`.
- OpenRouter deployment discovery maps runtime model metadata (context window, modalities, pricing/limits fields when present) into `DeploymentInfo`, and `ModelRegistry` persists that metadata in cache.
- `Config.get_available_providers()` exposes configured provider keys (including `openrouter`, `huggingface`, `local`, `local_llm`).

### Service Layer

- `InferenceService` handles inference flow, retries, and persistence integration.
- `StudyService` is v2-based and expects `RawCallRepository`.
- Provenance, artifacts, sweep request lifecycle, and job orchestration services are implemented in `src/study_query_llm/services/`.
- `SweepRequestService` supports typed sweep requests (`clustering`, `mcq`), execution-derived analysis state reads, parity diagnostics against legacy metadata fields, and OrchestrationJob planning.
- Standalone sweep worker execution is now an OrchestrationJob profile for both clustering and MCQ (jobs are planned and consumed through the canonical orchestration table).
- MCQ orchestration now plans both `mcq_run` execution jobs and dependent `analysis_run` jobs; `python -m study_query_llm.cli analyze` is a compatibility wrapper that processes those orchestration analysis jobs rather than a separate analysis write path.
- New MCQ runs persist explicit `provenanced_runs` method-execution rows with `determinism_class=non_deterministic`; analysis outcomes dual-write to compatibility `analysis_results` and canonical `provenanced_runs` execution rows; legacy rows remain visible through compatibility mapping in the unified execution view.
- Clustering ingestion now normalizes `dataset_snapshot_ids` and writes a primary snapshot pointer to `provenanced_runs.input_snapshot_group_id`; unified compatibility reads infer this pointer from legacy run metadata or `depends_on` run->snapshot links when explicit execution rows are absent.
- BANK77 bootstrap is now scriptable via `scripts/create_bank77_snapshot_and_embeddings.py`, which creates/reuses a deterministic `dataset_snapshot` manifest from `mteb/banking77`, persists full embedding matrices (`N x d`) plus intent means (`77 x d`), writes deterministic intent-row mapping metadata, and verifies lineage/integrity through readback checks.
- **Layer 0 dataset acquisition:** `study_query_llm.datasets.acquisition` plus `scripts/record_dataset_download.py` download pinned public files (`ACQUIRE_REGISTRY` slugs: `ausem`, `sources_uncertainty_qc` via Zenodo 16912394, `semeval2013_sra_5way` via pinned GitHub mirror gold files), expose `zenodo_file_download_url` for stable Zenodo GET URLs, write `acquisition.json` + mirrored `files/` with per-file SHA-256, and optionally create a `dataset` group with `CallArtifact` rows (`dataset_acquisition_file`, `dataset_acquisition_manifest`) via `ArtifactService.store_group_blob_artifact` when `ARTIFACT_STORAGE_BACKEND=azure_blob` and `DATABASE_URL` are set (`docs/DATASET_ACQUISITION_LAYER0.md`).
- Artifact writes enforce an Azure blob quota hard-stop (default 100 GiB; configurable via env).
- Embedding token-limit validation can use discovered model `context_length` from `ModelRegistry` cache when available (provider+deployment key match), then falls back to existing static/inferred limits.

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
