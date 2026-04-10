# API Quick Reference (Current)

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-09

## Configuration

- Main config object: `study_query_llm.config.config`
- Provider config lookup: `config.get_provider_config(provider_name)`
- Configured provider discovery: `config.get_available_providers()`

## Provider Factory (Current Surfaces)

- Chat providers:
  - `ProviderFactory.create_chat_provider(provider_name, model)`
  - `ProviderFactory.get_available_chat_providers()`
- Embedding providers:
  - `ProviderFactory.create_embedding_provider(provider_name)`
  - `ProviderFactory.get_available_embedding_providers()`
- Deployment listing (Azure + OpenRouter):
  - `await ProviderFactory.list_provider_deployments("azure" | "openrouter", modality="chat" | "embedding")`

Notes:

- `ProviderFactory.create()` remains in code but does not represent the full chat-provider surface.
- Prefer the explicit chat/embedding factory methods above for new integrations.
- OpenRouter deployment listing returns runtime catalog metadata (for example context length/modalities/pricing), cached by `ModelRegistry`.

## Core Services

- Inference:
  - `InferenceService.run_inference(prompt, temperature=..., max_tokens=...)`
  - `InferenceService.run_sampling_inference(prompt, n=..., batch_id=...)`
- Analytics:
  - `StudyService(repository=RawCallRepository(...))`
  - `get_summary_stats()`, `get_provider_comparison()`, `get_recent_inferences()`

## Database Access (Canonical)

- v2 connection: `DatabaseConnectionV2`
- v2 repository: `RawCallRepository`
- Primary entities: `RawCall`, `Group`, `GroupMember`, `CallArtifact`, `EmbeddingVector`, `GroupLink`

## Sweep / Execution APIs

- Request lifecycle:
  - `SweepRequestService.create_request(...)`
  - `SweepRequestService.get_request(request_id)` (returns execution-derived analysis state by default)
  - `SweepRequestService.list_requests(status=..., include_fulfilled=..., sweep_type=...)`
- Unified execution records:
  - `ProvenancedRunService.record_method_execution(...)`
  - `ProvenancedRunService.record_analysis_execution(...)`
  - Canonical writes use `run_kind=execution`; semantic role is in `metadata_json.execution_role`.
- Orchestration job types:
  - Clustering: `run_k_try`, `reduce_k`, `finalize_run`
  - MCQ: `mcq_run`, `analysis_run`
- CLI compatibility surfaces:
  - `python -m study_query_llm.cli sweep-worker --request-id <id>`
  - `python -m study_query_llm.cli analyze --request-id <id>` (compatibility wrapper over orchestrated `analysis_run` jobs)

Execution-model feature flags:

- `SQ_DERIVE_ANALYSIS_STATUS_READ` (default: enabled)
- `SQ_ENABLE_ANALYSIS_JOBS` (default: enabled)
- `SQ_RECORD_ANALYSIS_PARITY` (default: enabled)
- `SQ_UNIFIED_EXECUTION_WRITES` (default: enabled)

## Legacy Reference

`docs/API.md` is retained as legacy context and may contain stale v1-era examples.
