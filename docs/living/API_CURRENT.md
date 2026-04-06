# API Quick Reference (Current)

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-06

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
- Deployment listing (Azure):
  - `await ProviderFactory.list_provider_deployments("azure", modality="chat" | "embedding")`

Notes:

- `ProviderFactory.create()` remains in code but does not represent the full chat-provider surface.
- Prefer the explicit chat/embedding factory methods above for new integrations.

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

## Legacy Reference

`docs/API.md` is retained as legacy context and may contain stale v1-era examples.
