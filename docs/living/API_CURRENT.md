# API Quick Reference (Current)

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-05-01

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
  - `get_summary_stats()`, `get_provider_comparison()`, `get_recent_inferences()` (optional `modality` / `status` filters; default remains text+success; pass `None` for both to list all `raw_calls` rows in limit order)
- Method contract helpers:
  - `MethodService.resolve_method_input_requirements(name, version=None)` normalizes `input_schema.required_inputs` with backward-compatible defaults (`snapshot=true`, `embedding_batch=true` when absent/malformed).
  - `MethodService.register_method(..., input_schema=...)` validates `input_schema.required_inputs` on new writes (`required_inputs` must be an object; `snapshot` and `embedding_batch` must be booleans when present).

## Database Access (Canonical)

- v2 connection: `DatabaseConnectionV2`
- v2 repository: `RawCallRepository`
- Primary entities: `RawCall`, `Group`, `GroupMember`, `CallArtifact`, `EmbeddingVector`, `GroupLink`
- Non-sqlite `DatabaseConnectionV2` constructors now require explicit `write_intent` (or `SQLLM_WRITE_INTENT`) with lane/intent compatibility checks at connection time.
- Canonical-intent artifact writes are fail-closed to Azure Blob storage (`ArtifactService(write_intent=WriteIntent.CANONICAL)` rejects local backends).

## Sweep / Execution APIs

- Request lifecycle:
  - `SweepRequestService.create_request(...)`
  - `SweepRequestService.get_request(request_id)` (returns execution-derived analysis state by default)
  - `SweepRequestService.list_requests(status=..., include_fulfilled=..., sweep_type=...)`
  - Planner behavior: adapter-driven orchestration graph specs are enqueued via `SweepRequestService.ensure_orchestration_jobs(...)` (no planner-type hardcoding in service branches).
- Unified execution records:
  - `ProvenancedRunService.record_method_execution(...)`
  - `ProvenancedRunService.record_analysis_execution(...)`
  - Canonical writes use `run_kind=execution`; semantic role is in `metadata_json.execution_role`.
- Orchestration job types:
  - Clustering: `run_k_try`, `reduce_k`, `finalize_run`
  - MCQ: `mcq_run`, `analysis_run`
  - Terminology: these are `orchestration_job` types (control-plane units), not `algorithm_iteration` records; `run_k_try` represents a seeded `restart_try` work unit.
  - Job runner dispatch is registry-based (`create_job_runner(...)`); reduce/finalize runners consume the typed reducer plugin seam (`ReducerPlugin` / `ClusteringReducerPlugin`).
- CLI compatibility surfaces:
  - `python -m study_query_llm.cli sweep-worker --request-id <id>`
  - `python -m study_query_llm.cli analyze --request-id <id>` (compatibility wrapper over orchestrated `analysis_run` jobs)
- Pipeline analyze surface:
  - `study_query_llm.pipeline.analyze.analyze(snapshot_group_id, embedding_batch_group_id=None, ..., method_name=..., run_key=...)`
  - Built-in clustering runner dispatch is registry-driven via `study_query_llm.pipeline.clustering.registry` (`hdbscan`, `kmeans+silhouette+kneedle`, `gmm+bic+argmin`, `agglomerative+fixed-k`).
  - v1 clustering provenance envelope (`rules-v1.0.0.yaml` resolver/validators + clustering summary identity fields) applies only to registry methods marked `provenance_envelope=clustering_v1` (currently `hdbscan`, `kmeans+silhouette+kneedle`, `gmm+bic+argmin`).
  - `agglomerative+fixed-k` executes outside the v1 resolver/validator envelope (`provenance_envelope=none`).
  - Input requirements are resolved from `MethodDefinition.input_schema.required_inputs`; default behavior remains embedding-required when contract metadata is absent.
  - Snapshot-only methods (`required_inputs.embedding_batch=false`) execute without embedding artifacts and persist `config_json.analysis_input_mode=snapshot_only` for explicit provenance/fingerprint mode identity.

Embedding sweep runner (snapshot -> provider model sweep):

- Script: `scripts/run_snapshot_embedding_model_sweep.py`
- Core controls:
  - model-level concurrency: `--engine-concurrency`
  - provider budget cap: `--provider-concurrency-budget`
  - chunk workers (opt-in): `--chunk-worker-concurrency`
  - chunk fallback (opt-in): `--chunk-circuit-breaker-fallback`, `--chunk-failure-fallback-threshold`
  - model filtering: `--exclude-model`, `--exclude-models-file`
  - best-effort availability prefilter: `--pre-validate-models`, `--validation-concurrency`, `--validation-cache-ttl-seconds`, `--validation-timeout-seconds`
  - retry/singleflight runtime passthroughs: `--max-retries`, `--initial-wait-seconds`, `--max-wait-seconds`, `--singleflight-lease-seconds`, `--singleflight-wait-timeout-seconds`, `--singleflight-poll-seconds`
- Environment override:
  - `SQLLM_DISABLE_PARALLEL_CHUNKS=1` forces effective chunk-worker concurrency to 1.

Execution-model feature flags:

- `SQ_DERIVE_ANALYSIS_STATUS_READ` (default: enabled)
- `SQ_ENABLE_ANALYSIS_JOBS` (default: enabled)
- `SQ_RECORD_ANALYSIS_PARITY` (default: enabled)
- `SQ_UNIFIED_EXECUTION_WRITES` (default: enabled)

## Legacy Reference

`docs/API.md` is retained as legacy context and may contain stale v1-era examples.
