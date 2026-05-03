# API Quick Reference (Current)

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-05-03

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
  - Clustering axes and payload identity keep the persisted `summarizer` contract (`parameter_axes["summarizers"]` and payload field `summarizer`); `"None"` remains first-class.
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
  - Root `scripts/run_*.py` commands remain compatibility wrappers; canonical runtime behavior lives under `src/study_query_llm/**`.
- Pipeline analyze surface:
  - `study_query_llm.pipeline.analyze.analyze(snapshot_group_id, embedding_batch_group_id=None, ..., method_name=..., run_key=...)`
  - Embedding-backed runs normalize `parameters["representation_type"]` / `parameters["embedding_representation"]` (either key may be set). Accepted clustering input is **`full` only**. Values `label_centroid` and legacy alias `intent_mean` raise `ValueError` with migration text beginning `representation_type 'label_centroid' (alias 'intent_mean') was retired in Slice 1.6.` Snapshot-only methods continue to overwrite user-supplied representation keys to `snapshot_only` for fingerprint/provenance mode identity (unchanged).
  - Built-in clustering runner dispatch is registry-driven via `study_query_llm.pipeline.clustering.registry` (**23** bundled methods; authoritative list from `iter_algorithm_specs()` / `registry.py`). All ship with `provenance_envelope="none"`; the `clustering_v1` envelope literal was retired (Slice 1.5). Slice 2 Wave 1 adds `pipeline/clustering/grammar.py` (`parse_method_name`), `AlgorithmSpec.preprocessing_chain` + `parameters_schema`, fixed-bundle pipeline synthesis (`runner_common.synthesize_fixed_bundled_payload`, invoked from `analyze()` when `fit_mode=single_fit` and the preprocessing chain is non-empty), and grammar-bound sklearn runners. For bundled sweep methods (`kmeans+normalize+pca+sweep`, `gmm+normalize+pca+sweep`), `analyze()` continues to inject `_v1_pipeline_{resolved,effective}` via `_synthesize_v1_pipeline_for_bundled_method` only (unchanged fingerprint path). For methods with `+pca+` in the name, `pca_n_components` is **required** in parameters; values above `max(1, min(embedding_dim, n_samples-1))` raise `ValueError` (no silent clamp). The shipped sweep runners (`kmeans_runner`, `gmm_runner`) raise `ValueError` if called directly without synthesized pipeline payloads (Slice 2 invariant lock). `ensure_composite_recipe` / `_resolve_method_definition_id` take `parameters_schema` from the clustering registry when the composite name resolves to an `AlgorithmSpec` (fallback remains for `cosine_kllmeans_no_pca`).
  - The legacy method names (`hdbscan`, `kmeans+silhouette+kneedle`, `gmm+bic+argmin`) are pinned in `DEPRECATED_LEGACY_CLUSTERING_METHODS` and rejected at the top of `analyze()` by `raise_if_deprecated_clustering_method`. The guard fires *before* runner resolution so explicit `method_runner` injection cannot bypass it; historical `provenanced_runs` rows under those names remain queryable.
  - BANK77 strategy CLI tokens (`hdbscan`, `kmeans_silhouette_kneedle`, `gmm_bic_argmin`) are kept stable for operator continuity by attaching them as `strategy_aliases` on the bundled-grammar specs in the registry; the alias index resolves them to the bundled-grammar method names.
  - The bundled `kmeans+normalize+pca+sweep` and `gmm+normalize+pca+sweep` methods always include `normalize -> pca -> <algorithm>` in the effective pipeline (the legacy `<= 200` PCA-skip branch was dropped). `pca_n_components` is a method parameter (default 100, clamped at runtime); `kmeans_distance_metric` defaults to `cosine` and `gmm_covariance_type` defaults to `full`.
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
