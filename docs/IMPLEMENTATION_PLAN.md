# Implementation Plan - Study Query LLM

## Overview

This document outlines the phased implementation plan for building the Study-Query-LLM application. The approach is **bottom-up and incremental**, where each phase produces a testable, working component before moving to the next.

## Implementation Philosophy

### Bottom-Up, Incremental Development

✅ **Each component is testable in isolation**
✅ **Validate each piece works before adding complexity**
✅ **Easy to course-correct if requirements change**
✅ **Natural dependency order (no circular dependencies)**
✅ **Can use each layer immediately as you build it**

### Testing at Each Step

Every phase includes a "Test" section describing how to validate the component works. Don't skip these - they ensure each building block is solid before you stack the next layer.

### Status Legend (current repo)
✅ Implemented  
⚠️ Partially implemented  
⬜ Not implemented

### Repo Layout Note
Core Python modules live under `src/study_query_llm/` (providers, services, db, config). Panel UI lives in `panel_app/app.py`.

---

## Phase 1: LLM Provider Abstraction Layer
**Goal:** Build the foundational inference interface - no database, no GUI

**Dependencies:** None (this is the foundation)

### Step 1.1: Base Provider Interface ✅

**Implementation:** [`src/study_query_llm/providers/base.py`](src/study_query_llm/providers/base.py)

**Design:**
- Abstract base class `BaseLLMProvider` with `complete()` method
- Standardized `ProviderResponse` dataclass (text, provider, tokens, latency_ms, metadata, raw_response)
- All providers must implement `get_provider_name()`

**Tests:** [`tests/test_providers/test_base.py`](tests/test_providers/test_base.py)

---

### Step 1.2: Azure Provider Implementation ✅

**Implementation:** [`src/study_query_llm/providers/azure_provider.py`](src/study_query_llm/providers/azure_provider.py)

**Design:**
- Uses `AsyncAzureOpenAI` from OpenAI SDK
- Configurable: api_key, endpoint, deployment_name, api_version
- Measures latency and extracts token usage from response
- Returns standardized `ProviderResponse`

**Dependencies:** `pip install openai`

**Tests:** [`tests/test_providers/test_azure.py`](tests/test_providers/test_azure.py)

---

### Step 1.3: OpenAI Provider Implementation ⬜

**Files to create:**
- `src/study_query_llm/providers/openai_provider.py`

**Design:**
- Similar structure to Azure provider but uses `AsyncOpenAI` client
- Configurable model name (default: "gpt-4")
- Standardized response format matching `BaseLLMProvider` interface

**Dependencies:** `pip install openai`

**Test strategy:** Verify API calls return standardized `ProviderResponse` with tokens and latency

---

### Step 1.4: Hyperbolic Provider Implementation ⬜

**Files to create:**
- `src/study_query_llm/providers/hyperbolic_provider.py`

**Design:**
- HTTP-based provider using `httpx` for async requests
- Configurable base_url (default: "https://api.hyperbolic.xyz")
- Adapt to Hyperbolic's actual API structure (may differ from OpenAI format)

**Dependencies:** `pip install httpx`

**Test strategy:** Verify API calls return standardized `ProviderResponse` (adjust based on Hyperbolic API docs)

---

### Step 1.5: Provider Factory ⚠️ (Azure only)

**Implementation:** [`src/study_query_llm/providers/factory.py`](src/study_query_llm/providers/factory.py)

**Design:**
- Static factory method `create(provider_name, **config)` returns `BaseLLMProvider`
- Currently supports Azure only
- `get_available_providers()` returns list of supported provider names

**Still missing:**
- Factory support for OpenAI and Hyperbolic providers

**Tests:** [`tests/test_providers/test_factory.py`](tests/test_providers/test_factory.py)

---

### Phase 1 Milestone ⚠️

**What you have now:**
- Abstract provider interface + standardized response format
- Azure OpenAI provider implementation
- Provider factory (Azure only)

**Still missing:**
- OpenAI provider
- Hyperbolic provider
- Factory support for additional providers

**Next:** Implement remaining providers and expand the factory

---

## Phase 2: Business Logic Layer (Service Layer)
**Goal:** Add retry, preprocessing, and orchestration logic

**Dependencies:** Phase 1 (Provider layer)

### Step 2.1: Basic Inference Service ✅

**Implementation:** [`src/study_query_llm/services/inference_service.py`](src/study_query_llm/services/inference_service.py)

**Design:**
- Wraps `BaseLLMProvider` with business logic layer
- Standardized response format (dict with 'response' and 'metadata')
- Optional repository parameter for database logging (added in Phase 3)

**Tests:** [`tests/test_services/test_inference.py`](tests/test_services/test_inference.py)

---

### Step 2.2: Add Retry Logic ✅

**Implementation:** [`src/study_query_llm/services/inference_service.py`](src/study_query_llm/services/inference_service.py)

**Design:**
- Uses `tenacity` library for retry decorators
- Exponential backoff (1s → 10s max)
- Retries on `TimeoutError`, `ConnectionError`, and other transient exceptions
- Configurable max_retries and initial_wait

**Dependencies:** `pip install tenacity`

**Test strategy:** Simulate network errors and verify retry behavior

---

### Step 2.3: Add Prompt Preprocessing ✅

**Implementation:** 
- [`src/study_query_llm/services/preprocessors.py`](src/study_query_llm/services/preprocessors.py)
- Integrated into [`src/study_query_llm/services/inference_service.py`](src/study_query_llm/services/inference_service.py)

**Design:**
- `PromptPreprocessor` class with static methods:
  - `clean_whitespace()`: Normalize whitespace
  - `apply_template()`: Apply prompt templates
  - `truncate()`: Limit prompt length (default 10k chars)
  - `remove_pii()`: Basic PII removal (emails, phone numbers)
- Optional preprocessing flag in `InferenceService`

**Tests:** [`tests/test_services/test_preprocessing.py`](tests/test_services/test_preprocessing.py)

---

### Step 2.4: Multi-Turn Conversation Service ⬜ (Optional)

**Files to create:**
- `src/study_query_llm/services/conversation_service.py`

**Design:**
- Manages conversation state (in-memory dict keyed by conversation_id)
- Supports system prompts
- Maintains message history (role/content pairs)
- Assumes OpenAI-style chat format (messages array)

**Test strategy:** Verify conversation context maintained across multiple turns

---

### Step 2.5: Request Batching Service ⚠️ (batching/sampling in InferenceService; no dedup)

**Design:**
- Batching and sampling functionality exists in `InferenceService`
- Deduplication service not yet implemented

**Still missing:**
- Request deduplication service for identical concurrent requests
- Hash-based request identity for de-duping

**Test strategy:** Verify duplicate concurrent requests share single API call

---

### Phase 2 Milestone ⚠️

**What you have now:**
- Inference service with retry + preprocessing
- Batch + sampling helpers in `InferenceService`

**Still missing (optional):**
- Conversation service
- Request deduplication service

**Next:** Add conversation + dedup services if needed

---

## Phase 3: Database Layer
**Goal:** Add persistence for inference results

**Dependencies:** Phase 1 (Providers), Phase 2 (Services)

**Note:** This phase implements the v1 database schema. See Phase 7 for the v2 immutable schema.

### Step 3.1: Database Models ✅

**Implementation:** [`src/study_query_llm/db/models.py`](src/study_query_llm/db/models.py)

**Design:**
- `InferenceRun` model: stores prompt, response, provider, tokens, latency_ms, metadata, created_at
- SQLAlchemy declarative base
- Indexed fields: provider, created_at

**Dependencies:** `pip install sqlalchemy psycopg2-binary`

**Tests:** [`tests/test_db/test_models.py`](tests/test_db/test_models.py)

---

### Step 3.2: Database Connection ✅

**Implementation:** [`src/study_query_llm/db/connection.py`](src/study_query_llm/db/connection.py)

**Design:**
- `DatabaseConnection` class manages SQLAlchemy engine and sessions
- `session_scope()` context manager for transactional operations
- Supports PostgreSQL and SQLite connection strings
- `init_db()` creates all tables

**Tests:** [`tests/test_db/test_connection.py`](tests/test_db/test_connection.py)

---

### Step 3.3: Repository - Write Operations ✅

**Implementation:** [`src/study_query_llm/db/inference_repository.py`](src/study_query_llm/db/inference_repository.py)

**Design:**
- `InferenceRepository` class with session-based operations
- `insert_inference_run()`: Insert single record, returns ID
- `batch_insert_inferences()`: Batch insert multiple records

**Tests:** [`tests/test_db/test_repository.py`](tests/test_db/test_repository.py)

---

### Step 3.4: Repository - Query Operations ✅

**Implementation:** [`src/study_query_llm/db/inference_repository.py`](src/study_query_llm/db/inference_repository.py)

**Design:**
- `get_inference_by_id()`: Retrieve by ID
- `query_inferences()`: Filter by provider, date_range, pagination
- `get_provider_stats()`: Aggregate statistics by provider (count, avg_tokens, avg_latency, total_tokens)
- `search_by_prompt()`: Text search in prompts
- `get_total_count()`: Total inference count

**Tests:** [`tests/test_db/test_repository.py`](tests/test_db/test_repository.py)

---

### Step 3.5: Integrate Services with Repository ✅

**Implementation:** [`src/study_query_llm/services/inference_service.py`](src/study_query_llm/services/inference_service.py)

**Design:**
- Optional `repository` parameter in `InferenceService.__init__()`
- If repository provided, `run_inference()` automatically persists results
- Returns dict with 'id' field when persisted

**Tests:** [`tests/test_services/test_inference_with_db.py`](tests/test_services/test_inference_with_db.py)

---

### Phase 3 Milestone ✓

**What you have now:**
- Full database layer with SQLAlchemy
- Repository pattern for all DB operations
- Write operations (insert single/batch)
- Query operations (filters, aggregations, search)
- Services integrated with database
- Everything still testable

**Next:** Add analytics service and GUI

---

## Phase 4: Study/Analytics Service
**Goal:** Business logic for analyzing stored data

**Dependencies:** Phase 3 (Database layer)

### Step 4.1: Study Service ✅

**Implementation:** [`src/study_query_llm/services/study_service.py`](src/study_query_llm/services/study_service.py)

**Design:**
- Wraps `InferenceRepository` with analytics methods returning pandas DataFrames
- `get_provider_comparison()`: Provider metrics comparison with cost estimates
- `get_recent_inferences()`: Recent runs with truncated text
- `get_time_series_data()`: Time-series aggregation (day/hour/minute) for visualization
- `search_prompts()`: Search historical prompts
- `get_summary_stats()`: Overall statistics dict

**Dependencies:** `pip install pandas`

**Tests:** [`tests/test_services/test_study_service.py`](tests/test_services/test_study_service.py)

---

### Phase 4 Milestone ✓

**What you have now:**
- Complete analytics layer
- Provider comparisons
- Time-series data
- Search functionality
- Summary statistics
- Data ready for visualization in GUI

**Next:** Build the Panel GUI

---

## Phase 5: GUI Integration (Panel)
**Goal:** Connect everything to the Panel interface

**Dependencies:** All previous phases

### Step 5.1: Simple Inference UI ✅

**Implementation:** [`panel_app/app.py`](panel_app/app.py)

**Design:**
- Panel-based web interface
- Provider selection dropdown
- Prompt input text area
- Run inference button
- Response display with metadata (tokens, latency)

**Dependencies:** `pip install panel hvplot holoviews bokeh`

**Test strategy:** Run inference through GUI, verify it appears in database

---

### Step 5.2: Analytics Dashboard ⚠️ (tables/summary only)

**Implementation:** [`panel_app/app.py`](panel_app/app.py)

**Design:**
- Provider comparison bar chart
- Time-series line chart
- Recent inferences table
- Summary statistics cards
- Search interface

**Still missing:**
- Full analytics charts/time-series visualization
- Search UI integration

**Test strategy:** Verify charts update with real data from `StudyService`

---

### Step 5.3: Configuration Management ⚠️ (config exists; `.env.example` missing)

**Implementation:** [`src/study_query_llm/config.py`](src/study_query_llm/config.py)

**Design:**
- Loads from environment variables (via `python-dotenv`)
- `ProviderConfig` dataclass for each provider
- `DatabaseConfig` for database connection
- `AppConfig` class aggregates all configs

**Still missing:**
- `.env.example` file with template environment variables

**Test strategy:** Verify configuration loads correctly from environment

---

### Phase 5 Milestone ⚠️

**What you have now:**
- Inference UI (Panel)
- Analytics summary + provider comparison + recent table
- Config loader in `src/study_query_llm/config.py`

**Still missing:**
- Analytics charts/time-series + search UI
- `.env.example`

**Next:** Finish analytics UI + add config example

---

## Phase 6: Polish and Deployment
**Goal:** Production readiness

### Step 6.1: Error Handling and Logging ⚠️

**Implementation:** [`src/study_query_llm/utils/logging_config.py`](src/study_query_llm/utils/logging_config.py)

**Design:**
- Comprehensive logging throughout application
- Provider API calls logged
- Database operations logged
- User actions logged

**Still missing:**
- Enhanced error handling in some components
- Database connection resiliency (pool_pre_ping, pool_recycle) for long-running processes and environments with connection timeouts (e.g., Colab, cloud DBs with SSL)

---

### Step 6.2: Unit Tests ✅

**Implementation:** Test suite in [`tests/`](tests/) directory

**Test coverage:**
- [`tests/test_providers/`](tests/test_providers/): Provider tests
- [`tests/test_services/`](tests/test_services/): Service layer tests
- [`tests/test_db/`](tests/test_db/): Database and repository tests

---

### Step 6.3: Documentation ✅

**Implementation:**
- [`README.md`](README.md): Setup instructions
- [`docs/API.md`](docs/API.md): API documentation
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md): User guide
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Architecture overview
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): Deployment guide

---

### Step 6.4: Docker Setup ✅

**Implementation:**
- [`Dockerfile`](Dockerfile): Multi-stage build (Python 3.11 slim, non-root user)
- [`docker-compose.yml`](docker-compose.yml): App service with optional Postgres profile

**Design:**
- Runtime: Python 3.11 slim image, `panel_app.app` exposed on port `5006`
- Default SQLite database persisted via container volume
- Optional Postgres connection through `DATABASE_URL` env var
- Required env vars: Azure OpenAI credentials, optional provider keys, `PANEL_ADDRESS`, `PANEL_PORT`

**Deployment:** See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for full deployment guide

---

## Phase 7: Immutable Capture + Grouping DB (Postgres)

**Goal:** Add a v2 immutable raw capture schema that logs successes + failures across modalities, plus mutable grouping tables for experiments/batches/labels.

**Dependencies:** Phase 3 (DB layer), PostgreSQL target

### Step 7.1: V2 Schema (Immutable Raw Calls) ✅

**Implementation:**
- [`src/study_query_llm/db/models_v2.py`](src/study_query_llm/db/models_v2.py): V2 models (RawCall, Group, GroupMember, CallArtifact, EmbeddingVector)
- [`src/study_query_llm/db/raw_call_repository.py`](src/study_query_llm/db/raw_call_repository.py): Repository for v2 operations
- [`src/study_query_llm/db/connection_v2.py`](src/study_query_llm/db/connection_v2.py): Connection helper for v2 Postgres schema

**Core tables:**
- `RawCall`: Immutable log of all API calls (`provider`, `model`, `modality`, `status`, `request_json`, `response_json`, `error_json`, `latency_ms`, `tokens_json`, `metadata_json`, `created_at`)
- `CallArtifact`: Blob references for multimodal payloads (`uri`, `content_type`, `byte_size`, `metadata_json`)
- `EmbeddingVector`: Embeddings table with pgvector support (`vector`, `dimension`, `norm`, `metadata_json`)
- `Group`: Mutable grouping metadata (`group_type`, `name`, `description`, `created_at`, `metadata_json`)
- `GroupMember`: Join table (`group_id`, `call_id`, `added_at`, `position`, `role`)

**Tests:** [`tests/test_db/test_models_v2.py`](tests/test_db/test_models_v2.py), [`tests/test_db/test_repository_v2.py`](tests/test_db/test_repository_v2.py)

---

### Step 7.2: Log Success + Failure in RawCall ✅

**Update:** [`src/study_query_llm/services/inference_service.py`](src/study_query_llm/services/inference_service.py)

**Design:**
- On success: `status="success"`, `response_json` set
- On failure/exception: `status="failed"`, `response_json=null`, `error_json` set
- Always record `request_json` + runtime metadata (tokens, latency, provider)

**Test strategy:** Verify failed calls persist in v2 DB with proper error logging

---

### Step 7.3: Migration Script (v1 → v2) ✅

**Implementation:** [`scripts/migrate_v1_to_v2.py`](scripts/migrate_v1_to_v2.py)

**Design:**
- Read legacy `inference_runs` via `LEGACY_DATABASE_URL`
- Insert into `RawCall` with `request_json={"prompt": prompt}`, `response_json={"text": response}`, `status="success"`
- Convert `batch_id` into `Group` + `GroupMember` rows
- Keep translation in script only; v2 schema doesn't include legacy fields

**Tests:** [`tests/test_db/test_migration_v1_to_v2.py`](tests/test_db/test_migration_v1_to_v2.py)

---

### Step 7.4: Backfill Validation ✅

**Design:**
- Compare v1 vs v2 row counts
- Verify batch sizes and timestamps match expected ranges
- Sample prompts/responses and metadata parity

**Note:** v1 DB remains unchanged; v2 DB is a fresh Postgres schema

---

### Step 7.5: Embedding Service with Deterministic Caching ✅

**Goal:** Create a first-class embedding service that handles deployment validation, retry/backoff, deterministic caching, and persistence to v2 DB tables. This ensures embedding calls are de-duplicated, transient errors don't crash long runs, and all embeddings are stored with consistent metadata for reuse.

**Dependencies:**
- Phase 7.1 (V2 Schema with `RawCall` and `EmbeddingVector` tables)
- Phase 1 (Provider layer for embedding clients)
- Install: `pip install tenacity` (for retry decorators)

**Files to create:**
- `src/study_query_llm/services/embedding_service.py`

**Files to update:**
- `src/study_query_llm/services/__init__.py` (export `EmbeddingService`)

**Design:**

**Core Classes:**
- `EmbeddingService`: Main service class
- `EmbeddingRequest`: Request parameters dataclass
- `EmbeddingResponse`: Response dataclass (vector, model, dimension, request_hash, cached, raw_call_id)

**Key Methods:**
- `get_embedding()`: Get single embedding with caching
- `get_embeddings_batch()`: Batch embeddings with per-item caching
- `filter_valid_deployments()`: Pre-validate deployment list
- `_check_cache()`: Query DB for existing embedding
- `_compute_request_hash()`: Deterministic hash for cache identity
- `_validate_deployment()`: Probe deployment once, cache result
- `_create_embedding_with_retry()`: API call with retry decorator
- `_log_failure()`: Persist failed requests to RawCall

**Features:**
- **Deterministic caching:** Hash-based cache lookup (model + normalized_text + dimensions + encoding_format + provider)
- **Deployment validation:** One-time probe with cached results per deployment name; validates deployment exists and supports embedding API before use
- **Environment refresh:** Creates fresh `Config()` instance per deployment to ensure environment variable changes (e.g., `AZURE_OPENAI_DEPLOYMENT`) are picked up correctly
- **Retry/backoff:** Exponential backoff (1s → 30s), max 6 attempts, handles `InternalServerError`, `APIConnectionError`, `RateLimitError`, `OperationalError`
- **DB persistence:** Stores to `RawCall` (success/failure) and `EmbeddingVector` (vector data)
- **Failure logging:** Non-retryable errors logged with `status="failed"` and `error_json`
- **Grouping support:** Optional `group_id` in metadata for experiment tracking
- **Provenance:** Stores `request_hash`, `group_id`, custom metadata (seed, library versions)

**Test strategy:** See validation checklist below

**Validation checklist:**
- ✓ Cache hit returns stored vector without provider call
- ✓ Invalid deployment is skipped and logged once (not repeatedly)
- ✓ Retry resolves transient 5xx errors (502, 429, connection errors)
- ✓ Failed calls are persisted with `status="failed"` and `error_json`
- ✓ Deterministic hashing ensures same input produces same hash
- ✓ Text normalization removes null bytes and normalizes whitespace
- ✓ Batch operations work correctly with caching per item

---

### Step 7.6: Panel App Integration (v2 RawCall + Grouping) ✅

**Goal:** Update the Panel UI to read/write the v2 immutable capture schema and expose grouping/batching in the interface.

**Dependencies:**
- Phase 7.1 (V2 Schema)
- Phase 7.2 (Log Success + Failure in RawCall)

**Files to update:**
- `panel_app/app.py`
- `src/study_query_llm/services/inference_service.py` (optional adapter for v2 persistence)
- `src/study_query_llm/services/study_service.py` (v2 analytics entry points)

**Design:**
- Replace v1 `InferenceRepository` usage with v2 `RawCallRepository` (or add a service adapter that abstracts v1/v2).
- Log all inference calls to `RawCall` with `status`, `request_json`, `response_json`, `error_json`.
- Add UI controls for `group_type`, `group_name`, and optional `role/position`.
- Persist group metadata to `Group` and link `RawCall` via `GroupMember`.
- Update analytics widgets to pull from v2 tables (e.g., recent calls, provider stats, time series).

**Test strategy:**
- Run inference in the UI and verify a `RawCall` row is created.
- Create a group in the UI and verify `Group` + `GroupMember` rows.
- Verify analytics panels read from v2 data (counts, recent table, charts).

---

### Step 7.7: Algorithm Core Library (minimal deps) ✅

**Goal:** Extract core algorithm implementations (PCA/KLLMeans sweep, multi-restart clustering, stability metrics) into reusable modules in `src/` with minimal dependencies, separate from notebooks/scripts.

**Dependencies:**
- Phase 7.1 (V2 Schema) - for optional provenance integration
- Install: `pip install numpy scipy` (core algorithm dependencies)

**Files to create:**
- `src/study_query_llm/algorithms/__init__.py`
- `src/study_query_llm/algorithms/clustering.py` (KLLMeans, multi-restart, stability metrics)
- `src/study_query_llm/algorithms/dimensionality_reduction.py` (PCA/SVD projection)
- `src/study_query_llm/algorithms/sweep.py` (sweep orchestration, config dataclasses)

**Files to update:**
- `scripts/pca_kllmeans_sweep.py` (refactor to use new algorithm modules)

**Design:**

**Core Classes:**
- `SweepConfig`: Dataclass for sweep parameters (pca_dim, rank_r, k_min, k_max, max_iter, base_seed)
- `ClusteringResult`: Result dataclass (labels, objective, representatives, metadata)
- `SweepResult`: Aggregated results by K (by_k dict, pca_meta, stability metrics)

**Key Functions:**
- `mean_pool_tokens()`: Pool token embeddings to item-level embeddings
- `pca_svd_project()`: PCA/SVD dimensionality reduction
- `k_subspaces_kllmeans()`: K-subspaces KLLMeans clustering with rank-r approximation
- `run_sweep()`: Orchestrate sweep across K range with optional paraphraser
- `compute_stability_metrics()`: Pairwise ARI, silhouette scores, coverage metrics
- `select_representatives()`: Choose cluster representatives (closest to centroid)

**Features:**
- **Minimal dependencies:** Only numpy/scipy for core algorithms; no DB/LLM dependencies in core
- **Multi-restart support:** Run multiple initializations per K, compute stability metrics
- **Optional paraphraser:** Accept callable for post-processing representatives (LLM summarization)
- **Testable in isolation:** Pure functions with clear inputs/outputs

**Test strategy:**
- Unit tests for each algorithm function with synthetic data
- Verify sweep produces consistent results across runs (with fixed seed)
- Test stability metrics on known cluster structures
- Verify multi-restart produces expected ARI distributions

---

### Step 7.8: Run/Experiment Provenance via Groups ✅

**Goal:** Establish standard conventions for using `Group` and `GroupMember` to track algorithm runs, experiments, and data provenance.

**Dependencies:**
- Phase 7.1 (V2 Schema with Group/GroupMember tables)

**Files to create:**
- `src/study_query_llm/services/provenance_service.py`

**Files to update:**
- `src/study_query_llm/db/models_v2.py` (document group_type conventions in docstrings)

**Design:**

**Standard Group Types:**
- `dataset`: Input data collection (links to embedding RawCalls)
- `embedding_batch`: Batch of embeddings created together
- `run`: Complete algorithm execution (e.g., PCA+KLLMeans sweep)
- `step`: Individual step within a run (e.g., "pca_projection", "clustering_k=5")
- `metrics`: Computed metrics/analysis results
- `summarization_batch`: Batch of LLM summarization calls

**ProvenanceService Methods:**
- `create_run_group()`: Create a run group with metadata (algorithm, config, timestamp)
- `link_raw_calls_to_group()`: Add RawCalls to a group via GroupMember
- `link_artifacts_to_group()`: Create CallArtifact entries and link to group
- `get_run_provenance()`: Query all RawCalls, artifacts, and sub-groups for a run
- `create_step_group()`: Create a step group within a run

**Conventions:**
- Run groups store algorithm config in `metadata_json` (k_range, pca_dim, embedding_model, etc.)
- Step groups reference parent run via `metadata_json.parent_run_id`
- RawCalls linked to groups via `GroupMember` with optional `role` (e.g., "input", "output", "intermediate")
- Artifacts stored as files (JSON/NPY/CSV) with URIs in `CallArtifact`, linked to relevant RawCall or Group

**Test strategy:**
- Create a run group and verify it links to embedding RawCalls
- Verify step groups can reference parent runs
- Query provenance for a run and verify all linked RawCalls/artifacts are returned

---

### Step 7.9: Summarization Service ⬜

**Goal:** Create a service for LLM-based summarization/paraphrasing that logs all calls to RawCall and integrates with grouping for experiment tracking.

**Dependencies:**
- Phase 1 (Provider layer)
- Phase 2 (InferenceService)
- Phase 7.1 (V2 Schema)
- Phase 7.8 (Provenance via Groups)

**Files to create:**
- `src/study_query_llm/services/summarization_service.py`

**Files to update:**
- `src/study_query_llm/services/__init__.py` (export `SummarizationService`)

**Design:**

**Core Classes:**
- `SummarizationService`: Main service class
- `SummarizationRequest`: Request parameters dataclass (texts, llm_deployment, temperature, max_tokens, group_id)
- `SummarizationResponse`: Response dataclass (summaries, raw_call_ids, metadata)

**Key Methods:**
- `summarize_batch()`: Summarize a batch of texts using specified LLM deployment
- `create_paraphraser_for_llm()`: Factory function to create paraphraser callable for a specific deployment
- `_log_summarization_call()`: Persist each summarization call to RawCall with modality="text"

**Features:**
- **LLM deployment selection:** Accept deployment name, create fresh Config() per deployment to pick up env var changes
- **Batch processing:** Process multiple texts concurrently with asyncio
- **RawCall logging:** Every summarization call logged to RawCall (success/failure)
- **Group integration:** Optional `group_id` to link summarization calls to experiment runs
- **Error handling:** Failed calls logged with `status="failed"` and `error_json`
- **Deployment validation:** Optional pre-validation of deployment before batch processing

**Test strategy:**
- Verify summarization calls are logged to RawCall
- Test batch processing with multiple texts
- Verify group linkage works correctly
- Test error handling for invalid deployments

---

### Step 7.10: Analysis Artifacts ⬜

**Goal:** Store algorithm outputs (cluster labels, metrics, PCA components, sweep results) as artifacts linked to runs via CallArtifact and/or Group metadata.

**Dependencies:**
- Phase 7.1 (V2 Schema with CallArtifact)
- Phase 7.7 (Algorithm Core Library)
- Phase 7.8 (Provenance via Groups)

**Files to create:**
- `src/study_query_llm/services/artifact_service.py`

**Files to update:**
- `src/study_query_llm/algorithms/sweep.py` (optional integration with artifact service)

**Design:**

**ArtifactService Methods:**
- `store_sweep_results()`: Save sweep results (by_k dict, metrics) as JSON artifact
- `store_cluster_labels()`: Save cluster labels array as NPY artifact
- `store_pca_components()`: Save PCA components/vectors as NPY artifact
- `store_metrics()`: Save computed metrics (silhouette, ARI, coverage) as JSON artifact
- `link_artifact_to_group()`: Create CallArtifact entry and link to Group via metadata

**Artifact Storage:**
- **File-based:** Store artifacts as files (JSON/NPY/CSV) in configurable directory
- **URI format:** Use relative paths or full URIs in `CallArtifact.uri`
- **Metadata:** Store artifact type, dimensions, and provenance in `CallArtifact.metadata_json`
- **Group linkage:** Link artifacts to run groups via `CallArtifact` entries or `Group.metadata_json.artifacts`

**Conventions:**
- Artifact URIs follow pattern: `artifacts/{run_id}/{step_name}/{artifact_type}.{ext}`
- JSON artifacts for structured data (sweep results, metrics)
- NPY artifacts for numpy arrays (embeddings, cluster labels, PCA components)
- CSV artifacts for tabular data (representatives, metrics tables)

**Test strategy:**
- Store sweep results and verify CallArtifact entry is created
- Verify artifacts can be loaded back from URIs
- Test group linkage and artifact retrieval

---

### Step 7.11: Group-of-Groups / RunStep Schema ⬜

**Goal:** Add a schema table to explicitly model relationships between groups (e.g., run steps, parent-child relationships, execution order).

**Dependencies:**
- Phase 7.1 (V2 Schema)
- Phase 7.8 (Provenance via Groups)

**Files to create:**
- Update `src/study_query_llm/db/models_v2.py` (add `GroupLink` model)
- `src/study_query_llm/db/migrations/add_group_links.py` (migration script)

**Design:**

**New Table: GroupLink**
- `id`: Primary key
- `parent_group_id`: Foreign key to `Group` (parent group)
- `child_group_id`: Foreign key to `Group` (child group)
- `link_type`: Type of relationship ('step', 'contains', 'depends_on', 'generates')
- `position`: Optional ordering within parent (for step sequences)
- `metadata_json`: Additional relationship metadata
- `created_at`: Timestamp

**Use Cases:**
- **Run steps:** Link step groups to parent run group with `link_type='step'` and `position` for ordering
- **Data flow:** Link groups that depend on each other with `link_type='depends_on'`
- **Generation:** Link groups that generate other groups (e.g., embedding batch generates run) with `link_type='generates'`
- **Containment:** Link groups that contain other groups with `link_type='contains'`

**Repository Methods:**
- `create_group_link()`: Create a link between two groups
- `get_group_children()`: Get all child groups for a parent
- `get_group_parents()`: Get all parent groups for a child
- `get_run_step_sequence()`: Get ordered step sequence for a run

**Test strategy:**
- Create run with multiple steps and verify GroupLink entries
- Query step sequence and verify correct ordering
- Test dependency traversal (get all dependencies for a group)

---

## Summary: Implementation Checklist

### Phase 1: Provider Layer
- [x] Step 1.1: Base provider interface
- [x] Step 1.2: Azure provider
- [ ] Step 1.3: OpenAI provider
- [ ] Step 1.4: Hyperbolic provider
- [ ] Step 1.5: Provider factory (partial - Azure only)

### Phase 2: Service Layer
- [x] Step 2.1: Basic inference service
- [x] Step 2.2: Add retry logic
- [x] Step 2.3: Add preprocessing
- [ ] Step 2.4: Conversation service (optional)
- [ ] Step 2.5: Batch service (partial - batching/sampling in InferenceService, no dedup)

### Phase 3: Database Layer
- [x] Step 3.1: Database models
- [x] Step 3.2: Database connection
- [x] Step 3.3: Repository writes
- [x] Step 3.4: Repository queries
- [x] Step 3.5: Service integration

### Phase 4: Analytics Layer
- [x] Step 4.1: Study service

### Phase 5: GUI Layer
- [x] Step 5.1: Inference UI
- [ ] Step 5.2: Analytics dashboard (partial - tables/summary only)
- [ ] Step 5.3: Configuration (partial - config exists, .env.example missing)

### Phase 6: Production
- [ ] Step 6.1: Error handling (partial - logging exists)
- [x] Step 6.2: Unit tests
- [x] Step 6.3: Documentation
- [x] Step 6.4: Docker setup

### Phase 7: Immutable Capture + Grouping DB (Postgres)
- [x] Step 7.1: V2 Schema (Immutable Raw Calls)
- [ ] Step 7.2: Log Success + Failure in RawCall
- [x] Step 7.3: Migration Script (v1 → v2)
- [ ] Step 7.4: Backfill Validation
- [ ] Step 7.5: Embedding Service with Deterministic Caching
- [ ] Step 7.6: Panel App Integration (v2 RawCall + Grouping)
- [x] Step 7.7: Algorithm Core Library (minimal deps)
- [ ] Step 7.8: Run/Experiment Provenance via Groups
- [ ] Step 7.9: Summarization Service
- [ ] Step 7.10: Analysis Artifacts
- [ ] Step 7.11: Group-of-Groups / RunStep Schema

---

## Next Steps
- Implement OpenAI + Hyperbolic providers and expand `ProviderFactory`
- Add conversation service if needed
- Add request dedup batching service (or extend `InferenceService`)
- Finish analytics UI charts/time-series + search
- Integrate v2 schema into InferenceService to log failures (Phase 7.2)
- Complete backfill validation for v1→v2 migration (Phase 7.4)
- Implement Embedding Service with deterministic caching (Phase 7.5)
- Extract algorithm core library from scripts/notebooks (Phase 7.7)
- Implement provenance service and group conventions (Phase 7.8)
- Implement summarization service with RawCall logging (Phase 7.9)
- Add artifact storage service for algorithm outputs (Phase 7.10)
- Add GroupLink schema for explicit group relationships (Phase 7.11)