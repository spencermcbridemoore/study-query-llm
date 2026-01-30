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

**Files to create:**
- `panel_app/services/study_service.py`

**Dependencies:**
- Install: `pip install pandas`

**What to build:**

```python
# panel_app/services/study_service.py

import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta
from panel_app.db.inference_repository import InferenceRepository

class StudyService:
    """
    Service for analyzing and studying stored inference data.
    Provides business logic for analytics queries.
    """

    def __init__(self, repository: InferenceRepository):
        self.repository = repository

    def get_provider_comparison(self) -> pd.DataFrame:
        """
        Compare providers by various metrics.

        Returns:
            DataFrame with provider comparison
        """
        stats = self.repository.get_provider_stats()
        df = pd.DataFrame(stats)

        if not df.empty:
            # Add derived metrics
            df['avg_cost_estimate'] = df['avg_tokens'] * 0.00002  # Example pricing

        return df

    def get_recent_inferences(
        self,
        limit: int = 50,
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get recent inference runs.

        Returns:
            DataFrame with recent inferences
        """
        runs = self.repository.query_inferences(
            provider=provider,
            limit=limit
        )

        data = [
            {
                'id': run.id,
                'prompt': run.prompt[:100] + '...' if len(run.prompt) > 100 else run.prompt,
                'response': run.response[:100] + '...' if len(run.response) > 100 else run.response,
                'provider': run.provider,
                'tokens': run.tokens,
                'latency_ms': run.latency_ms,
                'created_at': run.created_at
            }
            for run in runs
        ]

        return pd.DataFrame(data)

    def get_time_series_data(
        self,
        days: int = 7,
        group_by: str = 'day'
    ) -> pd.DataFrame:
        """
        Get time-series data for visualization.

        Args:
            days: Number of days to look back
            group_by: 'day', 'hour', or 'minute'

        Returns:
            DataFrame with time-series data
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        runs = self.repository.query_inferences(
            date_range=(start_date, end_date),
            limit=10000  # Large limit for aggregation
        )

        if not runs:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                'created_at': run.created_at,
                'provider': run.provider,
                'tokens': run.tokens,
                'latency_ms': run.latency_ms
            }
            for run in runs
        ])

        df['created_at'] = pd.to_datetime(df['created_at'])

        # Group by time period
        freq_map = {'day': 'D', 'hour': 'H', 'minute': 'T'}
        freq = freq_map.get(group_by, 'D')

        grouped = df.groupby([pd.Grouper(key='created_at', freq=freq), 'provider']).agg({
            'tokens': ['sum', 'mean'],
            'latency_ms': 'mean'
        }).reset_index()

        return grouped

    def search_prompts(self, search_term: str) -> pd.DataFrame:
        """
        Search historical prompts.

        Returns:
            DataFrame with matching prompts
        """
        runs = self.repository.search_by_prompt(search_term, limit=100)

        data = [
            {
                'id': run.id,
                'prompt': run.prompt,
                'response': run.response[:200] + '...' if len(run.response) > 200 else run.response,
                'provider': run.provider,
                'created_at': run.created_at
            }
            for run in runs
        ]

        return pd.DataFrame(data)

    def get_summary_stats(self) -> dict:
        """
        Get overall summary statistics.

        Returns:
            Dict with summary stats
        """
        total = self.repository.get_total_count()
        provider_stats = self.repository.get_provider_stats()

        total_tokens = sum(s['total_tokens'] for s in provider_stats)

        return {
            'total_inferences': total,
            'total_tokens': total_tokens,
            'unique_providers': len(provider_stats),
            'provider_breakdown': provider_stats
        }
```

**Test:**
```python
# test_study_service.py
from panel_app.db.connection import DatabaseConnection
from panel_app.db.inference_repository import InferenceRepository
from panel_app.services.study_service import StudyService

db = DatabaseConnection("sqlite:///test.db")

with db.session_scope() as session:
    repo = InferenceRepository(session)
    study = StudyService(repo)

    # Get provider comparison
    comparison = study.get_provider_comparison()
    print(comparison)

    # Get recent inferences
    recent = study.get_recent_inferences(limit=10)
    print(recent)

    # Get summary
    summary = study.get_summary_stats()
    print(summary)
```

**Validation:** ✓ Analytics queries return useful data

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

**Update:** `panel_app/app.py`

**Dependencies:**
- Install: `pip install hvplot holoviews bokeh`

**What to build:**

See the detailed GUI implementation in ARCHITECTURE.md under "Presentation Layer"

Key components:
1. Provider selection dropdown
2. Prompt input text area
3. Run inference button
4. Response display
5. Metadata display (tokens, latency)

**Test:** Run inference through GUI, verify it appears in database

---

### Step 5.2: Analytics Dashboard ⚠️ (tables/summary only)

**What to add:**
1. Provider comparison bar chart
2. Time-series line chart
3. Recent inferences table
4. Summary statistics cards
5. Search interface

**Test:** Verify charts update with real data

---

### Step 5.3: Configuration Management ⚠️ (config exists; `.env.example` missing)

**Files to create:**
- `.env.example` (missing)
- `src/study_query_llm/config.py` (already implemented)

**What to build:**

```python
# panel_app/config.py

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider"""
    name: str
    api_key: str
    endpoint: Optional[str] = None
    model: Optional[str] = None
    deployment_name: Optional[str] = None

@dataclass
class DatabaseConfig:
    """Database configuration"""
    connection_string: str

class AppConfig:
    """Application configuration"""

    def __init__(self):
        self.database = DatabaseConfig(
            connection_string=os.getenv(
                "DATABASE_URL",
                "sqlite:///study_query_llm.db"
            )
        )

        self.providers = {
            'azure': ProviderConfig(
                name='azure',
                api_key=os.getenv('AZURE_API_KEY', ''),
                endpoint=os.getenv('AZURE_ENDPOINT', ''),
                deployment_name=os.getenv('AZURE_DEPLOYMENT', 'gpt-4')
            ),
            'openai': ProviderConfig(
                name='openai',
                api_key=os.getenv('OPENAI_API_KEY', ''),
                model=os.getenv('OPENAI_MODEL', 'gpt-4')
            ),
            'hyperbolic': ProviderConfig(
                name='hyperbolic',
                api_key=os.getenv('HYPERBOLIC_API_KEY', ''),
                endpoint=os.getenv('HYPERBOLIC_ENDPOINT', '')
            )
        }

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get configuration for specific provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers[provider_name]
```

**Create:** `.env.example`

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/study_query_llm

# Azure OpenAI
AZURE_API_KEY=your-azure-key
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_DEPLOYMENT=gpt-4

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4

# Hyperbolic
HYPERBOLIC_API_KEY=your-hyperbolic-key
HYPERBOLIC_ENDPOINT=https://api.hyperbolic.xyz
```

**Test:** Load configuration from environment

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

Add comprehensive logging throughout:
- Provider API calls
- Database operations
- User actions

### Step 6.2: Unit Tests ✅

Create test suite:
- `tests/test_providers.py`
- `tests/test_services.py`
- `tests/test_repository.py`

### Step 6.3: Documentation ✅

Update:
- README.md with full setup instructions
- API documentation
- User guide

### Step 6.4: Docker Setup ✅

**Runtime targets**
- Python 3.11 slim image with `panel_app.app` exposed on port `5006`
- Default SQLite database persisted via container volume (`study_query_llm.db`)
- Optional Postgres connection through `DATABASE_URL`

**Required env vars**
- Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`
- Optional providers: `OPENAI_API_KEY`, `OPENAI_MODEL`, `HYPERBOLIC_API_KEY`, `HYPERBOLIC_ENDPOINT`
- App: `PANEL_ADDRESS` (default `0.0.0.0`), `PANEL_PORT` (default `5006`), `DATABASE_URL`

**Deliverables**
- Multi-stage `Dockerfile` (build dependencies, runtime, non-root user)
- `docker-compose.yml` with app service, optional Postgres profile, healthcheck
- Deployment guide covering `docker build`, `docker compose up`, secrets, and volume usage

---

## Phase 7: Immutable Capture + Grouping DB (Postgres)

**Goal:** Add a v2 immutable raw capture schema that logs successes + failures across modalities, plus mutable grouping tables for experiments/batches/labels.

**Dependencies:** Phase 3 (DB layer), PostgreSQL target

### Step 7.1: V2 Schema (Immutable Raw Calls) ✅

**Files created:**
- `src/study_query_llm/db/models_v2.py` - V2 models (RawCall, Group, GroupMember, CallArtifact, EmbeddingVector)
- `src/study_query_llm/db/raw_call_repository.py` - Repository for v2 operations
- `src/study_query_llm/db/connection_v2.py` - Connection helper for v2 Postgres schema

**Core tables:**
- `RawCall`: `provider`, `model`, `modality`, `status`, `request_json`, `response_json`, `error_json`, `latency_ms`, `tokens_json`, `metadata_json`, `created_at`
- `CallArtifact`: blob references for multimodal payloads (`uri`, `content_type`, `byte_size`, `metadata_json`)
- `EmbeddingVector`: optional table for embeddings (`vector`, `dimension`, `norm`, `metadata_json`) with pgvector support
- `Group`: mutable grouping metadata (`group_type`, `name`, `description`, `created_at`, `metadata_json`)
- `GroupMember`: join table (`group_id`, `call_id`, `added_at`, `position`, `role`)

**Test:** ✅ Created tables + insert/fetch for each table (`tests/test_db/test_models_v2.py`, `tests/test_db/test_repository_v2.py`)

### Step 7.2: Log Success + Failure in RawCall ⬜

**Update:** `src/study_query_llm/services/inference_service.py`

- On success: `status="success"`, `response_json` set
- On failure/exception: `status="failed"`, `response_json=null`, `error_json` set
- Always record `request_json` + runtime metadata (tokens, latency, provider)

**Test:** Verify failed calls still persist in v2 DB

### Step 7.3: Migration Script (v1 → v2) ✅

**File created:** `scripts/migrate_v1_to_v2.py`

- Read legacy `inference_runs` via `LEGACY_DATABASE_URL`
- Insert into `RawCall` with:
  - `request_json={"prompt": prompt}`
  - `response_json={"text": response}`
  - `status="success"`
- Convert `batch_id` into `Group` + `GroupMember` rows
- Do not add legacy fields to v2 schema; keep translation in the script only

**Test:** ✅ Row counts + sample spot checks (`tests/test_db/test_migration_v1_to_v2.py`)

### Step 7.4: Backfill Validation ⬜

- Compare v1 vs v2 counts
- Verify batch sizes and timestamps match expected ranges
- Sample prompts/responses and metadata parity

**Design note:** v1 DB remains unchanged; v2 DB is a fresh Postgres schema so the new design is not constrained by legacy structure.

---

### Step 7.5: Embedding Service with Deterministic Caching ⬜

**Goal:** Create a first-class embedding service that handles deployment validation, retry/backoff, deterministic caching, and persistence to v2 DB tables. This ensures embedding calls are de-duplicated, transient errors don't crash long runs, and all embeddings are stored with consistent metadata for reuse.

**Dependencies:**
- Phase 7.1 (V2 Schema with `RawCall` and `EmbeddingVector` tables)
- Phase 1 (Provider layer for embedding clients)
- Install: `pip install tenacity` (for retry decorators)

**Files to create:**
- `src/study_query_llm/services/embedding_service.py`

**Files to update:**
- `src/study_query_llm/services/__init__.py` (export `EmbeddingService`)

**What to build:**

```python
# src/study_query_llm/services/embedding_service.py

import hashlib
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import (
    AzureOpenAI,
    OpenAI,
    BadRequestError,
    InternalServerError,
    APIConnectionError,
    RateLimitError,
    NotFoundError
)
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from study_query_llm.db.models_v2 import RawCall, EmbeddingVector
from study_query_llm.db.connection_v2 import DatabaseConnectionV2


@dataclass
class EmbeddingRequest:
    """Request parameters for embedding generation"""
    text: str
    model: str
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    provider: str = "azure_openai"


@dataclass
class EmbeddingResponse:
    """Response from embedding service"""
    vector: np.ndarray
    model: str
    dimension: int
    request_hash: str
    cached: bool
    raw_call_id: Optional[int] = None


class EmbeddingService:
    """
    Service for generating and caching embeddings with deterministic de-duplication.
    
    Features:
    - Deterministic request hashing for cache hits
    - Deployment validation (probe once, filter invalid deployments)
    - Retry/backoff for transient API errors (502/429/timeout)
    - DB persistence to v2 tables (RawCall + EmbeddingVector)
    - Failure logging with error_json
    - Optional grouping support for sweep runs
    """

    def __init__(
        self,
        db: DatabaseConnectionV2,
        embedding_client: Any,  # AzureOpenAI or OpenAI
        provider_name: str = "azure_openai",
        max_retries: int = 6,
        initial_wait: float = 1.0,
        validate_deployments: bool = True
    ):
        """
        Initialize embedding service.

        Args:
            db: DatabaseConnectionV2 instance
            embedding_client: AzureOpenAI or OpenAI client instance
            provider_name: Provider identifier ("azure_openai", "openai", etc.)
            max_retries: Maximum retry attempts for transient errors
            initial_wait: Initial wait time in seconds for exponential backoff
            validate_deployments: If True, validate deployments before use
        """
        self.db = db
        self.client = embedding_client
        self.provider_name = provider_name
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.validate_deployments = validate_deployments
        self._validated_deployments: Dict[str, bool] = {}

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing"""
        # Remove null bytes, strip whitespace
        return text.replace("\x00", "").strip()

    def _compute_request_hash(
        self,
        text: str,
        model: str,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        provider: str = None
    ) -> str:
        """
        Compute deterministic hash for request identity.

        Hash includes: model + normalized_text + dimensions + encoding_format + provider
        """
        provider = provider or self.provider_name
        normalized = self._normalize_text(text)
        content = f"{model}:{normalized}:{dimensions}:{encoding_format}:{provider}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _check_cache(
        self,
        session: Session,
        text: str,
        model: str
    ) -> Optional[EmbeddingVector]:
        """
        Check if embedding exists in cache (DB) for this text and model.

        Returns:
            EmbeddingVector if found, None otherwise
        """
        normalized_text = self._normalize_text(text)
        existing = (
            session.query(EmbeddingVector, RawCall)
            .join(RawCall, EmbeddingVector.call_id == RawCall.id)
            .filter(
                RawCall.modality == "embedding",
                RawCall.model == model,
                RawCall.status == "success",
                RawCall.request_json["input"].as_string() == normalized_text
            )
            .order_by(RawCall.id.desc())
            .first()
        )

        if existing:
            vector, _ = existing
            return vector
        return None

    def _validate_deployment(self, deployment: str) -> bool:
        """
        Validate that a deployment exists and is accessible.

        Returns:
            True if valid, False otherwise
        """
        if deployment in self._validated_deployments:
            return self._validated_deployments[deployment]

        try:
            # Cheap probe with minimal input
            self.client.embeddings.create(model=deployment, input=["ping"])
            self._validated_deployments[deployment] = True
            return True
        except NotFoundError:
            self._validated_deployments[deployment] = False
            return False
        except Exception:
            # Other errors might be transient; don't cache as invalid
            return False

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((
            InternalServerError,
            APIConnectionError,
            RateLimitError,
            OperationalError
        ))
    )
    def _create_embedding_with_retry(
        self,
        model: str,
        text: str,
        dimensions: Optional[int] = None,
        encoding_format: str = "float"
    ) -> Any:
        """
        Create embedding with retry logic for transient errors.

        Handles:
        - 502 Bad Gateway (InternalServerError)
        - Connection errors (APIConnectionError)
        - Rate limits (RateLimitError)
        - DB connection drops (OperationalError)
        """
        params = {
            "model": model,
            "input": [text],
        }
        if dimensions:
            params["dimensions"] = dimensions
        if encoding_format:
            params["encoding_format"] = encoding_format

        return self.client.embeddings.create(**params)

    def get_embedding(
        self,
        text: str,
        model: str,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        group_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResponse:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Input text to embed
            model: Embedding model/deployment name
            dimensions: Optional dimension override
            encoding_format: Encoding format ("float" or "base64")
            group_id: Optional group ID for sweep runs
            metadata: Optional metadata to store (e.g., seed, library versions)

        Returns:
            EmbeddingResponse with vector and metadata

        Raises:
            BadRequestError: If input is invalid (not retried)
            NotFoundError: If deployment doesn't exist (not retried)
        """
        # Normalize and validate input
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise ValueError("Input text cannot be empty after normalization")

        # Validate deployment if enabled
        if self.validate_deployments and not self._validate_deployment(model):
            raise NotFoundError(f"Deployment '{model}' not found or inaccessible")

        # Compute request hash for cache lookup
        request_hash = self._compute_request_hash(
            normalized_text, model, dimensions, encoding_format
        )

        # Check cache
        while True:
            try:
                with self.db.session_scope() as session:
                    cached = self._check_cache(session, normalized_text, model)
                    if cached:
                        return EmbeddingResponse(
                            vector=np.array(cached.vector, dtype=np.float64),
                            model=model,
                            dimension=cached.dimension,
                            request_hash=request_hash,
                            cached=True,
                            raw_call_id=cached.call_id
                        )

                    # Not in cache; create new embedding
                    start_time = time.perf_counter()
                    response = self._create_embedding_with_retry(
                        model, normalized_text, dimensions, encoding_format
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                    embedding = response.data[0].embedding
                    usage = getattr(response, "usage", None)

                    # Store RawCall
                    raw_call = RawCall(
                        provider=self.provider_name,
                        model=model,
                        modality="embedding",
                        status="success",
                        request_json={
                            "input": normalized_text,
                            "dimensions": dimensions,
                            "encoding_format": encoding_format
                        },
                        response_json={
                            "model": model,
                            "embedding_dim": len(embedding),
                            "object": "list"
                        },
                        latency_ms=elapsed_ms,
                        tokens_json={"total": usage.total_tokens} if usage else None,
                        metadata_json={
                            "request_hash": request_hash,
                            "group_id": group_id,
                            **(metadata or {})
                        }
                    )
                    session.add(raw_call)
                    session.flush()
                    session.refresh(raw_call)

                    # Store EmbeddingVector
                    vector_row = EmbeddingVector(
                        call_id=raw_call.id,
                        vector=embedding,
                        dimension=len(embedding),
                        norm=np.linalg.norm(embedding),
                        metadata_json={"model": model}
                    )
                    session.add(vector_row)

                    return EmbeddingResponse(
                        vector=np.array(embedding, dtype=np.float64),
                        model=model,
                        dimension=len(embedding),
                        request_hash=request_hash,
                        cached=False,
                        raw_call_id=raw_call.id
                    )

            except OperationalError:
                # DB connection dropped; recreate and retry
                self.db = DatabaseConnectionV2(
                    self.db.connection_string,
                    enable_pgvector=True
                )
                self.db.init_db()
                time.sleep(1)
            except (BadRequestError, NotFoundError):
                # Non-retryable errors; log failure and re-raise
                self._log_failure(normalized_text, model, dimensions, encoding_format)
                raise

    def _log_failure(
        self,
        text: str,
        model: str,
        dimensions: Optional[int],
        encoding_format: str
    ):
        """Log failed embedding request to RawCall with error_json"""
        try:
            with self.db.session_scope() as session:
                raw_call = RawCall(
                    provider=self.provider_name,
                    model=model,
                    modality="embedding",
                    status="failed",
                    request_json={
                        "input": text,
                        "dimensions": dimensions,
                        "encoding_format": encoding_format
                    },
                    response_json=None,
                    error_json={"error_type": "BadRequestError or NotFoundError"},
                    metadata_json={}
                )
                session.add(raw_call)
        except Exception:
            # Don't fail if logging fails
            pass

    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        group_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EmbeddingResponse]:
        """
        Get embeddings for multiple texts, with caching per item.

        Args:
            texts: List of input texts
            model: Embedding model/deployment name
            dimensions: Optional dimension override
            encoding_format: Encoding format
            group_id: Optional group ID for sweep runs
            metadata: Optional metadata to store

        Returns:
            List of EmbeddingResponse objects
        """
        return [
            self.get_embedding(
                text, model, dimensions, encoding_format, group_id, metadata
            )
            for text in texts
        ]

    def filter_valid_deployments(self, deployments: List[str]) -> List[str]:
        """
        Filter list of deployments to only those that exist and are accessible.

        Args:
            deployments: List of deployment names to validate

        Returns:
            List of valid deployment names
        """
        if not self.validate_deployments:
            return deployments

        valid = []
        for dep in deployments:
            if self._validate_deployment(dep):
                valid.append(dep)
            else:
                print(f"Skipping missing deployment: {dep}")
        return valid
```

**Update:** `src/study_query_llm/services/__init__.py`

```python
# Add to existing exports
from .embedding_service import EmbeddingService, EmbeddingRequest, EmbeddingResponse

__all__ = [
    # ... existing exports ...
    "EmbeddingService",
    "EmbeddingRequest",
    "EmbeddingResponse",
]
```

**Test:**

```python
# tests/test_services/test_embedding_service.py

import pytest
import numpy as np
from unittest.mock import Mock, patch
from openai import NotFoundError, InternalServerError
from study_query_llm.services.embedding_service import EmbeddingService
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import RawCall, EmbeddingVector


def test_cache_hit_returns_without_provider_call():
    """Test that cache hit returns stored vector without calling provider"""
    # Setup
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    
    mock_client = Mock()
    service = EmbeddingService(db, mock_client, validate_deployments=False)
    
    text = "test prompt"
    model = "text-embedding-3-small"
    
    # First call - should hit provider
    with patch.object(service, '_create_embedding_with_retry') as mock_create:
        mock_create.return_value.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        response1 = service.get_embedding(text, model)
        assert not response1.cached
        assert mock_create.called
    
    # Second call - should use cache
    mock_create.reset_mock()
    response2 = service.get_embedding(text, model)
    assert response2.cached
    assert not mock_create.called
    assert np.array_equal(response1.vector, response2.vector)


def test_invalid_deployment_skipped_and_logged_once():
    """Test that invalid deployment is skipped and logged once"""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    
    mock_client = Mock()
    mock_client.embeddings.create.side_effect = NotFoundError(
        "DeploymentNotFound", response=Mock(), body={}
    )
    
    service = EmbeddingService(db, mock_client, validate_deployments=True)
    
    # First validation attempt
    valid = service.filter_valid_deployments(["invalid-deployment"])
    assert len(valid) == 0
    assert mock_client.embeddings.create.call_count == 1
    
    # Second validation attempt - should use cache
    mock_client.embeddings.create.reset_mock()
    valid = service.filter_valid_deployments(["invalid-deployment"])
    assert len(valid) == 0
    assert mock_client.embeddings.create.call_count == 0  # Cached


def test_retry_resolves_transient_5xx_errors():
    """Test that retry resolves transient 5xx errors"""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    
    mock_client = Mock()
    service = EmbeddingService(db, mock_client, validate_deployments=False)
    
    # First two calls fail, third succeeds
    mock_client.embeddings.create.side_effect = [
        InternalServerError("502 Bad Gateway", response=Mock(), body={}),
        InternalServerError("502 Bad Gateway", response=Mock(), body={}),
        Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
    ]
    
    response = service.get_embedding("test", "text-embedding-3-small")
    assert response.vector is not None
    assert mock_client.embeddings.create.call_count == 3


def test_failed_calls_persisted_with_status_failed():
    """Test that failed calls are persisted with status='failed'"""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    
    mock_client = Mock()
    mock_client.embeddings.create.side_effect = BadRequestError(
        "Invalid input", response=Mock(), body={}
    )
    
    service = EmbeddingService(db, mock_client, validate_deployments=False)
    
    with pytest.raises(BadRequestError):
        service.get_embedding("invalid input with null\x00", "text-embedding-3-small")
    
    # Check that failure was logged
    with db.session_scope() as session:
        failed_call = session.query(RawCall).filter_by(
            status="failed",
            modality="embedding"
        ).first()
        assert failed_call is not None
        assert failed_call.error_json is not None
        assert failed_call.response_json is None


def test_deterministic_hashing_same_input_same_hash():
    """Test that same input produces same hash"""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    
    mock_client = Mock()
    service = EmbeddingService(db, mock_client, validate_deployments=False)
    
    text = "same text"
    model = "text-embedding-3-small"
    
    hash1 = service._compute_request_hash(text, model)
    hash2 = service._compute_request_hash(text, model)
    
    assert hash1 == hash2


def test_normalization_removes_null_bytes():
    """Test that text normalization removes null bytes"""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    
    mock_client = Mock()
    service = EmbeddingService(db, mock_client, validate_deployments=False)
    
    text_with_null = "text\x00with\x00nulls"
    normalized = service._normalize_text(text_with_null)
    
    assert "\x00" not in normalized
    assert normalized == "textwithnulls"
```

**Validation:**
- ✓ Cache hit returns stored vector without provider call
- ✓ Invalid deployment is skipped and logged once (not repeatedly)
- ✓ Retry resolves transient 5xx errors (502, 429, connection errors)
- ✓ Failed calls are persisted with `status="failed"` and `error_json`
- ✓ Deterministic hashing ensures same input produces same hash
- ✓ Text normalization removes null bytes and normalizes whitespace
- ✓ Batch operations work correctly with caching per item

**Design notes:**
- **Request identity:** Hash includes `model + normalized_text + dimensions + encoding_format + provider` for deterministic cache hits
- **Deterministic caching:** If hash exists in DB, return stored vector and skip API call entirely
- **Retry policy:** Exponential backoff (1s → 30s max), max 6 attempts, retries on `InternalServerError`, `APIConnectionError`, `RateLimitError`, `OperationalError`
- **Deployment validation:** Preflight `embeddings.create` with minimal "ping" input; cache validation results to avoid repeated probes
- **Failure logging:** On non-retryable errors (`BadRequestError`, `NotFoundError`), store failed `RawCall` with `status="failed"` and `error_json`
- **Result grouping:** Optional `group_id` parameter stored in `metadata_json` to support sweep runs and experiment tracking
- **Metadata/provenance:** Store `request_hash`, `group_id`, and custom metadata (seed, library versions, etc.) in `RawCall.metadata_json`

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

---

## Next Steps
- Implement OpenAI + Hyperbolic providers and expand `ProviderFactory`
- Add conversation service if needed
- Add request dedup batching service (or extend `InferenceService`)
- Finish analytics UI charts/time-series + search
- Integrate v2 schema into InferenceService to log failures (Phase 7.2)
- Complete backfill validation for v1→v2 migration (Phase 7.4)
- Implement Embedding Service with deterministic caching (Phase 7.5)
