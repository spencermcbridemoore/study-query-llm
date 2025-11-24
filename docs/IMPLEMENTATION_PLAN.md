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

---

## Phase 1: LLM Provider Abstraction Layer
**Goal:** Build the foundational inference interface - no database, no GUI

**Dependencies:** None (this is the foundation)

### Step 1.1: Base Provider Interface ⬜

**Files to create:**
- `panel_app/providers/__init__.py`
- `panel_app/providers/base.py`

**What to build:**

```python
# panel_app/providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ProviderResponse:
    """Standardized response from any LLM provider"""
    text: str
    provider: str
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = None
    raw_response: Any = None

class BaseLLMProvider(ABC):
    """Abstract base class all providers must implement"""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Send a completion request to the LLM provider.

        Args:
            prompt: The input prompt
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResponse with standardized fields
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider (e.g., 'azure', 'openai')"""
        pass
```

**Test:**
```python
# test_base_provider.py (manual test)
from panel_app.providers.base import ProviderResponse, BaseLLMProvider

# Create a mock provider for testing
class MockProvider(BaseLLMProvider):
    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        return ProviderResponse(
            text=f"Mock response to: {prompt}",
            provider="mock",
            tokens=10,
            latency_ms=50.0
        )

    def get_provider_name(self) -> str:
        return "mock"

# Test it
import asyncio

async def test():
    provider = MockProvider()
    response = await provider.complete("Hello!")
    print(f"Provider: {response.provider}")
    print(f"Response: {response.text}")
    print(f"Tokens: {response.tokens}")

asyncio.run(test())
```

**Validation:** ✓ Mock provider returns standardized response

---

### Step 1.2: Azure Provider Implementation ⬜

**Files to create:**
- `panel_app/providers/azure_provider.py`

**Dependencies:**
- Install: `pip install openai` (Azure uses OpenAI SDK)

**What to build:**

```python
# panel_app/providers/azure_provider.py

import time
from openai import AsyncAzureOpenAI
from .base import BaseLLMProvider, ProviderResponse

class AzureProvider(BaseLLMProvider):
    """Azure OpenAI provider implementation"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview"
    ):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Send completion request to Azure OpenAI"""
        start_time = time.time()

        # Default parameters
        params = {
            "model": self.deployment_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        # Make API call
        response = await self.client.chat.completions.create(**params)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return ProviderResponse(
            text=response.choices[0].message.content,
            provider="azure",
            tokens=response.usage.total_tokens,
            latency_ms=latency_ms,
            metadata={
                "model": self.deployment_name,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response
        )

    def get_provider_name(self) -> str:
        return "azure"
```

**Test:**
```python
# test_azure.py (manual test - requires API key)
import asyncio
from panel_app.providers.azure_provider import AzureProvider

async def test_azure():
    provider = AzureProvider(
        api_key="YOUR_API_KEY",
        endpoint="https://YOUR_ENDPOINT.openai.azure.com/",
        deployment_name="gpt-4"
    )

    response = await provider.complete("Say hello in 5 words")
    print(f"Response: {response.text}")
    print(f"Tokens: {response.tokens}")
    print(f"Latency: {response.latency_ms}ms")

asyncio.run(test_azure())
```

**Validation:** ✓ Azure API returns real response with tokens and latency

---

### Step 1.3: OpenAI Provider Implementation ⬜

**Files to create:**
- `panel_app/providers/openai_provider.py`

**What to build:**

```python
# panel_app/providers/openai_provider.py

import time
from openai import AsyncOpenAI
from .base import BaseLLMProvider, ProviderResponse

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Send completion request to OpenAI"""
        start_time = time.time()

        params = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        response = await self.client.chat.completions.create(**params)
        latency_ms = (time.time() - start_time) * 1000

        return ProviderResponse(
            text=response.choices[0].message.content,
            provider="openai",
            tokens=response.usage.total_tokens,
            latency_ms=latency_ms,
            metadata={
                "model": params["model"],
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw_response=response
        )

    def get_provider_name(self) -> str:
        return "openai"
```

**Test:** Similar to Azure test above

**Validation:** ✓ OpenAI API returns real response

---

### Step 1.4: Hyperbolic Provider Implementation ⬜

**Files to create:**
- `panel_app/providers/hyperbolic_provider.py`

**Note:** Implementation depends on Hyperbolic's API structure. Adjust as needed.

**What to build:**

```python
# panel_app/providers/hyperbolic_provider.py

import time
import httpx
from .base import BaseLLMProvider, ProviderResponse

class HyperbolicProvider(BaseLLMProvider):
    """Hyperbolic API provider implementation"""

    def __init__(self, api_key: str, base_url: str = "https://api.hyperbolic.xyz"):
        self.api_key = api_key
        self.base_url = base_url

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Send completion request to Hyperbolic"""
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                }
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.time() - start_time) * 1000

        return ProviderResponse(
            text=data["choices"][0]["text"],
            provider="hyperbolic",
            tokens=data.get("usage", {}).get("total_tokens"),
            latency_ms=latency_ms,
            metadata=data.get("usage", {}),
            raw_response=data
        )

    def get_provider_name(self) -> str:
        return "hyperbolic"
```

**Test:** Adjust based on Hyperbolic API documentation

**Validation:** ✓ Hyperbolic API returns response

---

### Step 1.5: Provider Factory ⬜

**Files to create:**
- `panel_app/providers/factory.py`

**What to build:**

```python
# panel_app/providers/factory.py

from typing import Optional
from .base import BaseLLMProvider
from .azure_provider import AzureProvider
from .openai_provider import OpenAIProvider
from .hyperbolic_provider import HyperbolicProvider

class ProviderFactory:
    """Factory for creating LLM provider instances"""

    @staticmethod
    def create(provider_name: str, **config) -> BaseLLMProvider:
        """
        Create a provider instance by name.

        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')
            **config: Provider-specific configuration

        Returns:
            BaseLLMProvider instance

        Raises:
            ValueError: If provider_name is unknown
        """
        provider_name = provider_name.lower()

        if provider_name == "azure":
            return AzureProvider(**config)
        elif provider_name == "openai":
            return OpenAIProvider(**config)
        elif provider_name == "hyperbolic":
            return HyperbolicProvider(**config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    @staticmethod
    def get_available_providers() -> list[str]:
        """Return list of supported provider names"""
        return ["azure", "openai", "hyperbolic"]
```

**Update:** `panel_app/providers/__init__.py`

```python
# panel_app/providers/__init__.py

from .base import BaseLLMProvider, ProviderResponse
from .factory import ProviderFactory
from .azure_provider import AzureProvider
from .openai_provider import OpenAIProvider
from .hyperbolic_provider import HyperbolicProvider

__all__ = [
    "BaseLLMProvider",
    "ProviderResponse",
    "ProviderFactory",
    "AzureProvider",
    "OpenAIProvider",
    "HyperbolicProvider",
]
```

**Test:**
```python
# test_factory.py
from panel_app.providers import ProviderFactory

async def test_factory():
    # Create Azure provider via factory
    azure = ProviderFactory.create(
        "azure",
        api_key="...",
        endpoint="...",
        deployment_name="gpt-4"
    )

    response = await azure.complete("Hello!")
    print(f"Factory created {response.provider} provider")

    # List available providers
    print(f"Available: {ProviderFactory.get_available_providers()}")

import asyncio
asyncio.run(test_factory())
```

**Validation:** ✓ Factory creates correct provider instances

---

### Phase 1 Milestone ✓

**What you have now:**
- Abstract interface for all LLM providers
- Azure, OpenAI, and Hyperbolic implementations
- Factory for easy provider creation
- Standardized response format
- Fully testable without database or GUI

**Next:** Add business logic layer

---

## Phase 2: Business Logic Layer (Service Layer)
**Goal:** Add retry, preprocessing, and orchestration logic

**Dependencies:** Phase 1 (Provider layer)

### Step 2.1: Basic Inference Service ⬜

**Files to create:**
- `panel_app/services/__init__.py`
- `panel_app/services/inference_service.py`

**What to build:**

```python
# panel_app/services/inference_service.py

from typing import Optional
from panel_app.providers.base import BaseLLMProvider, ProviderResponse

class InferenceService:
    """
    Core service for running LLM inferences.
    Handles business logic like retry, preprocessing, logging.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        repository=None  # Optional - we'll add this in Phase 3
    ):
        self.provider = provider
        self.repository = repository

    async def run_inference(self, prompt: str, **kwargs) -> dict:
        """
        Run a single inference through the provider.

        Args:
            prompt: User prompt
            **kwargs: Provider-specific parameters

        Returns:
            Dict with response and metadata
        """
        # For now, just pass through to provider
        response = await self.provider.complete(prompt, **kwargs)

        return {
            'response': response.text,
            'metadata': {
                'provider': response.provider,
                'tokens': response.tokens,
                'latency_ms': response.latency_ms,
            }
        }
```

**Test:**
```python
# test_inference_service.py
from panel_app.providers import ProviderFactory
from panel_app.services.inference_service import InferenceService

async def test_service():
    provider = ProviderFactory.create(
        "openai",
        api_key="YOUR_KEY",
        model="gpt-4"
    )

    service = InferenceService(provider)
    result = await service.run_inference("Count to 5")

    print(f"Response: {result['response']}")
    print(f"Metadata: {result['metadata']}")

import asyncio
asyncio.run(test_service())
```

**Validation:** ✓ Service wraps provider correctly

---

### Step 2.2: Add Retry Logic ⬜

**Update:** `panel_app/services/inference_service.py`

**Dependencies:**
- Install: `pip install tenacity` (for retry decorators)

**What to add:**

```python
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class InferenceService:
    def __init__(
        self,
        provider: BaseLLMProvider,
        repository=None,
        max_retries: int = 3,
        initial_wait: float = 1.0
    ):
        self.provider = provider
        self.repository = repository
        self.max_retries = max_retries
        self.initial_wait = initial_wait

    async def run_inference(self, prompt: str, **kwargs) -> dict:
        """Run inference with retry logic"""
        response = await self._call_with_retry(prompt, **kwargs)

        return {
            'response': response.text,
            'metadata': {
                'provider': response.provider,
                'tokens': response.tokens,
                'latency_ms': response.latency_ms,
            }
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def _call_with_retry(self, prompt: str, **kwargs) -> ProviderResponse:
        """Internal method with retry decorator"""
        return await self.provider.complete(prompt, **kwargs)
```

**Test:** Simulate network errors and verify retry behavior

**Validation:** ✓ Service retries on transient errors

---

### Step 2.3: Add Prompt Preprocessing ⬜

**Files to create:**
- `panel_app/services/preprocessors.py`

**What to build:**

```python
# panel_app/services/preprocessors.py

import re
from typing import Optional

class PromptPreprocessor:
    """Utilities for preprocessing prompts before sending to LLM"""

    @staticmethod
    def clean_whitespace(prompt: str) -> str:
        """Normalize whitespace"""
        return " ".join(prompt.split())

    @staticmethod
    def apply_template(prompt: str, template: str) -> str:
        """Apply a prompt template"""
        return template.format(user_input=prompt)

    @staticmethod
    def truncate(prompt: str, max_chars: int = 10000) -> str:
        """Truncate to max length"""
        if len(prompt) > max_chars:
            return prompt[:max_chars] + "..."
        return prompt

    @staticmethod
    def remove_pii(prompt: str) -> str:
        """Basic PII removal (emails, phone numbers)"""
        # Remove emails
        prompt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', prompt)
        # Remove phone numbers (US format)
        prompt = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', prompt)
        return prompt
```

**Update:** `panel_app/services/inference_service.py`

```python
from .preprocessors import PromptPreprocessor

class InferenceService:
    def __init__(self, provider, repository=None, preprocess: bool = True):
        self.provider = provider
        self.repository = repository
        self.preprocess = preprocess

    async def run_inference(self, prompt: str, template: Optional[str] = None, **kwargs) -> dict:
        """Run inference with preprocessing"""

        # Preprocess prompt
        if self.preprocess:
            prompt = PromptPreprocessor.clean_whitespace(prompt)
            prompt = PromptPreprocessor.truncate(prompt)
            if template:
                prompt = PromptPreprocessor.apply_template(prompt, template)

        # Call provider with retry
        response = await self._call_with_retry(prompt, **kwargs)

        return {...}
```

**Test:**
```python
def test_preprocessing():
    preprocessor = PromptPreprocessor()

    # Test whitespace
    assert preprocessor.clean_whitespace("hello   world") == "hello world"

    # Test PII removal
    text = "Contact me at test@example.com or 555-123-4567"
    cleaned = preprocessor.remove_pii(text)
    assert "@" not in cleaned
    assert "555" not in cleaned
```

**Validation:** ✓ Prompts are preprocessed correctly

---

### Step 2.4: Multi-Turn Conversation Service ⬜ (Optional for now)

**Files to create:**
- `panel_app/services/conversation_service.py`

**What to build:**

```python
# panel_app/services/conversation_service.py

from typing import Optional
from panel_app.providers.base import BaseLLMProvider

class ConversationService:
    """Manage multi-turn conversations with LLMs"""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.conversations: dict[str, list[dict]] = {}

    def start_conversation(self, conversation_id: str, system_prompt: Optional[str] = None):
        """Initialize a new conversation"""
        self.conversations[conversation_id] = []
        if system_prompt:
            self.conversations[conversation_id].append({
                "role": "system",
                "content": system_prompt
            })

    async def send_message(self, conversation_id: str, message: str) -> dict:
        """Send message in conversation context"""
        if conversation_id not in self.conversations:
            self.start_conversation(conversation_id)

        # Add user message
        self.conversations[conversation_id].append({
            "role": "user",
            "content": message
        })

        # Get response (provider needs to support messages parameter)
        # Note: This assumes OpenAI-style chat format
        response = await self.provider.complete(
            messages=self.conversations[conversation_id]
        )

        # Add assistant response
        self.conversations[conversation_id].append({
            "role": "assistant",
            "content": response.text
        })

        return {
            'response': response.text,
            'history_length': len(self.conversations[conversation_id])
        }

    def get_conversation(self, conversation_id: str) -> list[dict]:
        """Retrieve conversation history"""
        return self.conversations.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
```

**Test:** Have a multi-turn conversation

**Validation:** ✓ Conversation context maintained across turns

---

### Step 2.5: Request Batching Service ⬜ (Optional for now)

**Files to create:**
- `panel_app/services/batch_service.py`

**What to build:**

```python
# panel_app/services/batch_service.py

import asyncio
import hashlib
from typing import Optional
from panel_app.providers.base import BaseLLMProvider

class BatchInferenceService:
    """Deduplicate and batch identical concurrent requests"""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.pending_requests: dict[str, asyncio.Future] = {}

    async def run_inference_with_dedup(self, prompt: str, **kwargs) -> dict:
        """Run inference, deduplicating identical concurrent requests"""

        # Hash prompt + kwargs to create request ID
        request_hash = self._hash_request(prompt, **kwargs)

        # Check if identical request is already in flight
        if request_hash in self.pending_requests:
            # Wait for existing request
            return await self.pending_requests[request_hash]

        # Create new future for this request
        future = asyncio.create_task(self._execute_request(prompt, **kwargs))
        self.pending_requests[request_hash] = future

        try:
            result = await future
            return result
        finally:
            # Clean up
            del self.pending_requests[request_hash]

    async def _execute_request(self, prompt: str, **kwargs) -> dict:
        """Execute the actual provider call"""
        response = await self.provider.complete(prompt, **kwargs)
        return {
            'response': response.text,
            'metadata': {
                'provider': response.provider,
                'tokens': response.tokens,
            }
        }

    @staticmethod
    def _hash_request(prompt: str, **kwargs) -> str:
        """Create hash of prompt + parameters"""
        content = f"{prompt}:{sorted(kwargs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()
```

**Test:**
```python
async def test_dedup():
    provider = ProviderFactory.create("openai", api_key="...")
    batch_service = BatchInferenceService(provider)

    # Send 5 identical requests concurrently
    tasks = [
        batch_service.run_inference_with_dedup("Say hello")
        for _ in range(5)
    ]

    results = await asyncio.gather(*tasks)

    # Should only call provider once
    print(f"Got {len(results)} results from 1 API call")
```

**Validation:** ✓ Duplicate requests share single API call

---

### Phase 2 Milestone ✓

**What you have now:**
- Full-featured business logic layer
- Retry logic with exponential backoff
- Prompt preprocessing
- Multi-turn conversations (optional)
- Request deduplication (optional)
- Still no database - all testable in isolation

**Next:** Add database persistence

---

## Phase 3: Database Layer
**Goal:** Add persistence for inference results

**Dependencies:** Phase 1 (Providers), Phase 2 (Services)

### Step 3.1: Database Models ⬜

**Files to create:**
- `panel_app/db/__init__.py`
- `panel_app/db/models.py`

**Dependencies:**
- Install: `pip install sqlalchemy psycopg2-binary`

**What to build:**

```python
# panel_app/db/models.py

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InferenceRun(Base):
    """Model for storing LLM inference runs"""
    __tablename__ = 'inference_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    provider = Column(String(50), nullable=False, index=True)
    tokens = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<InferenceRun(id={self.id}, provider={self.provider}, created_at={self.created_at})>"
```

**Test:**
```python
# test_models.py
from panel_app.db.models import Base, InferenceRun
from sqlalchemy import create_engine

# Create in-memory SQLite database
engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)

print("Tables created successfully!")
```

**Validation:** ✓ Tables can be created

---

### Step 3.2: Database Connection ⬜

**Files to create:**
- `panel_app/db/connection.py`

**What to build:**

```python
# panel_app/db/connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from .models import Base

class DatabaseConnection:
    """Manages database connections and sessions"""

    def __init__(self, connection_string: str):
        """
        Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                Examples:
                - PostgreSQL: "postgresql://user:pass@localhost:5432/dbname"
                - SQLite: "sqlite:///path/to/db.sqlite"
        """
        self.engine = create_engine(connection_string, echo=False)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def init_db(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

**Test:**
```python
# test_connection.py
from panel_app.db.connection import DatabaseConnection

# Test with SQLite
db = DatabaseConnection("sqlite:///test.db")
db.init_db()

with db.session_scope() as session:
    # Session works
    print("Database connection successful!")
```

**Validation:** ✓ Can connect and create tables

---

### Step 3.3: Repository - Write Operations ⬜

**Files to create:**
- `panel_app/db/inference_repository.py`

**What to build:**

```python
# panel_app/db/inference_repository.py

from typing import Optional
from sqlalchemy.orm import Session
from .models import InferenceRun

class InferenceRepository:
    """
    Repository for all database operations on inference runs.
    Handles both writes (logging inferences) and queries (analytics).
    """

    def __init__(self, session: Session):
        self.session = session

    # ========== WRITE OPERATIONS ==========

    def insert_inference_run(
        self,
        prompt: str,
        response: str,
        provider: str,
        tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Insert a single inference run.

        Returns:
            The ID of the inserted record
        """
        inference = InferenceRun(
            prompt=prompt,
            response=response,
            provider=provider,
            tokens=tokens,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )

        self.session.add(inference)
        self.session.commit()
        self.session.refresh(inference)

        return inference.id

    def batch_insert_inferences(self, inferences: list[dict]) -> list[int]:
        """
        Batch insert multiple inference runs.

        Args:
            inferences: List of dicts with inference data

        Returns:
            List of inserted IDs
        """
        inference_objects = [
            InferenceRun(**inf) for inf in inferences
        ]

        self.session.add_all(inference_objects)
        self.session.commit()

        return [inf.id for inf in inference_objects]
```

**Test:**
```python
# test_repository_write.py
from panel_app.db.connection import DatabaseConnection
from panel_app.db.inference_repository import InferenceRepository

db = DatabaseConnection("sqlite:///test.db")
db.init_db()

with db.session_scope() as session:
    repo = InferenceRepository(session)

    # Insert single record
    inference_id = repo.insert_inference_run(
        prompt="Test prompt",
        response="Test response",
        provider="azure",
        tokens=100,
        latency_ms=500.0
    )

    print(f"Inserted inference with ID: {inference_id}")
```

**Validation:** ✓ Can write records to database

---

### Step 3.4: Repository - Query Operations ⬜

**Update:** `panel_app/db/inference_repository.py`

**What to add:**

```python
from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy import func

class InferenceRepository:
    # ... write operations above ...

    # ========== QUERY OPERATIONS ==========

    def get_inference_by_id(self, inference_id: int) -> Optional[InferenceRun]:
        """Retrieve a specific inference run by ID"""
        return self.session.query(InferenceRun).filter_by(id=inference_id).first()

    def query_inferences(
        self,
        provider: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[InferenceRun]:
        """
        Query inferences with filters.

        Args:
            provider: Filter by provider name
            date_range: Tuple of (start_date, end_date)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of InferenceRun objects
        """
        query = self.session.query(InferenceRun)

        if provider:
            query = query.filter(InferenceRun.provider == provider)

        if date_range:
            start_date, end_date = date_range
            query = query.filter(InferenceRun.created_at.between(start_date, end_date))

        query = query.order_by(InferenceRun.created_at.desc())
        query = query.limit(limit).offset(offset)

        return query.all()

    def get_provider_stats(self) -> list[dict]:
        """
        Get aggregate statistics by provider.

        Returns:
            List of dicts with provider stats
        """
        results = self.session.query(
            InferenceRun.provider,
            func.count(InferenceRun.id).label('count'),
            func.avg(InferenceRun.tokens).label('avg_tokens'),
            func.avg(InferenceRun.latency_ms).label('avg_latency_ms'),
            func.sum(InferenceRun.tokens).label('total_tokens')
        ).group_by(InferenceRun.provider).all()

        return [
            {
                'provider': r.provider,
                'count': r.count,
                'avg_tokens': round(r.avg_tokens, 2) if r.avg_tokens else 0,
                'avg_latency_ms': round(r.avg_latency_ms, 2) if r.avg_latency_ms else 0,
                'total_tokens': r.total_tokens or 0
            }
            for r in results
        ]

    def search_by_prompt(self, search_term: str, limit: int = 50) -> list[InferenceRun]:
        """
        Search inferences by prompt content.

        Args:
            search_term: Text to search for in prompts
            limit: Maximum results

        Returns:
            List of matching InferenceRun objects
        """
        return self.session.query(InferenceRun)\
            .filter(InferenceRun.prompt.ilike(f'%{search_term}%'))\
            .order_by(InferenceRun.created_at.desc())\
            .limit(limit)\
            .all()

    def get_total_count(self) -> int:
        """Get total number of inference runs"""
        return self.session.query(func.count(InferenceRun.id)).scalar()
```

**Test:**
```python
# test_repository_query.py
from panel_app.db.connection import DatabaseConnection
from panel_app.db.inference_repository import InferenceRepository

db = DatabaseConnection("sqlite:///test.db")

with db.session_scope() as session:
    repo = InferenceRepository(session)

    # Insert test data
    for i in range(10):
        repo.insert_inference_run(
            prompt=f"Test prompt {i}",
            response=f"Response {i}",
            provider="azure" if i % 2 == 0 else "openai",
            tokens=100 + i * 10,
            latency_ms=500 + i * 50
        )

    # Query by provider
    azure_runs = repo.query_inferences(provider="azure")
    print(f"Azure runs: {len(azure_runs)}")

    # Get stats
    stats = repo.get_provider_stats()
    print(f"Provider stats: {stats}")

    # Search
    results = repo.search_by_prompt("prompt 5")
    print(f"Search results: {len(results)}")
```

**Validation:** ✓ Can query and aggregate data

---

### Step 3.5: Integrate Services with Repository ⬜

**Update:** `panel_app/services/inference_service.py`

**What to change:**

```python
from typing import Optional
from panel_app.providers.base import BaseLLMProvider
from panel_app.db.inference_repository import InferenceRepository

class InferenceService:
    def __init__(
        self,
        provider: BaseLLMProvider,
        repository: Optional[InferenceRepository] = None,  # Now optional
        preprocess: bool = True
    ):
        self.provider = provider
        self.repository = repository
        self.preprocess = preprocess

    async def run_inference(self, prompt: str, **kwargs) -> dict:
        """Run inference and optionally persist to database"""

        # Preprocess if enabled
        if self.preprocess:
            # ... preprocessing logic ...
            pass

        # Call provider with retry
        response = await self._call_with_retry(prompt, **kwargs)

        # Persist to database if repository provided
        inference_id = None
        if self.repository:
            inference_id = self.repository.insert_inference_run(
                prompt=prompt,
                response=response.text,
                provider=response.provider,
                tokens=response.tokens,
                latency_ms=response.latency_ms,
                metadata=response.metadata
            )

        return {
            'id': inference_id,
            'response': response.text,
            'metadata': {
                'provider': response.provider,
                'tokens': response.tokens,
                'latency_ms': response.latency_ms,
            }
        }
```

**Test:**
```python
# test_service_with_db.py
from panel_app.providers import ProviderFactory
from panel_app.services.inference_service import InferenceService
from panel_app.db.connection import DatabaseConnection
from panel_app.db.inference_repository import InferenceRepository

async def test():
    # Setup database
    db = DatabaseConnection("sqlite:///test.db")
    db.init_db()

    # Create provider and service
    provider = ProviderFactory.create("openai", api_key="YOUR_KEY")

    with db.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(provider, repository=repo)

        # Run inference - should save to DB
        result = await service.run_inference("Hello world!")

        print(f"Saved with ID: {result['id']}")
        print(f"Response: {result['response']}")

        # Verify it's in database
        saved = repo.get_inference_by_id(result['id'])
        print(f"Retrieved from DB: {saved.prompt}")

import asyncio
asyncio.run(test())
```

**Validation:** ✓ Inferences are automatically saved to database

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

### Step 4.1: Study Service ⬜

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

### Step 5.1: Simple Inference UI ⬜

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

### Step 5.2: Analytics Dashboard ⬜

**What to add:**
1. Provider comparison bar chart
2. Time-series line chart
3. Recent inferences table
4. Summary statistics cards
5. Search interface

**Test:** Verify charts update with real data

---

### Step 5.3: Configuration Management ⬜

**Files to create:**
- `.env.example`
- `panel_app/config.py`

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

### Phase 5 Milestone ✓

**What you have now:**
- Full GUI application
- Inference testing interface
- Analytics dashboard
- Configuration management
- Complete end-to-end functionality

**Next:** Polish, testing, deployment

---

## Phase 6: Polish and Deployment
**Goal:** Production readiness

### Step 6.1: Error Handling and Logging ⬜

Add comprehensive logging throughout:
- Provider API calls
- Database operations
- User actions

### Step 6.2: Unit Tests ⬜

Create test suite:
- `tests/test_providers.py`
- `tests/test_services.py`
- `tests/test_repository.py`

### Step 6.3: Documentation ⬜

Update:
- README.md with full setup instructions
- API documentation
- User guide

### Step 6.4: Docker Setup ⬜

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

## Summary: Implementation Checklist

### Phase 1: Provider Layer
- [ ] Step 1.1: Base provider interface
- [ ] Step 1.2: Azure provider
- [ ] Step 1.3: OpenAI provider
- [ ] Step 1.4: Hyperbolic provider
- [ ] Step 1.5: Provider factory

### Phase 2: Service Layer
- [ ] Step 2.1: Basic inference service
- [ ] Step 2.2: Add retry logic
- [ ] Step 2.3: Add preprocessing
- [ ] Step 2.4: Conversation service (optional)
- [ ] Step 2.5: Batch service (optional)

### Phase 3: Database Layer
- [ ] Step 3.1: Database models
- [ ] Step 3.2: Database connection
- [ ] Step 3.3: Repository writes
- [ ] Step 3.4: Repository queries
- [ ] Step 3.5: Service integration

### Phase 4: Analytics Layer
- [ ] Step 4.1: Study service

### Phase 5: GUI Layer
- [ ] Step 5.1: Inference UI
- [ ] Step 5.2: Analytics dashboard
- [ ] Step 5.3: Configuration

### Phase 6: Production
- [ ] Step 6.1: Error handling
- [ ] Step 6.2: Unit tests
- [ ] Step 6.3: Documentation
- [ ] Step 6.4: Docker setup

---

## Next Steps

Ready to begin **Phase 1, Step 1.1: Base Provider Interface**

This is the smallest possible starting point that:
- Has zero dependencies
- Is fully testable
- Sets the foundation for everything else
- Can be validated in < 5 minutes

After confirming this works, each subsequent step builds on the previous one!
