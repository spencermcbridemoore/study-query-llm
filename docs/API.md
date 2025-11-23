# API Documentation - Study Query LLM

This document provides reference documentation for the programmatic API of Study Query LLM.

## Table of Contents

- [Providers](#providers)
- [Services](#services)
- [Database](#database)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Providers

### BaseLLMProvider

Abstract base class for all LLM providers.

```python
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse

class MyProvider(BaseLLMProvider):
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        # Implementation
        pass
    
    def get_provider_name(self) -> str:
        return "my_provider"
```

### ProviderResponse

Standardized response format from all providers.

```python
@dataclass
class ProviderResponse:
    text: str                    # Generated text
    provider: str                # Provider name
    tokens: Optional[int] = None # Total tokens used
    latency_ms: Optional[float] = None  # Response time in milliseconds
    metadata: dict[str, Any] = None      # Provider-specific metadata
    raw_response: Any = None     # Raw API response
```

### AzureOpenAIProvider

Azure OpenAI provider implementation.

```python
from study_query_llm.providers.azure_provider import AzureOpenAIProvider
from study_query_llm.config import ProviderConfig

config = ProviderConfig(
    name="azure",
    api_key="your-key",
    endpoint="https://your-resource.openai.azure.com/",
    deployment_name="gpt-4o",
    api_version="2024-02-15-preview"
)

provider = AzureOpenAIProvider(config)
response = await provider.complete("Hello, world!")
```

**Methods:**
- `complete(prompt, temperature=0.7, max_tokens=None, **kwargs)` - Generate completion
- `get_provider_name()` - Return provider name
- `close()` - Close the client connection
- `list_deployments(config)` - Static method to list available deployments

### ProviderFactory

Factory for creating provider instances.

```python
from study_query_llm.providers.factory import ProviderFactory

factory = ProviderFactory()

# Create from config
provider = factory.create_from_config("azure")

# Create with custom config
from study_query_llm.config import ProviderConfig
config = ProviderConfig(...)
provider = factory.create("azure", config)

# List available providers
providers = factory.get_available_providers()  # ['azure', 'openai', 'hyperbolic']

# List provider deployments (Azure)
deployments = await factory.list_provider_deployments("azure")
```

## Services

### InferenceService

Core service for running LLM inferences with business logic.

```python
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.providers.factory import ProviderFactory

# Create provider
factory = ProviderFactory()
provider = factory.create_from_config("azure")

# Create service
service = InferenceService(
    provider=provider,
    repository=repository,  # Optional
    max_retries=3,
    preprocess=False
)

# Run single inference
result = await service.run_inference(
    prompt="What is Python?",
    temperature=0.7,
    max_tokens=100
)
# Returns: {'response': '...', 'metadata': {...}, 'id': 123}

# Run batch inference (different prompts)
prompts = ["What is Python?", "What is JavaScript?"]
results = await service.run_batch_inference(
    prompts,
    temperature=0.7
)
# Returns: [{'response': '...', ...}, ...]

# Run sampling inference (same prompt, multiple times)
results = await service.run_sampling_inference(
    prompt="Say hello",
    n=5,
    temperature=0.7
)
# Returns: [{'response': '...', ...}, ...]
```

**Parameters:**
- `provider` - BaseLLMProvider instance
- `repository` - Optional InferenceRepository for database logging
- `max_retries` - Maximum retry attempts (default: 3)
- `initial_wait` - Initial wait time for exponential backoff (default: 1.0)
- `max_wait` - Maximum wait time between retries (default: 10.0)
- `preprocess` - Enable prompt preprocessing (default: False)
- `clean_whitespace` - Normalize whitespace (default: True)
- `truncate_prompts` - Truncate long prompts (default: True)
- `max_prompt_length` - Maximum prompt length (default: 10000)
- `remove_pii` - Remove PII from prompts (default: False)
- `strip_control_chars` - Remove control characters (default: False)

### StudyService

Service for analyzing stored inference data.

```python
from study_query_llm.services.study_service import StudyService
from study_query_llm.db.inference_repository import InferenceRepository

with db.session_scope() as session:
    repository = InferenceRepository(session)
    study = StudyService(repository)
    
    # Get provider comparison
    comparison_df = study.get_provider_comparison()
    # Returns: pandas DataFrame with provider stats
    
    # Get recent inferences
    recent_df = study.get_recent_inferences(limit=50, provider="azure")
    # Returns: pandas DataFrame with recent inference data
    
    # Get time series data
    time_series_df = study.get_time_series_data(days=7, group_by='day')
    # Returns: pandas DataFrame with time-series aggregated data
    
    # Search prompts
    results_df = study.search_prompts("python")
    # Returns: pandas DataFrame with matching prompts
    
    # Get summary statistics
    stats = study.get_summary_stats()
    # Returns: dict with total_inferences, total_tokens, unique_providers, etc.
```

## Database

### DatabaseConnection

Manages database connections and sessions.

```python
from study_query_llm.db.connection import DatabaseConnection

# Initialize connection
db = DatabaseConnection("sqlite:///study_query_llm.db")

# Initialize tables
db.init_db()

# Use session context manager
with db.session_scope() as session:
    # Perform database operations
    pass
```

**Methods:**
- `init_db()` - Create all database tables
- `get_session()` - Get a new database session
- `session_scope()` - Context manager for transactional operations
- `drop_all_tables()` - Drop all tables (WARNING: deletes all data)
- `recreate_db()` - Drop and recreate all tables

### InferenceRepository

Repository for database operations on inference runs.

```python
from study_query_llm.db.inference_repository import InferenceRepository

with db.session_scope() as session:
    repo = InferenceRepository(session)
    
    # Insert inference
    inference_id = repo.insert_inference_run(
        prompt="What is Python?",
        response="Python is a programming language...",
        provider="azure_openai_gpt-4o",
        tokens=150,
        latency_ms=500.0,
        metadata={"temperature": 0.7},
        batch_id="batch-123"  # Optional
    )
    
    # Query inferences
    runs = repo.query_inferences(
        provider="azure",
        date_range=(start_date, end_date),
        limit=100,
        offset=0
    )
    
    # Get by ID
    inference = repo.get_inference_by_id(inference_id)
    
    # Get provider statistics
    stats = repo.get_provider_stats()
    # Returns: [{'provider': '...', 'count': 10, 'avg_tokens': 150, ...}, ...]
    
    # Search by prompt
    results = repo.search_by_prompt("python", limit=50)
    
    # Get total count
    total = repo.get_total_count()
    
    # Batch operations
    batch_ids = repo.batch_insert_inferences([{...}, {...}])
    
    # Batch tracking
    batch_runs = repo.get_inferences_by_batch_id("batch-123")
    batch_summary = repo.get_batch_summary("batch-123")
```

### InferenceRun Model

SQLAlchemy model for inference runs.

```python
from study_query_llm.db.models import InferenceRun

# Model fields:
inference.id              # Integer, primary key
inference.prompt          # Text, required
inference.response        # Text, required
inference.provider        # String(50), indexed
inference.tokens          # Integer, optional
inference.latency_ms      # Float, optional
inference.metadata_json   # JSON, optional
inference.batch_id        # String(36), optional, indexed
inference.created_at      # DateTime, indexed

# Convert to dict
inference_dict = inference.to_dict()
```

## Configuration

### Config

Application configuration loaded from environment variables.

```python
from study_query_llm.config import config

# Get database config
db_url = config.database.connection_string

# Get provider config
azure_config = config.get_provider_config("azure")
# Returns: ProviderConfig with api_key, endpoint, deployment_name, etc.

# Get available providers
providers = config.get_available_providers()
# Returns: ['azure', 'openai', ...] (only providers with API keys)
```

### ProviderConfig

Configuration for a specific LLM provider.

```python
from study_query_llm.config import ProviderConfig

config = ProviderConfig(
    name="azure",
    api_key="your-key",
    endpoint="https://...",
    deployment_name="gpt-4o",
    api_version="2024-02-15-preview"
)
```

## Utilities

### Logging

Centralized logging configuration.

```python
from study_query_llm.utils.logging_config import get_logger, setup_logging
import logging

# Setup logging (called automatically on import)
setup_logging(
    level=logging.INFO,
    log_file=Path("logs/app.log"),  # Optional
    format_string=None  # Optional custom format
)

# Get logger for a module
logger = get_logger(__name__)
logger.info("Application started")
logger.debug("Detailed debug info")
logger.error("Error occurred", exc_info=True)
```

## Examples

### Complete Example: Run Inference and Analyze

```python
import asyncio
from study_query_llm.config import config
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.inference_repository import InferenceRepository
from study_query_llm.services.study_service import StudyService

async def main():
    # Setup database
    db = DatabaseConnection(config.database.connection_string)
    db.init_db()
    
    # Create provider and service
    factory = ProviderFactory()
    provider = factory.create_from_config("azure")
    
    with db.session_scope() as session:
        repository = InferenceRepository(session)
        service = InferenceService(provider, repository=repository)
        
        # Run inference
        result = await service.run_inference("What is Python?")
        print(f"Response: {result['response']}")
        print(f"Inference ID: {result['id']}")
        
        # Analyze results
        study = StudyService(repository)
        stats = study.get_summary_stats()
        print(f"Total inferences: {stats['total_inferences']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: Batch Inference

```python
async def run_batch():
    factory = ProviderFactory()
    provider = factory.create_from_config("azure")
    
    db = DatabaseConnection(config.database.connection_string)
    with db.session_scope() as session:
        repository = InferenceRepository(session)
        service = InferenceService(provider, repository=repository)
        
        prompts = [
            "What is Python?",
            "What is JavaScript?",
            "What is Rust?"
        ]
        
        results = await service.run_batch_inference(prompts)
        print(f"Ran {len(results)} inferences")
        for result in results:
            print(f"Batch ID: {result.get('batch_id')}")
```

### Example: Sampling Inference

```python
async def run_sampling():
    factory = ProviderFactory()
    provider = factory.create_from_config("azure")
    
    db = DatabaseConnection(config.database.connection_string)
    with db.session_scope() as session:
        repository = InferenceRepository(session)
        service = InferenceService(provider, repository=repository)
        
        # Run same prompt 5 times
        results = await service.run_sampling_inference(
            "Say hello in one word",
            n=5,
            temperature=0.9  # Higher temperature for more variety
        )
        
        print(f"Got {len(results)} varied responses")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['response']}")
```

