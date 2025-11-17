# Study-Query-LLM Architecture

## Overview

This project is a Panel-based web application for running LLM inference experiments across multiple providers (Azure, OpenAI, Hyperbolic, etc.) and analyzing the results stored in a PostgreSQL database via Langfuse.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                         │
│                     (Panel GUI)                              │
│  - Interactive dashboard for running inferences              │
│  - Analytics views for studying results                      │
│  - Real-time visualization of metrics                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Service Layer                               │
│               (Business Logic)                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  InferenceService                                     │  │
│  │  - Retry logic with exponential backoff              │  │
│  │  - Prompt preprocessing                              │  │
│  │  - Error handling & logging                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ConversationService (Future)                        │  │
│  │  - Multi-turn conversation management                │  │
│  │  - Context window handling                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  BatchService (Future)                               │  │
│  │  - Request deduplication                             │  │
│  │  - Batch processing                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  StudyService                                        │  │
│  │  - Analytics queries                                 │  │
│  │  - Provider comparisons                              │  │
│  │  - Historical analysis                               │  │
│  └──────────────────────────────────────────────────────┘  │
└────────┬───────────────────────────────────────────┬────────┘
         │                                           │
         │ Uses                                      │ Uses
         │                                           │
┌────────▼──────────────────┐         ┌─────────────▼──────────┐
│   Provider Layer          │         │  Data Access Layer     │
│  (LLM Abstraction)        │         │   (Repository)         │
├───────────────────────────┤         ├────────────────────────┤
│  BaseLLMProvider (ABC)    │         │  InferenceRepository   │
│  ├─ AzureProvider         │         │  - insert_*()          │
│  ├─ OpenAIProvider        │         │  - query_*()           │
│  ├─ HyperbolicProvider    │         │  - get_stats()         │
│  └─ ...more providers     │         │  - search_*()          │
│                           │         └────────────┬───────────┘
│  ProviderFactory          │                      │
│  - create(name, config)   │                      │
└───────────────────────────┘                      │
                                         ┌─────────▼───────────┐
                                         │  Database Layer     │
                                         │  (PostgreSQL)       │
                                         ├─────────────────────┤
                                         │  SQLAlchemy Models  │
                                         │  - InferenceRun     │
                                         │  - Metadata         │
                                         │  - (Langfuse)       │
                                         └─────────────────────┘
```

## Layer Responsibilities

### 1. Presentation Layer (Panel GUI)
**Location:** `panel_app/app.py`

**Responsibilities:**
- User interface for running inferences
- Displaying analytics and visualizations
- Reactive UI updates
- User input validation

**Key Patterns:**
- Panel's `@pn.depends` for reactivity
- FastListTemplate for consistent layout
- Tabbed interface for different workflows

**Does NOT:**
- Make direct LLM API calls
- Contain business logic
- Access database directly

---

### 2. Service Layer (Business Logic)
**Location:** `panel_app/services/`

**Responsibilities:**
- Orchestrate provider calls with retry/preprocessing
- Implement multi-turn conversations
- Handle request batching and deduplication
- Transform data for analytics
- Coordinate between providers and repository

**Key Services:**

#### InferenceService
- Primary service for single inference runs
- Retry logic with exponential backoff
- Prompt preprocessing (whitespace, templates, truncation)
- Error handling and logging
- Persists results to database via repository

#### ConversationService (Future)
- Maintains conversation history in memory
- Manages multi-turn context
- Handles conversation persistence

#### BatchService (Future)
- Deduplicates identical concurrent requests
- Batches requests to same provider
- Optimizes API usage

#### StudyService
- Analytics and reporting
- Provider performance comparison
- Historical query analysis
- Data aggregation for visualization

**Key Patterns:**
- Dependency injection (services receive provider + repository)
- Async/await for I/O operations
- Composition over inheritance (services can wrap services)

---

### 3. Provider Layer (LLM Abstraction)
**Location:** `panel_app/providers/`

**Responsibilities:**
- Abstract interface for all LLM providers
- Normalize provider responses to common format
- Handle provider-specific authentication
- Raw API communication (no retry/preprocessing here)

**Key Components:**

#### BaseLLMProvider (ABC)
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> ProviderResponse

    @abstractmethod
    def get_provider_name(self) -> str
```

#### Individual Providers
- `AzureProvider` - Azure OpenAI
- `OpenAIProvider` - OpenAI API
- `HyperbolicProvider` - Hyperbolic
- Extensible for more providers

#### ProviderFactory
- Creates provider instances based on name
- Loads configuration
- Enables easy provider switching

**Key Patterns:**
- Strategy pattern (interchangeable providers)
- Factory pattern for creation
- Standardized response format

---

### 4. Data Access Layer (Repository)
**Location:** `panel_app/db/`

**Responsibilities:**
- All database interactions (writes and queries)
- SQL query optimization
- Transaction management
- Data validation against schema

**Key Components:**

#### InferenceRepository
**Write Operations:**
- `insert_inference_run()` - Single inference
- `batch_insert_inferences()` - Bulk insert

**Query Operations:**
- `get_inference_by_id()` - Fetch specific run
- `query_inferences()` - Filter by provider, date, etc.
- `get_provider_stats()` - Aggregations
- `search_by_prompt_similarity()` - Vector search (future)

**Key Patterns:**
- Repository pattern (data access abstraction)
- Unit of Work pattern (session management)
- Single responsibility (all DB logic here)

---

### 5. Database Layer (PostgreSQL + Langfuse)
**Location:** `panel_app/db/models.py`

**Responsibilities:**
- Data persistence
- Schema definition
- Relationships and constraints

**Core Models:**

#### InferenceRun
```python
- id (PK)
- prompt (text)
- response (text)
- provider (string)
- tokens (int)
- latency_ms (float)
- metadata (JSON)
- created_at (timestamp)
```

**Integration:**
- Langfuse for LLM observability
- PostgreSQL for structured storage
- Potential vector extensions for similarity search

---

## Data Flow Examples

### Running an Inference

```
User Input (Panel)
  │
  ├─> InferenceService.run_inference()
  │     ├─> Preprocessor.clean_prompt()
  │     ├─> AzureProvider.complete()  [API Call]
  │     │     └─> ProviderResponse
  │     ├─> InferenceRepository.insert_inference_run()
  │     │     └─> PostgreSQL INSERT
  │     └─> Return formatted result
  │
  └─> Display in Panel UI
```

### Querying Analytics

```
User Request (Panel)
  │
  ├─> StudyService.get_provider_comparison()
  │     ├─> InferenceRepository.get_provider_stats()
  │     │     └─> PostgreSQL SELECT with GROUP BY
  │     └─> Transform to DataFrame
  │
  └─> Render chart in Panel
```

### Multi-Turn Conversation (Future)

```
User Message (Panel)
  │
  ├─> ConversationService.send_message(conv_id, message)
  │     ├─> Load history from memory/DB
  │     ├─> Append user message
  │     ├─> OpenAIProvider.complete(messages=history)
  │     ├─> Append assistant response
  │     ├─> InferenceRepository.insert_inference_run()
  │     └─> Return response
  │
  └─> Update conversation UI
```

---

## Key Design Decisions

### 1. Single Repository for Writes and Queries
**Decision:** Use one `InferenceRepository` with both write and query methods.

**Rationale:**
- Avoids duplicate connection management
- Easier transaction handling (operations that read + write)
- Simpler dependency injection
- Single source of truth for database logic

**Alternative Rejected:** Separate InferenceWriter and QueryReader classes
- Would create artificial separation
- More boilerplate
- Harder to maintain consistency

### 2. Service Layer for Business Logic
**Decision:** Create service layer between GUI and providers/database.

**Rationale:**
- GUI stays thin and focused on presentation
- Business logic (retry, batching, etc.) reusable
- Easier to test in isolation
- Supports future needs (CLI, API endpoints)

### 3. Provider Abstraction
**Decision:** Abstract all LLM providers behind common interface.

**Rationale:**
- Easy to switch providers
- Consistent response handling
- Testable with mock providers
- Add new providers without changing service layer

### 4. Bottom-Up Implementation
**Decision:** Build smallest components first, compose upward.

**Rationale:**
- Each component testable in isolation
- No circular dependencies
- Can validate each layer before building next
- Easier to course-correct

### 5. Async Throughout
**Decision:** Use async/await for all I/O operations.

**Rationale:**
- LLM API calls can take seconds
- Enables concurrent requests
- Panel supports async reactivity
- Better resource utilization

---

## Technology Stack

### Frontend
- **Panel** - Python-based web framework with reactive UI

### Backend
- **Python 3.10+** - Core language
- **asyncio** - Async operations
- **SQLAlchemy** - ORM for database

### LLM Providers
- **Azure OpenAI** - Enterprise LLM API
- **OpenAI** - GPT models
- **Hyperbolic** - Alternative provider
- **Extensible** - Can add more

### Database
- **PostgreSQL** - Primary datastore
- **Langfuse** - LLM observability layer
- **pgvector** (Future) - Vector similarity search

### Observability
- **Langfuse** - Trace inference runs
- **Python logging** - Application logs

---

## Security Considerations

### API Key Management
- Store in environment variables
- Never commit to git
- Use `.env` file for local development
- Azure Key Vault for production

### Database Security
- Parameterized queries (SQLAlchemy ORM)
- Connection string encryption
- Least privilege database user
- SSL/TLS for connections

### Input Validation
- Sanitize user prompts
- Validate provider selection
- Rate limiting on inference requests
- Token limit enforcement

---

## Future Enhancements

### Phase 2+ Features
- [ ] Multi-turn conversation support
- [ ] Request batching and deduplication
- [ ] Vector similarity search for prompts
- [ ] Cost tracking per provider
- [ ] A/B testing framework
- [ ] Prompt template library
- [ ] Experiment versioning
- [ ] Export to various formats
- [ ] Real-time streaming responses
- [ ] Multi-provider fallback

### Scalability
- [ ] Async database operations
- [ ] Connection pooling
- [ ] Caching layer (Redis)
- [ ] Background job queue (Celery)
- [ ] Read replicas for analytics

---

## Development Workflow

### Adding a New Provider
1. Create `panel_app/providers/new_provider.py`
2. Implement `BaseLLMProvider` interface
3. Add to `ProviderFactory`
4. Test with simple script
5. Update documentation

### Adding a New Service
1. Create `panel_app/services/new_service.py`
2. Inject required dependencies (provider, repository)
3. Implement business logic methods
4. Add unit tests
5. Integrate with GUI

### Adding New Database Queries
1. Add method to `InferenceRepository`
2. Implement SQL logic
3. Test with sample data
4. Create corresponding service method if needed
5. Expose in GUI

---

## Testing Strategy

### Unit Tests
- Mock providers for service layer tests
- In-memory SQLite for repository tests
- Pytest fixtures for common setups

### Integration Tests
- Test full stack with test database
- Use test API keys for providers
- Verify end-to-end flows

### Manual Testing
- Panel UI for exploratory testing
- Jupyter notebooks for data validation
- Local PostgreSQL instance

---

## Deployment

### Local Development
```bash
# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
python -m panel_app.app
```

### Production (Future)
- Docker containers
- Environment-based configuration
- Kubernetes for orchestration
- Monitoring and alerting

---

## References

- [Panel Documentation](https://panel.holoviz.org/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [Langfuse](https://langfuse.com/docs)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
