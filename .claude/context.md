# Claude Code Context - Study Query LLM

## Quick Project Summary

**What:** Panel-based web app for running LLM inference experiments across multiple providers (Azure, OpenAI, Hyperbolic) and analyzing results stored in PostgreSQL/Langfuse.

**Current Status:** Minimal Panel boilerplate - ready for implementation

**Architecture:** Bottom-up layered design
- Provider Layer (LLM abstraction)
- Service Layer (business logic)
- Database Layer (PostgreSQL via SQLAlchemy)
- Presentation Layer (Panel GUI)

## Key Files

### Documentation
- `docs/ARCHITECTURE.md` - Full system architecture and design decisions
- `docs/IMPLEMENTATION_PLAN.md` - Phased implementation roadmap (6 phases, 15+ steps)
- `README.md` - Project overview and quickstart

### Current Code
- `panel_app/app.py` - Minimal Panel demo app (needs LLM functionality)
- `panel_app/__init__.py` - Package exports
- `setup.py` - Package setup
- `requirements.txt` - Currently minimal (just panel>=1.3.0)

### To Be Created
- `panel_app/providers/` - LLM provider abstraction layer
- `panel_app/services/` - Business logic (retry, preprocessing, analytics)
- `panel_app/db/` - Database models and repository
- `.env` - Configuration (API keys, database URL)

## Implementation Status

### Current Phase: **Not Started - Ready to Begin**

**Next Step:** Phase 1, Step 1.1 - Create base provider interface
- File: `panel_app/providers/base.py`
- Creates: `BaseLLMProvider` (ABC) and `ProviderResponse` (dataclass)
- Time: ~5 minutes
- Testable: Yes, with mock provider

### Completed
- ✅ Project structure cleaned up (minimalist rewrite)
- ✅ Architecture documented
- ✅ Implementation plan defined

### TODO (High Priority)
1. Phase 1: Provider abstraction layer (base + Azure + OpenAI + Hyperbolic + factory)
2. Phase 2: Service layer (inference service + retry + preprocessing)
3. Phase 3: Database layer (models + repository + integration)
4. Phase 4: Analytics service
5. Phase 5: GUI integration
6. Phase 6: Polish and deployment

## Design Decisions to Remember

### 1. Single Repository for Writes and Queries
Use one `InferenceRepository` class with both write methods (`insert_*`) and query methods (`query_*`, `get_stats`).

**Why:** Avoids duplicate connection management, easier transactions, simpler dependency injection.

### 2. Bottom-Up Implementation
Build smallest components first (provider interface), then compose upward (service layer, then GUI).

**Why:** Each layer is testable before the next is built. No rework needed.

### 3. Optional Database in Services
Services accept repository as optional parameter: `InferenceService(provider, repository=None)`

**Why:** Can test service logic without database. Repository injected in production.

### 4. Async Throughout
All I/O operations use `async`/`await`.

**Why:** LLM API calls take seconds. Enables concurrent requests. Panel supports async reactivity.

### 5. Provider Abstraction
All LLM providers implement `BaseLLMProvider` interface with standardized `ProviderResponse`.

**Why:** Easy to switch providers. Consistent response handling. Testable with mocks.

## Common Workflows

### Adding a New Provider
1. Create `panel_app/providers/new_provider.py`
2. Implement `BaseLLMProvider` interface
3. Add to `ProviderFactory.create()`
4. Test with simple async script
5. Add config to `.env` and `AppConfig`

### Adding Business Logic
1. Add method to existing service (or create new service in `panel_app/services/`)
2. Inject required dependencies (provider, repository)
3. Use repository for database operations
4. Use provider for LLM calls
5. Test in isolation with mocks

### Adding Database Queries
1. Add method to `InferenceRepository`
2. Use SQLAlchemy ORM for queries
3. Return domain objects or dicts
4. Create corresponding `StudyService` method if analytics-related
5. Expose in GUI via Panel components

## Tech Stack

**Frontend:** Panel (Python web framework)
**Backend:** Python 3.10+, asyncio
**Database:** PostgreSQL + Langfuse (LLM observability)
**ORM:** SQLAlchemy
**LLM Providers:** Azure OpenAI, OpenAI, Hyperbolic (extensible)
**Visualization:** hvplot, holoviews, bokeh

## Development Commands

```bash
# Setup (when ready)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Run current demo app
python -m panel_app.app
# Or: panel serve panel_app/app.py

# When tests are added
pytest tests/
```

## Configuration

Environment variables (create `.env` file):
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/study_llm
AZURE_API_KEY=...
AZURE_ENDPOINT=...
AZURE_DEPLOYMENT=gpt-4
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4
HYPERBOLIC_API_KEY=...
```

## Testing Strategy

**Unit Tests:** Mock providers and database for service layer tests
**Integration Tests:** Use test database and test API keys for full stack
**Manual Tests:** Panel GUI for exploratory testing

## Important Notes

- **Never commit API keys** - use .env and .gitignore
- **Test each phase** before moving to next - don't skip validation steps
- **Keep services thin** - complex logic goes in service layer, not GUI
- **Use repository pattern** - all database operations through repository
- **Inject dependencies** - don't instantiate providers/repos inside services

## Quick Reference: Layer Responsibilities

| Layer | Responsibilities | What it DOESN'T do |
|-------|-----------------|-------------------|
| **Provider** | Talk to LLM APIs, normalize responses | Retry, preprocessing, database |
| **Service** | Business logic, retry, orchestration | Direct API calls, SQL queries |
| **Repository** | All database operations (read/write) | Business logic, API calls |
| **GUI (Panel)** | Display, user input, reactivity | Database access, API calls |

## When Returning to This Project

1. Check implementation status in `docs/IMPLEMENTATION_PLAN.md`
2. Review architecture in `docs/ARCHITECTURE.md`
3. Continue from next unchecked step in plan
4. Each step has:
   - Files to create
   - Code to write
   - Test to run
   - Validation criteria

## Git Workflow

Current branch: `main`

**Recent commits:**
- e25cb38 - minimalist codex rewrite attempt (current)
- 29a61bd - Got initial web served version working
- ccb3090 - Initial commit

**Modified files (uncommitted):**
- README.md
- notebooks/demo.ipynb
- panel_app/__init__.py
- panel_app/app.py
- Deleted: panel_app/components/, panel_app/utils/
- setup.py

## Questions to Ask User (if needed)

- Which provider to implement first? (Default: Azure)
- PostgreSQL or SQLite for development? (Default: SQLite for simplicity)
- Integrate Langfuse SDK or just use PostgreSQL? (Default: Just PostgreSQL initially)
- Multi-turn conversation support in Phase 2 or later? (Default: Optional, can add later)

## Useful Links

- Panel docs: https://panel.holoviz.org/
- SQLAlchemy ORM: https://docs.sqlalchemy.org/
- Langfuse: https://langfuse.com/docs
- Azure OpenAI: https://learn.microsoft.com/en-us/azure/ai-services/openai/
