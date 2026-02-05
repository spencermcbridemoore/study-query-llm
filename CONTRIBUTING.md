# Contributing Guidelines

## Development Process

### 1. Check Current Status
Before starting any work, review `docs/IMPLEMENTATION_PLAN.md` to understand:
- What phases are complete (✅)
- What's partially implemented (⚠️)
- What's planned but not started (⬜)
- Current development phase and next steps

### 2. Follow Architecture
Adhere to patterns documented in `docs/ARCHITECTURE.md`:
- Bottom-up, incremental development
- Layered architecture (Provider → Service → Repository → Database)
- Dependency injection (no global state)
- Async/await for I/O operations

### 3. Write Tests
All new code must have tests:
- Place tests in `tests/` directory mirroring `src/` structure
- Use fixtures from `tests/conftest.py`
- Mark API-dependent tests with `@pytest.mark.requires_api`
- Run `pytest` before committing

### 4. Update Documentation
Keep documentation in sync with code changes:
- **IMPLEMENTATION_PLAN.md**: Update status markers when completing work
- **ARCHITECTURE.md**: Update if design patterns change
- **README.md**: Update status section when phases complete
- **API.md**: Update if public APIs change
- Add docstrings to all public classes and functions

### 5. Update Status
When completing work, update `docs/IMPLEMENTATION_PLAN.md`:
- ✅ Implemented - Mark complete
- ⚠️ Partially implemented - Note what's missing
- ⬜ Not implemented - Leave as-is until started

## Status Markers in IMPLEMENTATION_PLAN.md

- ✅ **Implemented** - Feature is complete and tested
- ⚠️ **Partially implemented** - Feature exists but missing some components
- ⬜ **Not implemented** - Planned but not yet started

## Code Review Checklist

Before submitting changes, verify:
- [ ] Tests pass (`pytest`)
- [ ] Documentation updated (IMPLEMENTATION_PLAN.md, relevant docs)
- [ ] IMPLEMENTATION_PLAN.md status markers updated
- [ ] No secrets committed (check for API keys, credentials)
- [ ] Follows architecture patterns (see ARCHITECTURE.md)
- [ ] Uses v2 database schema for new features
- [ ] File operations use `encoding='utf-8'` (Windows compatibility)

## Project Structure

```
study-query-llm/
├── src/study_query_llm/    # Core package (framework-agnostic)
│   ├── providers/          # LLM provider abstractions
│   ├── services/           # Business logic layer
│   ├── db/                 # Database models and repositories
│   ├── algorithms/         # Core algorithms (minimal deps)
│   └── utils/              # Utilities
├── panel_app/              # Panel GUI application
├── tests/                  # Test suite (mirrors src/ structure)
├── scripts/                # Standalone utility scripts
├── notebooks/              # Jupyter notebooks
└── docs/                   # Documentation
```

## Adding New Features

### Adding a New Provider
1. Create `src/study_query_llm/providers/your_provider.py`
2. Implement `BaseLLMProvider` interface
3. Add to `ProviderFactory`
4. Update configuration in `config.py`
5. Add tests in `tests/test_providers/`
6. Update documentation

### Adding a New Service
1. Create `src/study_query_llm/services/your_service.py`
2. Inject required dependencies (provider, repository) via constructor
3. Implement business logic methods
4. Add unit tests in `tests/test_services/`
5. Integrate with GUI if needed
6. Update IMPLEMENTATION_PLAN.md

### Adding Database Queries
1. Add method to appropriate repository (`raw_call_repository.py` for v2)
2. Implement SQL logic using SQLAlchemy ORM
3. Test with sample data
4. Create corresponding service method if needed
5. Expose in GUI if user-facing

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest --cov=study_query_llm tests/

# Run specific test file
pytest tests/test_providers/test_azure.py

# Skip API-dependent tests
pytest -m "not requires_api"
```

### Test Structure
- Tests mirror source structure: `tests/test_services/` mirrors `src/study_query_llm/services/`
- Use fixtures from `tests/conftest.py` for common setup
- Mock providers for service layer tests
- Use in-memory SQLite for repository tests

## Git Workflow

- Stage/commit at logical milestones
- Stage/commit/push when finishing checklists
- Use descriptive commit messages
- Never commit `.env` files or API keys (see `SECURITY.md`)

## Security

- **Never commit secrets**: API keys, credentials, or `.env` files
- Use environment variables via `config.py`
- See `SECURITY.md` for detailed security guidelines

## Getting Help

- Check `docs/ARCHITECTURE.md` for design patterns
- Review `docs/IMPLEMENTATION_PLAN.md` for current status
- See `docs/API.md` for API reference
- Check existing tests for examples
