# Contributing Guidelines

## Development Process

### 1. Check Current Status
Before starting any work, review:
- `docs/living/CURRENT_STATE.md` for current capabilities
- `docs/living/ARCHITECTURE_CURRENT.md` for current architecture
- `docs/IMPLEMENTATION_PLAN.md` for historical phase context

### 2. Follow Architecture
Adhere to patterns documented in `docs/living/ARCHITECTURE_CURRENT.md`:
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
- **docs/living/CURRENT_STATE.md**: Update current capability status
- **docs/living/ARCHITECTURE_CURRENT.md**: Update if current design patterns change
- **README.md**: Update status section when phases complete
- **docs/living/API_CURRENT.md**: Update if public APIs change
- Add docstrings to all public classes and functions

### 5. Update Status
When completing work, update:
- `docs/living/CURRENT_STATE.md` for current behavior changes
- `docs/review/DOC_PARITY_LEDGER.md` for major documentation-claim changes
- `docs/IMPLEMENTATION_PLAN.md` only for historical/roadmap chronology

## Status Markers in IMPLEMENTATION_PLAN.md

`docs/IMPLEMENTATION_PLAN.md` is maintained as historical phased context.

## Code Review Checklist

Before submitting changes, verify:
- [ ] Tests pass (`pytest`)
- [ ] Documentation updated (`docs/living/*`, runbooks, relevant docs)
- [ ] Parity-sensitive claims updated in `docs/review/DOC_PARITY_LEDGER.md`
- [ ] No secrets committed (check for API keys, credentials)
- [ ] Follows architecture patterns (see `docs/living/ARCHITECTURE_CURRENT.md`)
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
6. Update `docs/living/CURRENT_STATE.md` and related living docs; update `docs/IMPLEMENTATION_PLAN.md` only for historical roadmap context

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

- Check `docs/living/ARCHITECTURE_CURRENT.md` for current design patterns
- Review `docs/living/CURRENT_STATE.md` for current status
- See `docs/living/API_CURRENT.md` for current API reference
- Check existing tests for examples
