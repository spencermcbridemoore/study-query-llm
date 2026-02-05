# Standing Orders for Development

This document establishes consistent practices for all development work on this project, ensuring that multiple developers (human or AI) maintain consistency in planning, implementation, and documentation.

## Planning Consistency

### Primary Source of Truth
- **`docs/IMPLEMENTATION_PLAN.md`** is the authoritative source for:
  - What has been implemented (✅)
  - What is partially complete (⚠️)
  - What is planned but not started (⬜)
  - Current development phase and next steps

### Before Starting Work
1. **Always check `IMPLEMENTATION_PLAN.md`** for current phase status
2. Verify your work aligns with the documented phases
3. Check if similar functionality already exists
4. Review `ARCHITECTURE.md` to understand design patterns

### After Completing Work
1. **Immediately update status markers** in `IMPLEMENTATION_PLAN.md`
2. Update `README.md` status section if a phase completes
3. Update relevant documentation (ARCHITECTURE.md, API.md) if design changed
4. Ensure tests are written and passing

### Plan File Management (Cursor-specific)
When creating temporary plan files in "plan mode", use session-aware naming to prevent conflicts across agent sessions. See `.cursor/PLANNING_GUIDE.md` for details on:
- Session identification using `CURSOR_TRACE_ID`
- Creating uniquely-named plan files in `.plans/` directory
- Using the `session_utils` module for plan file management

## Code Consistency

### Development Approach
- **Bottom-up, incremental development** (see IMPLEMENTATION_PLAN.md philosophy)
- Each component must be **testable in isolation**
- Components are **independent of layers above them**
- Natural dependency order (no circular dependencies)

### Code Style
- **File operations**: Always use `encoding='utf-8'` for Windows compatibility
  ```python
  with open(file_path, 'r', encoding='utf-8') as f:
      content = f.read()
  ```
- **Async/await**: Use for all I/O operations (LLM API calls, database operations)
- **Dependency injection**: Services receive dependencies via constructor (no global state)
- **Type hints**: Use where appropriate for clarity

### Architecture Patterns
- **Provider Layer**: Abstract base class `BaseLLMProvider` with standardized `ProviderResponse`
- **Service Layer**: Business logic with dependency injection
- **Repository Pattern**: All database access through repository classes
- **Factory Pattern**: `ProviderFactory` for creating provider instances

## Documentation Consistency

### Documentation Sync Process

**When completing a phase/step:**
1. Update status in `IMPLEMENTATION_PLAN.md` (✅/⚠️/⬜)
2. Update `README.md` status section if applicable
3. Add/update relevant docstrings in code
4. Update `ARCHITECTURE.md` if design changed

**When adding new features:**
1. Check if it fits existing phases or needs new phase
2. Document in `IMPLEMENTATION_PLAN.md` with appropriate status
3. Update `ARCHITECTURE.md` if architectural change
4. Update `README.md` features list
5. Update `API.md` if public API changed

**When changing architecture:**
1. Update `ARCHITECTURE.md` with new patterns/decisions
2. Update `IMPLEMENTATION_PLAN.md` if phases affected
3. Update code docstrings to reflect changes
4. Update `README.md` if user-facing changes

### Documentation Files
- **`docs/IMPLEMENTATION_PLAN.md`**: Phased roadmap and status tracking
- **`docs/ARCHITECTURE.md`**: System design, layer responsibilities, patterns
- **`docs/API.md`**: Programmatic API reference
- **`docs/USER_GUIDE.md`**: End-user documentation
- **`README.md`**: Project overview and quickstart
- **`SECURITY.md`**: Security guidelines
- **`CONTRIBUTING.md`**: Contributor guidelines

## Testing Consistency

### Test Requirements
- **All new code must have tests**
- Tests mirror source structure: `tests/test_services/` mirrors `src/study_query_llm/services/`
- Use existing fixtures from `tests/conftest.py`
- Mark API-dependent tests with `@pytest.mark.requires_api`

### Test Execution
- Run `pytest` before committing
- Ensure all tests pass
- Use in-memory SQLite for repository tests
- Mock providers for service layer tests

## Database Schema Consistency

### Schema Versions
- **v1 schema**: Legacy schema (Phase 3) - `models.py`, `connection.py`, `inference_repository.py`
- **v2 schema**: Current immutable schema (Phase 7.1) - `models_v2.py`, `connection_v2.py`, `raw_call_repository.py`

### Schema Usage Rules
- **Always use v2 schema for new features**
- v2 schema includes: `RawCall`, `Group`, `GroupMember`, `CallArtifact`, `EmbeddingVector`, `GroupLink`
- Migration from v1 to v2: Use `scripts/migrate_v1_to_v2.py`

## Git Workflow Consistency

### Commit Practices
- **Stage/commit at sensible checkpoints**: Complete a feature, fix a bug, finish a logical unit of work
- Commit after making related changes that form a coherent unit (e.g., all changes for one feature, all fixes for one bug)
- Avoid committing broken/incomplete code that would break the build or tests
- Stage/commit/push when finishing checklists or completing tasks
- Use descriptive commit messages
- Never commit `.env` files or API keys

### Branch Strategy
- Work on `main` branch (or feature branches as needed)
- Keep commits focused and atomic
- Push regularly to maintain backup

## Security Consistency

### Secrets Management
- **Never hardcode** API keys or credentials
- Use environment variables via `config.py`
- Never commit `.env` files (see `SECURITY.md`)
- Validate user input in services layer

## Algorithm Development Consistency

### Algorithm Location
- Core algorithms in `src/study_query_llm/algorithms/`
- Minimal dependencies (numpy/scipy only)
- No dependencies on DB/LLM layers

### Algorithm Usage
- Notebooks/scripts can use algorithms
- Algorithms should be testable in isolation
- Use provenance service for tracking runs (Phase 7.8)

## Project Structure Consistency

```
study-query-llm/
├── src/study_query_llm/    # Core package (framework-agnostic)
│   ├── providers/          # LLM provider abstractions
│   ├── services/           # Business logic layer
│   ├── db/                 # Database models and repositories
│   ├── algorithms/         # Core algorithms (minimal deps)
│   └── utils/              # Utilities (logging, etc.)
├── panel_app/              # Panel GUI application
├── tests/                  # Test suite (mirrors src/ structure)
├── scripts/                # Standalone utility scripts
├── notebooks/              # Jupyter notebooks for analysis
└── docs/                   # All project documentation
```

## Key Principles

1. **Single Source of Truth**: `IMPLEMENTATION_PLAN.md` for status, `ARCHITECTURE.md` for design
2. **Documentation Sync**: Keep all docs in sync with code changes
3. **Test First**: Write tests for all new components
4. **Incremental Development**: Build and test each component before moving to next
5. **Consistency**: Follow established patterns and conventions

## Quick Reference

### Status Update Checklist
- [ ] Update `IMPLEMENTATION_PLAN.md` status markers
- [ ] Update `README.md` if phase completed
- [ ] Update relevant documentation (ARCHITECTURE.md, API.md)
- [ ] Add/update code docstrings
- [ ] Ensure tests pass

### Before Starting Work
- [ ] Check `IMPLEMENTATION_PLAN.md` for current status
- [ ] Review `ARCHITECTURE.md` for design patterns
- [ ] Verify no duplicate functionality exists
- [ ] Plan test strategy

### Code Quality Checklist
- [ ] Tests written and passing
- [ ] Uses `encoding='utf-8'` for file operations
- [ ] Follows async/await pattern for I/O
- [ ] Uses dependency injection
- [ ] Uses v2 database schema (if database work)
- [ ] No secrets committed
