# Standing Orders for Development

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-06

This document establishes consistent practices for all development work on this project, ensuring that multiple developers (human or AI) maintain consistency in planning, implementation, and documentation.

## Planning Consistency

### Primary Source of Truth
- **`docs/living/CURRENT_STATE.md`** is authoritative for "what currently exists/works."
- **`docs/living/ARCHITECTURE_CURRENT.md`** is authoritative for current architecture.
- **`docs/IMPLEMENTATION_PLAN.md`** and **`docs/ARCHITECTURE.md`** are historical/roadmap context unless explicitly updated as current.
- **`docs/review/DOC_PARITY_LEDGER.md`** is the evidence ledger for doc-to-code parity claims.

### Before Starting Work
1. **Always check `docs/living/CURRENT_STATE.md`** for current capabilities
2. Verify your work aligns with the documented phases
3. Check if similar functionality already exists
4. Review `docs/living/ARCHITECTURE_CURRENT.md` for current design patterns

### After Completing Work
1. **Immediately update living docs** that changed (`CURRENT_STATE`, current architecture/API, runbooks as needed)
2. Update `docs/review/DOC_PARITY_LEDGER.md` when significant claims change
3. Update historical docs only when preserving chronology/context
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

**When completing a phase/milestone:**
1. Update `docs/living/CURRENT_STATE.md` if behavior/capabilities changed
2. Update `docs/review/DOC_PARITY_LEDGER.md` if claim parity changed
3. Add/update relevant docstrings in code
4. Update `docs/living/ARCHITECTURE_CURRENT.md` if design changed

**When adding new features:**
1. Check if it fits existing phases or needs new phase
2. Document in `docs/living/CURRENT_STATE.md` and route from `docs/README.md`
3. Update `docs/living/ARCHITECTURE_CURRENT.md` if architectural change
4. Update `README.md` features list when user-facing behavior changes
5. Update `docs/living/API_CURRENT.md` if public API changed

**When changing architecture:**
1. Update `docs/living/ARCHITECTURE_CURRENT.md` with new patterns/decisions
2. Update `docs/living/CURRENT_STATE.md` if capability boundaries changed
3. Update code docstrings to reflect changes
4. Update `README.md`/`docs/README.md` routing if user-facing doc entrypoints changed

### Documentation Files
- **`docs/README.md`**: Documentation routing and taxonomy
- **`docs/living/CURRENT_STATE.md`**: Authoritative current capabilities
- **`docs/living/ARCHITECTURE_CURRENT.md`**: Current architecture reference
- **`docs/living/API_CURRENT.md`**: Current API quick reference
- **`docs/USER_GUIDE.md`**: End-user guide (v2-first)
- **`docs/IMPLEMENTATION_PLAN.md`**: Historical phased roadmap
- **`docs/ARCHITECTURE.md`**: Historical architecture narrative
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
- Migration from v1 to v2: follow currently supported migration tooling in `scripts/README.md`

### Method Definitions and Provenance
- **Whenever possible, register methods in the methods table** (`method_definitions`) and record results in `analysis_results`, rather than encoding "which method" only as new group types, link types, or free-form metadata in `groups.metadata_json`.
- **Runs stay in groups**: A *run* (Group with provenance stages, artifacts, raw_calls) is still the unit of execution; the method is *which* algorithm/version that run used. Use `method_definitions` for the method identity and version (e.g. `code_ref`, `code_commit`); use groups/links for run and provenance-stage lineage.
- **New analysis or algorithms**: Prefer registering a row in `method_definitions` (via `MethodService.register_method`) and writing structured results to `analysis_results` (via `MethodService.record_result`) keyed by `source_group_id` (and optionally `analysis_group_id`). Avoid introducing new `group_type` or link types solely to represent "kind of method."
- **Parameters convention (soft)**: When a method has `parameters_schema`, include `result_json["parameters"]` with the run parameters (e.g., job payload) when recording results. This links parameters to results for queryability; validation is optional.
- **Canonical run fingerprint**: Every `provenanced_runs` row should carry a `fingerprint_json`/`fingerprint_hash` that captures algorithmic identity (method, config, input, data regime) and excludes scheduling mechanics. Use `canonical_run_fingerprint()` from `provenanced_run_service.py`; compare runs with `fingerprints_match()`.
- **Scheduling vs provenance boundary**: See [SCHEDULING_PROVENANCE_BOUNDARY.md](living/SCHEDULING_PROVENANCE_BOUNDARY.md) for the rule on when a sub-stage should be an orchestration job vs an in-job provenance event.
- **Exceptions**: One-off or throwaway analyses that are not reused or versioned need not be registered; legacy or existing code that uses `metadata_json.algorithm` (or similar) can remain until migrated.

### Execution Vocabulary (Terminology Guardrail)

To avoid ambiguity, use the following canonical terms in docs and reviews:

- **`provenance_stage`**: A lineage node within a run/request provenance graph.
- **`algorithm_iteration`**: One inner-loop update cycle inside an iterative algorithm.
- **`restart_try`**: One seeded restart/try for a fixed run configuration.
- **`orchestration_job`**: One schedulable/leased control-plane unit.
- **`planning_step`**: A roadmap milestone such as `STEP-*` in `docs/plans/*`.

Rules:

- Avoid bare conceptual use of "step" in prose.
- Keep literal schema/code identifiers unchanged and quoted in backticks (for example `step_name`, `step_type`, `clustering_step`).
- Use [SCHEDULING_PROVENANCE_BOUNDARY.md](living/SCHEDULING_PROVENANCE_BOUNDARY.md) for job-vs-stage boundary decisions.

## Git Workflow Consistency

### Commit Practices
- **MANDATORY: Stage and commit when completing a feature or task**
  - When all todos for a feature are completed → stage/commit immediately
  - When a bug is fixed → stage/commit immediately
  - When a logical unit of work is finished → stage/commit immediately
  - **Do NOT wait for user to ask** - commit proactively at natural completion points
- **Stage/commit at sensible checkpoints**: Complete a feature, fix a bug, finish a logical unit of work
- Commit after making related changes that form a coherent unit (e.g., all changes for one feature, all fixes for one bug)
- Avoid committing broken/incomplete code that would break the build or tests
- Stage/commit/push when finishing checklists or completing tasks
- Use descriptive commit messages that explain what was changed and why
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

1. **Single Source of Truth (Current)**: `docs/living/CURRENT_STATE.md` for status and `docs/living/ARCHITECTURE_CURRENT.md` for design
2. **Documentation Sync**: Keep all docs in sync with code changes
3. **Test First**: Write tests for all new components
4. **Incremental Development**: Build and test each component before moving to next
5. **Consistency**: Follow established patterns and conventions

## Quick Reference

### Status Update Checklist
- [ ] Update `docs/living/CURRENT_STATE.md` if current behavior changed
- [ ] Update `docs/review/DOC_PARITY_LEDGER.md` for changed doc claims
- [ ] Update `README.md` if phase completed
- [ ] Update relevant documentation (`docs/living/ARCHITECTURE_CURRENT.md`, `docs/living/API_CURRENT.md`, runbooks)
- [ ] Add/update code docstrings
- [ ] Ensure tests pass

### Before Starting Work
- [ ] Check `docs/living/CURRENT_STATE.md` for current status
- [ ] Review `docs/living/ARCHITECTURE_CURRENT.md` for design patterns
- [ ] Verify no duplicate functionality exists
- [ ] Plan test strategy

### Code Quality Checklist
- [ ] Tests written and passing
- [ ] Uses `encoding='utf-8'` for file operations
- [ ] Follows async/await pattern for I/O
- [ ] Uses dependency injection
- [ ] Uses v2 database schema (if database work)
- [ ] New analysis/algorithm methods registered in `method_definitions` and results in `analysis_results` when applicable (see § Method Definitions and Provenance)
- [ ] No secrets committed
