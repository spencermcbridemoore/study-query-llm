# Test Suite

This directory contains the pytest test suite for Study Query LLM.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── test_providers/          # Tests for LLM provider implementations
│   ├── test_base.py        # Base provider interface tests
│   └── test_azure.py       # Azure provider tests
└── test_services/          # Tests for business logic services
    └── test_inference.py   # Inference service tests
```

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_providers/test_base.py
```

### Run specific test

```bash
pytest tests/test_providers/test_base.py::test_basic_completion
```

### Run with coverage

```bash
pytest --cov=study_query_llm --cov-report=html
```

### Skip API tests (if credentials not configured)

```bash
pytest -m "not requires_api"
```

### Run only fast tests

```bash
pytest -m "not slow"
```

### Run in parallel

```bash
pytest -n auto
```

## Test Markers

- `@pytest.mark.requires_api` - Tests that need API keys (will skip if not configured)
- `@pytest.mark.slow` - Tests that take a long time
- `@pytest.mark.integration` - Integration tests

## Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_provider` - Mock LLM provider for unit tests
- `echo_provider` - Provider that echoes prompts (for preprocessing tests)
- `failing_provider` - Provider that fails (for retry tests)
- `azure_config` - Azure provider configuration (skips if not configured)
- `openai_config` - OpenAI provider configuration (skips if not configured)

## Migration from Old Test Scripts

The old test scripts (`test_phase_*.py`) are still in the root directory for reference.
They can be removed once all tests are migrated to pytest format.

