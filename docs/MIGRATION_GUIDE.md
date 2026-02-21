# Migration Guide: Test Scripts to Pytest

This guide explains how to migrate from the old test scripts to the new pytest-based test suite.

## Quick Start

1. **Install pytest dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run the new tests:**
   ```bash
   pytest
   ```

3. **Old test scripts are still available** in the root directory for reference.

## What Changed

### Old Approach (test_phase_*.py)
- Standalone scripts with `asyncio.run()`
- Manual print statements for output
- Manual assertion checking
- Run with: `python test_phase_1_1.py`

### New Approach (pytest)
- Standard pytest test functions
- Automatic test discovery
- Rich assertion reporting
- Run with: `pytest tests/`

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_providers/
│   ├── test_base.py        # Phase 1.1 tests
│   └── test_azure.py       # Phase 1.2 tests
└── test_services/
    ├── test_inference.py   # Phase 2.1 tests
    └── test_preprocessing.py # Phase 2.3 tests
```

## Key Differences

### 1. Test Functions

**Old:**
```python
async def test_basic_completion():
    print("Testing...")
    provider = MockProvider()
    response = await provider.complete("test")
    assert response.text is not None
    print("[PASS]")
```

**New:**
```python
@pytest.mark.asyncio
async def test_basic_completion(mock_provider):
    response = await mock_provider.complete("test")
    assert response.text is not None
```

### 2. Fixtures Instead of Manual Setup

**Old:**
```python
provider = MockProvider()
service = InferenceService(provider)
```

**New:**
```python
def test_something(mock_provider):  # Fixture injected automatically
    service = InferenceService(mock_provider)
```

### 3. Test Markers

**Old:**
```python
# Manual skip logic
try:
    config = config.get_provider_config("azure")
except ValueError:
    print("Skipping - no credentials")
    return
```

**New:**
```python
@pytest.mark.requires_api
def test_azure(azure_config):  # Fixture handles skipping
    # Test code here
```

## Converting Your Own Tests

### Step 1: Move to tests/ directory

Create a file like `tests/test_providers/test_your_feature.py`

### Step 2: Convert test function

```python
# Add pytest imports
import pytest

# Add marker for async tests
@pytest.mark.asyncio

# Use fixtures from conftest.py
async def test_your_feature(mock_provider):
    # Your test code
    pass
```

### Step 3: Use fixtures

Check `tests/conftest.py` for available fixtures:
- `mock_provider` - Mock LLM provider
- `echo_provider` - Echo provider for preprocessing tests
- `failing_provider` - Provider that fails (for retry tests)
- `azure_config` - Azure config (skips if not configured)

### Step 4: Add markers if needed

```python
@pytest.mark.slow  # For slow tests
@pytest.mark.requires_api  # For API tests
@pytest.mark.integration  # For integration tests
```

## Running Tests

### All tests
```bash
pytest
```

### Specific file
```bash
pytest tests/test_providers/test_base.py
```

### Specific test
```bash
pytest tests/test_providers/test_base.py::test_basic_completion
```

### With coverage
```bash
pytest --cov=study_query_llm --cov-report=html
```

### Skip API tests
```bash
pytest -m "not requires_api"
```

## Benefits

1. **Automatic discovery** - No need to manually run scripts
2. **Better output** - Rich assertion diffs and error messages
3. **Fixtures** - Reusable test setup
4. **Markers** - Easy test categorization
5. **Parallel execution** - Run tests faster with `-n auto`
6. **CI/CD ready** - Standard format for automation

## Next Steps

1. Convert remaining test files (test_phase_2_2.py, test_phase_2_4.py, etc.)
2. Add more fixtures as needed
3. Set up CI/CD to run pytest automatically
4. Remove old test scripts once migration is complete

## Need Help?

- See `tests/README.md` for more examples
- Check `tests/conftest.py` for available fixtures
- Run `pytest --fixtures` to see all available fixtures

