"""
Tests for Phase 2.2 - Retry Logic with Exponential Backoff.

Tests that InferenceService correctly retries transient errors using
exponential backoff, while not retrying permanent failures.
"""

import pytest
import time
from study_query_llm.services.inference_service import InferenceService


@pytest.mark.asyncio
async def test_retry_on_timeout_errors(failing_provider):
    """Test that timeout errors are retried."""
    provider = failing_provider(fail_count=2, error_type="timeout")
    service = InferenceService(
        provider,
        max_retries=3,
        initial_wait=0.1,  # Short wait for testing
        max_wait=1.0
    )
    
    start = time.time()
    result = await service.run_inference("Test prompt")
    duration = time.time() - start
    
    assert provider.attempts == 3, "Should have made 3 attempts"
    assert result['response'] == "Success after retries!"
    assert duration > 0.1  # Should include retry delays


@pytest.mark.asyncio
async def test_retry_on_rate_limit_errors(failing_provider):
    """Test that rate limit errors are retried."""
    provider = failing_provider(fail_count=1, error_type="rate_limit")
    service = InferenceService(provider, max_retries=3, initial_wait=0.1)
    
    result = await service.run_inference("Test prompt")
    
    assert provider.attempts == 2, "Should have made 2 attempts"
    assert result['response'] == "Success after retries!"


@pytest.mark.asyncio
async def test_retry_on_503_errors(failing_provider):
    """Test that 503 Service Unavailable errors are retried."""
    provider = failing_provider(fail_count=1, error_type="503")
    service = InferenceService(provider, max_retries=3, initial_wait=0.1)
    
    result = await service.run_inference("Test prompt")
    
    assert provider.attempts == 2, "Should have made 2 attempts"
    assert result['response'] == "Success after retries!"


@pytest.mark.asyncio
async def test_retry_on_connection_errors(failing_provider):
    """Test that connection errors are retried."""
    provider = failing_provider(fail_count=1, error_type="connection")
    service = InferenceService(provider, max_retries=3, initial_wait=0.1)
    
    result = await service.run_inference("Test prompt")
    
    assert provider.attempts == 2, "Should have made 2 attempts"
    assert result['response'] == "Success after retries!"


@pytest.mark.asyncio
async def test_retry_exhaustion(failing_provider):
    """Test that retries are exhausted when max_retries is reached."""
    provider = failing_provider(fail_count=10, error_type="timeout")  # Fail more than max_retries
    service = InferenceService(provider, max_retries=2, initial_wait=0.1)
    
    with pytest.raises(Exception):
        await service.run_inference("Test prompt")
    
    assert provider.attempts == 2, "Should have made exactly max_retries attempts"


@pytest.mark.asyncio
async def test_no_retry_on_permanent_errors(permanently_failing_provider):
    """Test that permanent errors (like 401) are NOT retried."""
    service = InferenceService(permanently_failing_provider, max_retries=3, initial_wait=0.1)
    
    with pytest.raises(Exception, match="401 Unauthorized"):
        await service.run_inference("Test prompt")
    
    assert permanently_failing_provider.attempts == 1, "Should NOT have retried permanent error"


@pytest.mark.asyncio
async def test_exponential_backoff_timing(failing_provider):
    """Test that exponential backoff increases wait times between retries."""
    provider = failing_provider(fail_count=2, error_type="timeout")
    service = InferenceService(
        provider,
        max_retries=3,
        initial_wait=0.1,
        max_wait=1.0
    )
    
    start = time.time()
    await service.run_inference("Test prompt")
    duration = time.time() - start
    
    # With exponential backoff, should take at least initial_wait * (1 + 2) = 0.3s
    # Allow some margin for execution time
    assert duration >= 0.2, "Should include exponential backoff delays"


@pytest.mark.asyncio
async def test_retry_with_different_max_retries(failing_provider):
    """Test that max_retries parameter is respected."""
    # Test with max_retries=1
    provider1 = failing_provider(fail_count=2, error_type="timeout")
    service1 = InferenceService(provider1, max_retries=1, initial_wait=0.1)
    
    with pytest.raises(Exception):
        await service1.run_inference("Test prompt")
    
    assert provider1.attempts == 1, "Should have made exactly 1 attempt"
    
    # Test with max_retries=5
    provider2 = failing_provider(fail_count=3, error_type="timeout")
    service2 = InferenceService(provider2, max_retries=5, initial_wait=0.1)
    
    result = await service2.run_inference("Test prompt")
    assert provider2.attempts == 4, "Should have made 4 attempts (3 failures + 1 success)"
    assert result['response'] == "Success after retries!"

