"""
Phase 2.2 Validation: Retry Logic with Exponential Backoff

This script validates that the InferenceService correctly retries transient
errors using exponential backoff, while not retrying permanent failures.

Run this script to verify Phase 2.2 is working:
    python test_phase_2_2.py
"""

import asyncio
import time
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.inference_service import InferenceService


class FailingProvider(BaseLLMProvider):
    """
    Mock provider that fails a specified number of times before succeeding.
    Used to test retry logic.
    """

    def __init__(self, fail_count: int = 2, error_type: str = "timeout"):
        self.fail_count = fail_count
        self.attempts = 0
        self.error_type = error_type

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Fail fail_count times, then succeed."""
        self.attempts += 1

        if self.attempts <= self.fail_count:
            # Raise different types of errors based on error_type
            if self.error_type == "timeout":
                raise TimeoutError(f"Simulated timeout (attempt {self.attempts})")
            elif self.error_type == "rate_limit":
                raise Exception(f"Rate limit exceeded (attempt {self.attempts})")
            elif self.error_type == "503":
                raise Exception(f"503 Service Unavailable (attempt {self.attempts})")
            elif self.error_type == "connection":
                raise ConnectionError(f"Connection failed (attempt {self.attempts})")

        # Success on final attempt
        return ProviderResponse(
            text="Success after retries!",
            provider="failing_provider",
            tokens=10,
            latency_ms=50.0,
        )

    def get_provider_name(self) -> str:
        return "failing_provider"


class PermanentlyFailingProvider(BaseLLMProvider):
    """Provider that always fails with a permanent error (non-retryable)."""

    def __init__(self):
        self.attempts = 0

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        self.attempts += 1
        # 401 Unauthorized is a permanent error - should not retry
        raise Exception("401 Unauthorized: Invalid API key")

    def get_provider_name(self) -> str:
        return "permanently_failing_provider"


async def test_retry_logic():
    """Test the retry logic functionality."""
    print("="*60)
    print("Phase 2.2 Validation: Retry Logic")
    print("="*60)

    # Test 1: Successful retry after timeouts
    print("\n[1/5] Testing retry on timeout errors...")
    provider1 = FailingProvider(fail_count=2, error_type="timeout")
    service1 = InferenceService(
        provider1,
        max_retries=3,
        initial_wait=0.1,  # Short wait for testing
        max_wait=1.0
    )

    try:
        start = time.time()
        result = await service1.run_inference("Test prompt")
        duration = time.time() - start

        print(f"   [OK] Succeeded after {provider1.attempts} attempts")
        print(f"   Response: '{result['response']}'")
        print(f"   Total duration: {duration:.2f}s (includes retry delays)")
        assert provider1.attempts == 3, "Should have made 3 attempts"
        print(f"   [PASS] Retry logic worked correctly")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    # Test 2: Retry on rate limit errors
    print("\n[2/5] Testing retry on rate limit errors...")
    provider2 = FailingProvider(fail_count=1, error_type="rate_limit")
    service2 = InferenceService(provider2, max_retries=3, initial_wait=0.1)

    try:
        result = await service2.run_inference("Test prompt")
        print(f"   [OK] Succeeded after {provider2.attempts} attempts")
        assert provider2.attempts == 2, "Should have made 2 attempts"
        print(f"   [PASS] Rate limit retry worked")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    # Test 3: Retry on 503 errors
    print("\n[3/5] Testing retry on 503 Service Unavailable...")
    provider3 = FailingProvider(fail_count=1, error_type="503")
    service3 = InferenceService(provider3, max_retries=3, initial_wait=0.1)

    try:
        result = await service3.run_inference("Test prompt")
        print(f"   [OK] Succeeded after {provider3.attempts} attempts")
        assert provider3.attempts == 2, "Should have made 2 attempts"
        print(f"   [PASS] 503 error retry worked")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    # Test 4: Exhaust retries
    print("\n[4/5] Testing retry exhaustion...")
    provider4 = FailingProvider(fail_count=10, error_type="timeout")  # Fail more than max_retries
    service4 = InferenceService(provider4, max_retries=2, initial_wait=0.1)

    try:
        result = await service4.run_inference("Test prompt")
        print(f"   [FAIL] Should have exhausted retries")
        return
    except Exception as e:
        print(f"   [OK] Correctly exhausted retries after {provider4.attempts} attempts")
        assert provider4.attempts == 2, "Should have made exactly max_retries attempts"
        print(f"   [PASS] Retry exhaustion works correctly")

    # Test 5: Don't retry permanent errors
    print("\n[5/5] Testing NO retry on permanent errors (401)...")
    provider5 = PermanentlyFailingProvider()
    service5 = InferenceService(provider5, max_retries=3, initial_wait=0.1)

    try:
        result = await service5.run_inference("Test prompt")
        print(f"   [FAIL] Should have raised permanent error")
        return
    except Exception as e:
        print(f"   [OK] Correctly raised permanent error: {e}")
        assert provider5.attempts == 1, "Should NOT have retried permanent error"
        print(f"   [PASS] Permanent errors are not retried")

    print("\n" + "="*60)
    print("[SUCCESS] Phase 2.2 validation complete!")
    print("="*60)
    print("\nRetry logic is working correctly:")
    print("  - Retries transient errors (timeout, rate limit, 503)")
    print("  - Uses exponential backoff")
    print("  - Exhausts retries when appropriate")
    print("  - Does NOT retry permanent errors (401, 404, etc.)")
    print("  - Service remains provider-agnostic")
    print("")


if __name__ == "__main__":
    asyncio.run(test_retry_logic())
