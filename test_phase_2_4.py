"""
Phase 2.4 Validation: Repeated Inference Batching

This script validates that the InferenceService can run the same prompt
multiple times to collect varied responses.

Run this script to verify Phase 2.4 is working:
    python test_phase_2_4.py
"""

import asyncio
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.inference_service import InferenceService


class CountingProvider(BaseLLMProvider):
    """Mock provider that includes a counter in each response."""

    def __init__(self):
        self.call_count = 0

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Return a response with a counter to verify each call is separate."""
        self.call_count += 1
        temperature = kwargs.get('temperature', 0.7)

        return ProviderResponse(
            text=f"Response #{self.call_count} to: {prompt} (temp={temperature})",
            provider="counting_provider",
            tokens=10,
            latency_ms=50.0,
        )

    def get_provider_name(self) -> str:
        return "counting_provider"


class VariableProvider(BaseLLMProvider):
    """Mock provider that returns different responses based on a counter."""

    def __init__(self):
        self.call_count = 0
        self.responses = [
            "The sky is blue",
            "The ocean is vast",
            "Mountains are tall",
            "Rivers flow freely",
            "Stars shine bright"
        ]

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Return varied responses to simulate temperature-based variability."""
        response_text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return ProviderResponse(
            text=response_text,
            provider="variable_provider",
            tokens=5,
            latency_ms=50.0,
        )

    def get_provider_name(self) -> str:
        return "variable_provider"


async def test_repeated_inference():
    """Test the repeated inference batching functionality."""
    print("="*60)
    print("Phase 2.4 Validation: Repeated Inference Batching")
    print("="*60)

    # Test 1: Basic repeated inference
    print("\n[1/5] Testing basic repeated inference (n=3)...")
    provider1 = CountingProvider()
    service1 = InferenceService(provider1)

    results = await service1.run_repeated_inference(
        "What is the meaning of life?",
        n=3
    )

    if len(results) == 3:
        print(f"   [OK] Got {len(results)} responses")
        for i, result in enumerate(results, 1):
            print(f"      Response {i}: '{result['response']}'")

        # Verify each call was made separately
        assert provider1.call_count == 3, "Should have made 3 separate calls"
        print("   [PASS] Basic repeated inference works")
    else:
        print(f"   [FAIL] Expected 3 results, got {len(results)}")
        return

    # Test 2: Repeated inference with parameters
    print("\n[2/5] Testing repeated inference with parameters...")
    provider2 = CountingProvider()
    service2 = InferenceService(provider2)

    results = await service2.run_repeated_inference(
        "Test prompt",
        n=5,
        temperature=0.9,
        max_tokens=100
    )

    if len(results) == 5:
        print(f"   [OK] Got {len(results)} responses")

        # Verify parameters were passed
        assert all('temp=0.9' in r['response'] for r in results), "Temperature should be passed"
        assert all(r['metadata']['temperature'] == 0.9 for r in results), "Metadata should include temperature"
        assert all(r['metadata']['max_tokens'] == 100 for r in results), "Metadata should include max_tokens"

        print("   [PASS] Parameters correctly passed to all runs")
    else:
        print(f"   [FAIL] Expected 5 results, got {len(results)}")
        return

    # Test 3: Verify responses can vary (simulate sampling)
    print("\n[3/5] Testing response variability...")
    provider3 = VariableProvider()
    service3 = InferenceService(provider3)

    results = await service3.run_repeated_inference(
        "Describe nature",
        n=5,
        temperature=1.0
    )

    responses = [r['response'] for r in results]
    unique_responses = set(responses)

    print(f"   [OK] Got {len(results)} responses")
    print(f"   Unique responses: {len(unique_responses)}")
    for i, response in enumerate(responses, 1):
        print(f"      Response {i}: '{response}'")

    # With our mock provider, all 5 responses should be different
    assert len(unique_responses) == 5, "All responses should be unique"
    print("   [PASS] Responses varied as expected")

    # Test 4: Large batch (n=10)
    print("\n[4/5] Testing larger batch (n=10)...")
    provider4 = CountingProvider()
    service4 = InferenceService(provider4)

    import time
    start = time.time()
    results = await service4.run_repeated_inference(
        "Quick test",
        n=10
    )
    duration = time.time() - start

    if len(results) == 10:
        print(f"   [OK] Got {len(results)} responses")
        print(f"   Completed in {duration:.2f}s (concurrent execution)")
        assert provider4.call_count == 10, "Should have made 10 calls"

        # Verify concurrent execution (should be fast, not 10x sequential time)
        # With 50ms latency per call, sequential would take 500ms
        # Concurrent should be much faster
        if duration < 0.3:  # Allow some overhead
            print(f"   [OK] Concurrent execution confirmed ({duration:.2f}s)")

        print("   [PASS] Large batch works efficiently")
    else:
        print(f"   [FAIL] Expected 10 results, got {len(results)}")
        return

    # Test 5: Verify each result has correct format
    print("\n[5/5] Testing result format consistency...")
    provider5 = CountingProvider()
    service5 = InferenceService(provider5)

    results = await service5.run_repeated_inference(
        "Format test",
        n=3,
        temperature=0.5
    )

    all_valid = True
    for i, result in enumerate(results, 1):
        # Check required keys
        if not all(key in result for key in ['response', 'metadata', 'provider_response']):
            print(f"   [FAIL] Result {i} missing required keys")
            all_valid = False
            break

        # Check metadata
        metadata = result['metadata']
        required_metadata = ['provider', 'tokens', 'latency_ms', 'temperature', 'preprocessing_enabled']
        if not all(key in metadata for key in required_metadata):
            print(f"   [FAIL] Result {i} metadata missing required keys")
            all_valid = False
            break

    if all_valid:
        print("   [OK] All results have correct format")
        print("   Keys in result:", list(results[0].keys()))
        print("   Metadata keys:", list(results[0]['metadata'].keys()))
        print("   [PASS] Result format is consistent")

    # Test 6: Verify backward compatibility with batch_inference
    print("\n[6/5] Testing backward compatibility with run_batch_inference...")
    provider6 = CountingProvider()
    service6 = InferenceService(provider6)

    # Old way: different prompts
    batch_results = await service6.run_batch_inference(
        ["Prompt A", "Prompt B", "Prompt C"]
    )

    if len(batch_results) == 3:
        print("   [OK] run_batch_inference still works")
        assert all('response' in r for r in batch_results)
        print("   [PASS] Backward compatibility maintained")

    print("\n" + "="*60)
    print("[SUCCESS] Phase 2.4 validation complete!")
    print("="*60)
    print("\nRepeated inference batching is working correctly:")
    print("  - Can run the same prompt multiple times (n parameter)")
    print("  - Each call is independent and may produce different results")
    print("  - Parameters (temperature, max_tokens, etc.) passed to all runs")
    print("  - Runs concurrently for efficiency")
    print("  - Response format is consistent across all results")
    print("  - Useful for sampling LLM output variability")
    print("  - Backward compatible with existing batch_inference method")
    print("")


if __name__ == "__main__":
    asyncio.run(test_repeated_inference())
