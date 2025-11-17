"""
Test script for Phase 1.1 - Base Provider Interface

This script validates that the base provider interface works correctly
by creating a mock provider and testing the response format.

Run this script to verify Phase 1.1 is working:
    python test_phase_1_1.py
"""

import asyncio
from panel_app.providers.base import BaseLLMProvider, ProviderResponse


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing the base interface."""

    def __init__(self, name: str = "mock"):
        self.name = name

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Simulate an LLM response."""
        # Simulate some processing time
        await asyncio.sleep(0.05)

        # Generate a mock response
        response_text = f"Mock response to: '{prompt[:50]}...'"

        return ProviderResponse(
            text=response_text,
            provider=self.name,
            tokens=len(prompt.split()) + len(response_text.split()),
            latency_ms=50.0,
            metadata={
                "model": "mock-model-v1",
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            },
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return self.name


async def test_basic_completion():
    """Test basic completion functionality."""
    print("=" * 60)
    print("Testing Phase 1.1: Base Provider Interface")
    print("=" * 60)

    # Create a mock provider
    provider = MockProvider("test-provider")
    print(f"\n[OK] Created provider: {provider.get_provider_name()}")

    # Test a simple completion
    print("\n[TEST] Testing simple completion...")
    response = await provider.complete("What is the capital of France?")

    print(f"  Provider: {response.provider}")
    print(f"  Response: {response.text}")
    print(f"  Tokens: {response.tokens}")
    print(f"  Latency: {response.latency_ms}ms")
    print(f"  Metadata: {response.metadata}")

    assert response.provider == "test-provider", "Provider name mismatch"
    assert response.text is not None, "Response text is None"
    assert response.tokens > 0, "Token count should be positive"
    print("  [PASS] Basic completion test passed")

    # Test with kwargs
    print("\n[TEST] Testing completion with parameters...")
    response = await provider.complete(
        "Write a haiku about coding",
        temperature=0.9,
        max_tokens=50,
    )

    assert response.metadata["temperature"] == 0.9, "Temperature not passed correctly"
    assert response.metadata["max_tokens"] == 50, "Max tokens not passed correctly"
    print("  [PASS] Parameter passing test passed")

    # Test ProviderResponse repr
    print(f"\n[TEST] Testing ProviderResponse representation...")
    print(f"  {repr(response)}")
    print("  [PASS] String representation works")

    print("\n" + "=" * 60)
    print("[SUCCESS] All Phase 1.1 tests passed!")
    print("=" * 60)
    print("\nNext step: Phase 1.2 - Implement Azure Provider")
    print("See docs/IMPLEMENTATION_PLAN.md for details")


def test_abstract_enforcement():
    """Test that abstract methods must be implemented."""
    print("\n[TEST] Testing abstract class enforcement...")

    try:
        # This should fail - can't instantiate abstract class
        bad_provider = BaseLLMProvider()
        print("  [FAIL] ERROR: Abstract class was instantiated!")
        return False
    except TypeError as e:
        print(f"  [PASS] Abstract class correctly prevents instantiation")
        return True


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_basic_completion())

    # Test abstract enforcement (synchronous)
    test_abstract_enforcement()

    print("\n[SUCCESS] Phase 1.1 validation complete!\n")
