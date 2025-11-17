"""
Phase 2.1 Validation: Basic Inference Service

This script validates that the InferenceService works correctly as a wrapper
around LLM providers, adding business logic on top of the provider layer.

Run this script to verify Phase 2.1 is working:
    python test_phase_2_1.py
"""

import asyncio
from study_query_llm.providers.azure_provider import AzureOpenAIProvider
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.config import config


async def test_basic_inference_service():
    """Test the basic inference service functionality."""
    print("="*60)
    print("Phase 2.1 Validation: Basic Inference Service")
    print("="*60)

    # Setup
    print("\n[1/6] Loading Azure provider configuration...")
    try:
        azure_config = config.get_provider_config("azure")
        print(f"   [OK] Config loaded for: {azure_config.deployment_name}")
    except ValueError as e:
        print(f"   [FAIL] {e}")
        return

    print("\n[2/6] Creating Azure provider...")
    provider = AzureOpenAIProvider(azure_config)
    print(f"   [OK] Provider created: {provider.get_provider_name()}")

    print("\n[3/6] Creating InferenceService...")
    service = InferenceService(provider)
    print(f"   [OK] Service created with provider: {service.get_provider_name()}")

    # Test single inference
    print("\n[4/6] Testing single inference...")
    test_prompt = "What is 5+5? Answer with just the number."

    try:
        async with service:  # Use context manager for cleanup
            result = await service.run_inference(
                test_prompt,
                temperature=0.0,
                max_tokens=10
            )

            print(f"   [OK] Inference successful!")
            print(f"   Prompt: '{test_prompt}'")
            print(f"   Response: '{result['response']}'")
            print(f"   Provider: {result['metadata']['provider']}")
            print(f"   Tokens: {result['metadata']['tokens']}")
            print(f"   Latency: {result['metadata']['latency_ms']:.2f}ms")
            print(f"   Temperature: {result['metadata']['temperature']}")
            print(f"   Max tokens: {result['metadata'].get('max_tokens', 'default')}")

    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    # Test response format
    print("\n[5/6] Validating response format...")
    try:
        assert 'response' in result, "Result should have 'response' key"
        assert 'metadata' in result, "Result should have 'metadata' key"
        assert 'provider_response' in result, "Result should have 'provider_response' key"

        assert result['response'], "Response text should not be empty"
        assert result['metadata']['provider'], "Provider should be specified"
        assert result['metadata']['tokens'] > 0, "Token count should be positive"
        assert result['metadata']['latency_ms'] > 0, "Latency should be positive"

        print("   [OK] Response format is valid")
        print(f"   Keys: {list(result.keys())}")
        print(f"   Metadata keys: {list(result['metadata'].keys())}")
    except AssertionError as e:
        print(f"   [FAIL] {e}")
        return

    # Test batch inference
    print("\n[6/6] Testing batch inference...")
    batch_prompts = [
        "What is 1+1? Just the number.",
        "What is 2+2? Just the number.",
        "What is 3+3? Just the number.",
    ]

    try:
        async with InferenceService(AzureOpenAIProvider(azure_config)) as batch_service:
            results = await batch_service.run_batch_inference(
                batch_prompts,
                temperature=0.0,
                max_tokens=5
            )

            print(f"   [OK] Batch inference successful!")
            print(f"   Processed {len(results)} prompts")
            for i, result in enumerate(results):
                print(f"   [{i+1}] '{batch_prompts[i]}' -> '{result['response'].strip()}'")

            # Verify all succeeded
            assert len(results) == len(batch_prompts), "Should process all prompts"
            for result in results:
                assert result['response'], "All results should have responses"

            print(f"   [OK] All batch results valid")

    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    print("\n" + "="*60)
    print("[SUCCESS] Phase 2.1 validation complete!")
    print("="*60)
    print("\nInferenceService is working correctly.")
    print("\nKey features demonstrated:")
    print("  - Service wraps provider with standardized interface")
    print("  - Returns consistent dict format with response + metadata")
    print("  - Supports single and batch inference")
    print("  - Works with async context managers")
    print("  - Ready for Phase 2.2 (retry logic) and 2.3 (preprocessing)")
    print("")


if __name__ == "__main__":
    asyncio.run(test_basic_inference_service())
