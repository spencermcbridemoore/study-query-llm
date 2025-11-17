"""
Phase 1.2 Validation: Azure OpenAI Provider

This script validates that the Azure OpenAI provider works correctly
with your configured credentials and deployment.

Run this script to verify Phase 1.2 is working:
    python test_phase_1_2.py
"""

import asyncio
from study_query_llm.providers.azure_provider import AzureOpenAIProvider
from study_query_llm.config import config


async def test_azure_provider():
    """Test the Azure OpenAI provider with a simple completion."""
    print("="*60)
    print("Phase 1.2 Validation: Azure OpenAI Provider")
    print("="*60)

    # Load Azure configuration
    print("\n[1/5] Loading Azure configuration from .env...")
    try:
        azure_config = config.get_provider_config("azure")
        print(f"   [OK] Loaded config for deployment: {azure_config.deployment_name}")
        print(f"   [OK] Endpoint: {azure_config.endpoint}")
        print(f"   [OK] API Version: {azure_config.api_version}")
    except ValueError as e:
        print(f"   [FAIL] {e}")
        print("\nPlease ensure your .env file has:")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_DEPLOYMENT")
        print("  - AZURE_OPENAI_API_VERSION")
        return

    # Initialize provider
    print("\n[2/5] Initializing Azure OpenAI provider...")
    try:
        provider = AzureOpenAIProvider(azure_config)
        print(f"   [OK] Provider initialized: {provider.get_provider_name()}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    # Test basic completion
    print("\n[3/5] Testing basic completion...")
    test_prompt = "What is 2+2? Answer in one word."

    try:
        async with provider:  # Use context manager for automatic cleanup
            response = await provider.complete(test_prompt)

            print(f"   [OK] Completion successful!")
            print(f"   Prompt: '{test_prompt}'")
            print(f"   Response: '{response.text}'")
            print(f"   Provider: {response.provider}")
            print(f"   Latency: {response.latency_ms:.2f}ms")
            print(f"   Tokens: {response.tokens}")

    except Exception as e:
        print(f"   [FAIL] Completion failed: {e}")
        print("\nPossible issues:")
        print("  - Check that your API key is valid")
        print("  - Verify deployment name matches Azure portal")
        print("  - Ensure endpoint URL is correct")
        print("  - Check network connectivity")
        return

    # Validate response format
    print("\n[4/5] Validating response format...")
    try:
        assert response.text, "Response text should not be empty"
        assert response.provider.startswith("azure_openai"), "Provider name should start with 'azure_openai'"
        assert response.tokens is not None and response.tokens > 0, "Token count should be positive"
        assert response.latency_ms is not None and response.latency_ms > 0, "Latency should be positive"
        assert response.metadata, "Metadata should not be empty"
        assert "model" in response.metadata, "Metadata should include model"
        assert "finish_reason" in response.metadata, "Metadata should include finish_reason"
        print("   [OK] Response format is valid")
        print(f"   Model: {response.metadata.get('model')}")
        print(f"   Finish reason: {response.metadata.get('finish_reason')}")
        print(f"   Prompt tokens: {response.metadata.get('prompt_tokens')}")
        print(f"   Completion tokens: {response.metadata.get('completion_tokens')}")
    except AssertionError as e:
        print(f"   [FAIL] {e}")
        return

    # Test with parameters
    print("\n[5/5] Testing with custom parameters...")
    try:
        async with AzureOpenAIProvider(azure_config) as provider2:
            response2 = await provider2.complete(
                "Say 'hello' in French.",
                temperature=0.5,
                max_tokens=10
            )
            print(f"   [OK] Completion with custom params successful!")
            print(f"   Response: '{response2.text}'")
            print(f"   Temperature: {response2.metadata.get('temperature')}")
            print(f"   Max tokens: {response2.metadata.get('max_tokens')}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return

    print("\n" + "="*60)
    print("[SUCCESS] Phase 1.2 validation complete!")
    print("="*60)
    print("\nAzure OpenAI provider is working correctly.")
    print("You can now use it in your application:")
    print("")
    print("  from study_query_llm.providers import AzureOpenAIProvider")
    print("  from study_query_llm.config import config")
    print("")
    print("  azure_config = config.get_provider_config('azure')")
    print("  provider = AzureOpenAIProvider(azure_config)")
    print("  response = await provider.complete('Your prompt here')")
    print("")


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_azure_provider())
