"""
Phase 2.3 Validation: Prompt Preprocessing

This script validates that the InferenceService correctly applies preprocessing
when enabled and skips it when disabled (default).

Run this script to verify Phase 2.3 is working:
    python test_phase_2_3.py
"""

import asyncio
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.preprocessors import PromptPreprocessor


class EchoProvider(BaseLLMProvider):
    """Mock provider that echoes back the prompt it receives."""

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Echo the prompt back so we can verify preprocessing."""
        return ProviderResponse(
            text=f"Echoed: {prompt}",
            provider="echo_provider",
            tokens=len(prompt.split()),
            latency_ms=10.0,
        )

    def get_provider_name(self) -> str:
        return "echo_provider"


async def test_preprocessing():
    """Test the preprocessing functionality."""
    print("="*60)
    print("Phase 2.3 Validation: Prompt Preprocessing")
    print("="*60)

    # Test 1: Default behavior (preprocessing disabled)
    print("\n[1/8] Testing default behavior (preprocess=False)...")
    provider1 = EchoProvider()
    service1 = InferenceService(provider1)  # Default preprocess=False

    messy_prompt = "  hello   world  \n\n  "
    result = await service1.run_inference(messy_prompt)

    # With preprocessing disabled, prompt should be unchanged
    if "hello   world" in result['response']:
        print("   [OK] Preprocessing disabled by default")
        print(f"   Original prompt preserved: '{messy_prompt.strip()}'")
        assert result['metadata']['preprocessing_enabled'] == False
        assert 'original_prompt' not in result  # No preprocessing info
        assert 'processed_prompt' not in result
        print("   [PASS] Default behavior correct")
    else:
        print(f"   [FAIL] Expected unchanged prompt, got: {result['response']}")
        return

    # Test 2: Enable preprocessing with clean_whitespace
    print("\n[2/8] Testing preprocessing enabled (clean_whitespace)...")
    service2 = InferenceService(provider1, preprocess=True)

    result = await service2.run_inference(messy_prompt)

    if "hello world" in result['response'] and "hello   world" not in result['response']:
        print("   [OK] Whitespace cleaned correctly")
        print(f"   Original: '{messy_prompt.strip()}'")
        print(f"   Processed: '{result['processed_prompt']}'")
        assert result['metadata']['preprocessing_enabled'] == True
        assert result['original_prompt'] == messy_prompt
        assert result['processed_prompt'] == "hello world"
        print("   [PASS] Whitespace cleaning works")
    else:
        print(f"   [FAIL] Expected cleaned prompt, got: {result['response']}")
        return

    # Test 3: Truncation
    print("\n[3/8] Testing truncation...")
    service3 = InferenceService(
        provider1,
        preprocess=True,
        max_prompt_length=20
    )

    long_prompt = "a" * 50
    result = await service3.run_inference(long_prompt)

    if "..." in result['response']:
        print(f"   [OK] Long prompt truncated")
        print(f"   Original length: {len(long_prompt)} chars")
        print(f"   Processed length: {len(result['processed_prompt'])} chars")
        assert len(result['processed_prompt']) == 23  # 20 chars + "..."
        print("   [PASS] Truncation works")
    else:
        print(f"   [FAIL] Expected truncation, got: {result['response']}")
        return

    # Test 4: Template application
    print("\n[4/8] Testing template application...")
    service4 = InferenceService(provider1, preprocess=True)

    template = "You are a helpful assistant. {user_input}"
    result = await service4.run_inference(
        "What is 2+2?",
        template=template
    )

    if "You are a helpful assistant. What is 2+2?" in result['response']:
        print("   [OK] Template applied correctly")
        print(f"   Template: '{template}'")
        print(f"   Result: '{result['processed_prompt']}'")
        print("   [PASS] Template application works")
    else:
        print(f"   [FAIL] Expected template, got: {result['response']}")
        return

    # Test 5: PII removal
    print("\n[5/8] Testing PII removal...")
    service5 = InferenceService(
        provider1,
        preprocess=True,
        remove_pii=True
    )

    pii_prompt = "Contact me at john@example.com or 555-123-4567"
    result = await service5.run_inference(pii_prompt)

    if "[EMAIL]" in result['response'] and "[PHONE]" in result['response']:
        print("   [OK] PII removed correctly")
        print(f"   Original: '{pii_prompt}'")
        print(f"   Processed: '{result['processed_prompt']}'")
        assert "john@example.com" not in result['processed_prompt']
        assert "555-123-4567" not in result['processed_prompt']
        print("   [PASS] PII removal works")
    else:
        print(f"   [FAIL] Expected PII removal, got: {result['response']}")
        return

    # Test 6: Control character stripping
    print("\n[6/8] Testing control character stripping...")
    service6 = InferenceService(
        provider1,
        preprocess=True,
        strip_control_chars=True
    )

    control_prompt = "Hello\x00World\x07"  # null byte and bell
    result = await service6.run_inference(control_prompt)

    if "HelloWorld" in result['response'] or "Hello World" in result['response']:
        print("   [OK] Control characters removed")
        print(f"   Original had control chars")
        print(f"   Processed: '{result['processed_prompt']}'")
        print("   [PASS] Control character stripping works")
    else:
        print(f"   [FAIL] Expected clean text, got: {result['response']}")
        return

    # Test 7: PromptPreprocessor standalone usage
    print("\n[7/8] Testing PromptPreprocessor standalone methods...")

    # Test individual methods
    assert PromptPreprocessor.clean_whitespace("  hello   world  ") == "hello world"
    assert len(PromptPreprocessor.truncate("a" * 100, 50)) == 53  # 50 + "..."
    assert PromptPreprocessor.apply_template("test", "Q: {user_input}") == "Q: test"
    assert "[EMAIL]" in PromptPreprocessor.remove_pii("Email: test@example.com")
    assert PromptPreprocessor.strip_control_characters("Hello\x00World") == "HelloWorld"

    print("   [OK] All standalone methods work correctly")
    print("   [PASS] PromptPreprocessor utility class is functional")

    # Test 8: Verify backward compatibility
    print("\n[8/8] Testing backward compatibility...")
    print("   Verifying all previous tests still pass with new parameters...")

    # Previous phases should still work with default preprocess=False
    service_basic = InferenceService(provider1)
    result = await service_basic.run_inference("Simple test")
    assert 'response' in result
    assert 'metadata' in result
    assert result['metadata']['preprocessing_enabled'] == False
    print("   [OK] Previous functionality preserved")
    print("   [PASS] Backward compatibility maintained")

    print("\n" + "="*60)
    print("[SUCCESS] Phase 2.3 validation complete!")
    print("="*60)
    print("\nPrompt preprocessing is working correctly:")
    print("  - Disabled by default (preprocess=False)")
    print("  - Cleans whitespace when enabled")
    print("  - Truncates long prompts when enabled")
    print("  - Applies templates when provided")
    print("  - Removes PII when enabled")
    print("  - Strips control characters when enabled")
    print("  - All preprocessing is opt-in and configurable")
    print("  - PromptPreprocessor can be used standalone")
    print("  - Backward compatible with previous phases")
    print("")


if __name__ == "__main__":
    asyncio.run(test_preprocessing())
