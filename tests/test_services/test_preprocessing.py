"""
Tests for Phase 2.3 - Prompt Preprocessing.

Tests the preprocessing functionality in InferenceService.
"""

import pytest
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.preprocessors import PromptPreprocessor


@pytest.mark.asyncio
async def test_preprocessing_disabled_by_default(echo_provider):
    """Test that preprocessing is disabled by default."""
    service = InferenceService(echo_provider)  # Default preprocess=False
    
    messy_prompt = "  hello   world  \n\n  "
    result = await service.run_inference(messy_prompt)
    
    # With preprocessing disabled, prompt should be unchanged
    assert "hello   world" in result['response']
    assert result['metadata']['preprocessing_enabled'] is False
    assert 'original_prompt' not in result
    assert 'processed_prompt' not in result


@pytest.mark.asyncio
async def test_whitespace_cleaning(echo_provider):
    """Test that whitespace is cleaned when preprocessing is enabled."""
    service = InferenceService(echo_provider, preprocess=True)
    
    messy_prompt = "  hello   world  \n\n  "
    result = await service.run_inference(messy_prompt)
    
    assert "hello world" in result['response']
    assert "hello   world" not in result['response']
    assert result['metadata']['preprocessing_enabled'] is True
    assert result['original_prompt'] == messy_prompt
    assert result['processed_prompt'] == "hello world"


@pytest.mark.asyncio
async def test_truncation(echo_provider):
    """Test that long prompts are truncated."""
    service = InferenceService(
        echo_provider,
        preprocess=True,
        max_prompt_length=20
    )
    
    long_prompt = "a" * 50
    result = await service.run_inference(long_prompt)
    
    assert "..." in result['response']
    assert len(result['processed_prompt']) == 23  # 20 chars + "..."


@pytest.mark.asyncio
async def test_template_application(echo_provider):
    """Test that templates are applied correctly."""
    service = InferenceService(echo_provider, preprocess=True)
    
    template = "You are a helpful assistant. {user_input}"
    result = await service.run_inference(
        "What is 2+2?",
        template=template
    )
    
    assert "You are a helpful assistant. What is 2+2?" in result['response']
    assert result['processed_prompt'] == "You are a helpful assistant. What is 2+2?"


@pytest.mark.asyncio
async def test_pii_removal(echo_provider):
    """Test that PII is removed when enabled."""
    service = InferenceService(
        echo_provider,
        preprocess=True,
        remove_pii=True
    )
    
    pii_prompt = "Contact me at john@example.com or 555-123-4567"
    result = await service.run_inference(pii_prompt)
    
    assert "[EMAIL]" in result['response']
    assert "[PHONE]" in result['response']
    assert "john@example.com" not in result['processed_prompt']
    assert "555-123-4567" not in result['processed_prompt']


@pytest.mark.asyncio
async def test_control_character_stripping(echo_provider):
    """Test that control characters are stripped."""
    service = InferenceService(
        echo_provider,
        preprocess=True,
        strip_control_chars=True
    )
    
    control_prompt = "Hello\x00World\x07"  # null byte and bell
    result = await service.run_inference(control_prompt)
    
    # Should not contain control characters
    assert "\x00" not in result['processed_prompt']
    assert "\x07" not in result['processed_prompt']


def test_preprocessor_standalone_methods():
    """Test PromptPreprocessor standalone methods."""
    # Test whitespace cleaning
    assert PromptPreprocessor.clean_whitespace("  hello   world  ") == "hello world"
    
    # Test truncation
    assert len(PromptPreprocessor.truncate("a" * 100, 50)) == 53  # 50 + "..."
    
    # Test template
    assert PromptPreprocessor.apply_template("test", "Q: {user_input}") == "Q: test"
    
    # Test PII removal
    assert "[EMAIL]" in PromptPreprocessor.remove_pii("Email: test@example.com")
    
    # Test control character stripping
    assert PromptPreprocessor.strip_control_characters("Hello\x00World") == "HelloWorld"


def test_preprocessor_preprocess_method():
    """Test the combined preprocess method."""
    prompt = "  Contact: test@example.com  "
    
    result = PromptPreprocessor.preprocess(
        prompt,
        clean_whitespace=True,
        remove_pii=True
    )
    
    assert result == "Contact: [EMAIL]"
    assert "test@example.com" not in result


@pytest.mark.parametrize("clean_whitespace,expected", [
    (True, "hello world"),
    (False, "  hello   world  "),
])
def test_preprocessor_clean_whitespace_param(clean_whitespace, expected):
    """Test preprocess method with different clean_whitespace settings."""
    prompt = "  hello   world  "
    result = PromptPreprocessor.preprocess(
        prompt,
        clean_whitespace=clean_whitespace
    )
    assert result == expected

