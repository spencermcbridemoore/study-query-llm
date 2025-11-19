"""
Tests for Phase 2.1 - Basic Inference Service.

Tests the InferenceService wrapper around providers.
"""

import pytest
from study_query_llm.services.inference_service import InferenceService


@pytest.mark.asyncio
async def test_basic_inference_service(mock_provider):
    """Test basic inference service functionality."""
    service = InferenceService(mock_provider)
    
    result = await service.run_inference("What is 5+5?")
    
    assert 'response' in result
    assert 'metadata' in result
    assert 'provider_response' in result
    assert result['response'] is not None
    assert result['metadata']['provider'] == "mock"
    assert result['metadata']['tokens'] > 0
    assert result['metadata']['latency_ms'] > 0


@pytest.mark.asyncio
async def test_inference_with_parameters(mock_provider):
    """Test inference service with custom parameters."""
    service = InferenceService(mock_provider)
    
    result = await service.run_inference(
        "Test prompt",
        temperature=0.5,
        max_tokens=100
    )
    
    assert result['metadata']['temperature'] == 0.5
    assert result['metadata']['max_tokens'] == 100


@pytest.mark.asyncio
async def test_batch_inference(mock_provider):
    """Test batch inference with multiple prompts."""
    service = InferenceService(mock_provider)
    
    prompts = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
    ]
    
    results = await service.run_batch_inference(prompts)
    
    assert len(results) == len(prompts)
    for i, result in enumerate(results):
        assert 'response' in result
        assert 'metadata' in result
        assert result['response'] is not None


@pytest.mark.asyncio
async def test_sampling_inference(mock_provider):
    """Test sampling inference (same prompt multiple times)."""
    service = InferenceService(mock_provider)
    
    results = await service.run_sampling_inference(
        "What is the meaning of life?",
        n=3
    )
    
    assert len(results) == 3
    for result in results:
        assert 'response' in result
        assert 'metadata' in result
        assert result['response'] is not None


@pytest.mark.asyncio
async def test_service_context_manager(mock_provider):
    """Test that service works as async context manager."""
    async with InferenceService(mock_provider) as service:
        result = await service.run_inference("Test")
    
    assert result['response'] is not None


@pytest.mark.asyncio
async def test_service_get_provider_name(mock_provider):
    """Test that service exposes provider name."""
    service = InferenceService(mock_provider)
    
    assert service.get_provider_name() == "mock"

