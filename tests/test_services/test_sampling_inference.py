"""
Tests for Phase 2.4 - Sampling Inference.

Tests that InferenceService can run the same prompt multiple times
to collect varied responses (samples), useful for sampling LLM output variability.
"""

import pytest
import time
from study_query_llm.services.inference_service import InferenceService


@pytest.mark.asyncio
async def test_basic_sampling_inference(counting_provider):
    """Test basic sampling inference with same prompt."""
    service = InferenceService(counting_provider)
    
    results = await service.run_sampling_inference(
        "What is the meaning of life?",
        n=3
    )
    
    assert len(results) == 3
    assert counting_provider.call_count == 3, "Should have made 3 separate calls"
    
    # Verify each response has a different counter
    for i, result in enumerate(results, 1):
        assert f"Response #{i}" in result['response']


@pytest.mark.asyncio
async def test_sampling_inference_with_parameters(counting_provider):
    """Test sampling inference with custom parameters."""
    service = InferenceService(counting_provider)
    
    results = await service.run_sampling_inference(
        "Test prompt",
        n=5,
        temperature=0.9,
        max_tokens=100
    )
    
    assert len(results) == 5
    
    # Verify parameters were passed to all runs
    for result in results:
        assert 'temp=0.9' in result['response']
        assert result['metadata']['temperature'] == 0.9
        assert result['metadata']['max_tokens'] == 100


@pytest.mark.asyncio
async def test_response_variability(variable_provider):
    """Test that sampling inference can produce varied responses."""
    service = InferenceService(variable_provider)
    
    results = await service.run_sampling_inference(
        "Describe nature",
        n=5,
        temperature=1.0
    )
    
    responses = [r['response'] for r in results]
    unique_responses = set(responses)
    
    # With our mock provider, all 5 responses should be different
    assert len(unique_responses) == 5, "All responses should be unique"
    assert len(results) == 5


@pytest.mark.asyncio
async def test_large_sampling_concurrent_execution(counting_provider):
    """Test that large sampling runs execute concurrently."""
    service = InferenceService(counting_provider)
    
    start = time.time()
    results = await service.run_sampling_inference(
        "Quick test",
        n=10
    )
    duration = time.time() - start
    
    assert len(results) == 10
    assert counting_provider.call_count == 10, "Should have made 10 calls"
    
    # Verify concurrent execution (should be fast, not 10x sequential time)
    # With 50ms latency per call, sequential would take ~500ms
    # Concurrent should be much faster
    assert duration < 0.3, f"Should execute concurrently, took {duration:.2f}s"


@pytest.mark.asyncio
async def test_result_format_consistency(counting_provider):
    """Test that all results have consistent format."""
    service = InferenceService(counting_provider)
    
    results = await service.run_sampling_inference(
        "Format test",
        n=3,
        temperature=0.5
    )
    
    assert len(results) == 3
    
    # Check required keys in each result
    required_keys = ['response', 'metadata', 'provider_response']
    required_metadata = ['provider', 'tokens', 'latency_ms', 'temperature', 'preprocessing_enabled']
    
    for result in results:
        assert all(key in result for key in required_keys), "Result missing required keys"
        
        metadata = result['metadata']
        assert all(key in metadata for key in required_metadata), "Metadata missing required keys"


@pytest.mark.asyncio
async def test_backward_compatibility_with_batch_inference(counting_provider):
    """Test that run_batch_inference still works (backward compatibility)."""
    service = InferenceService(counting_provider)
    
    # Old way: different prompts
    batch_results = await service.run_batch_inference(
        ["Prompt A", "Prompt B", "Prompt C"]
    )
    
    assert len(batch_results) == 3
    assert all('response' in r for r in batch_results)
    assert counting_provider.call_count == 3


@pytest.mark.asyncio
async def test_sampling_vs_batch_difference(counting_provider):
    """Test that sampling inference and batch inference behave differently."""
    service = InferenceService(counting_provider)
    
    # Sampling: same prompt multiple times
    sampling_results = await service.run_sampling_inference("Same prompt", n=3)
    
    # Reset counter
    counting_provider.call_count = 0
    
    # Batch: different prompts
    batch_results = await service.run_batch_inference(
        ["Prompt 1", "Prompt 2", "Prompt 3"]
    )
    
    # Both should have 3 results
    assert len(sampling_results) == 3
    assert len(batch_results) == 3
    
    # But sampling should have same prompt in all responses
    assert all("Same prompt" in r['response'] for r in sampling_results)
    
    # Batch should have different prompts
    assert "Prompt 1" in batch_results[0]['response']
    assert "Prompt 2" in batch_results[1]['response']
    assert "Prompt 3" in batch_results[2]['response']


@pytest.mark.asyncio
async def test_sampling_inference_with_different_n_values(counting_provider):
    """Test sampling inference with various n values."""
    service = InferenceService(counting_provider)
    
    for n in [1, 2, 5, 10]:
        counting_provider.call_count = 0
        results = await service.run_sampling_inference("Test", n=n)
        
        assert len(results) == n
        assert counting_provider.call_count == n

