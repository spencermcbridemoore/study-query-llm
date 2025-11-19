"""
Tests for database models.
"""

import pytest
from datetime import datetime
from study_query_llm.db.models import InferenceRun


def test_inference_run_creation():
    """Test creating an InferenceRun instance."""
    inference = InferenceRun(
        prompt="Test prompt",
        response="Test response",
        provider="azure_openai_gpt-4",
        tokens=100,
        latency_ms=500.0,
        metadata_json={"key": "value"}
    )
    
    assert inference.prompt == "Test prompt"
    assert inference.response == "Test response"
    assert inference.provider == "azure_openai_gpt-4"
    assert inference.tokens == 100
    assert inference.latency_ms == 500.0
    assert inference.metadata_json == {"key": "value"}
    # created_at is set by database on insert, not on object creation


def test_inference_run_optional_fields():
    """Test InferenceRun with optional fields."""
    inference = InferenceRun(
        prompt="Test",
        response="Response",
        provider="test"
    )
    
    assert inference.tokens is None
    assert inference.latency_ms is None
    assert inference.metadata_json is None
    # created_at is set by database on insert, not on object creation


def test_inference_run_repr():
    """Test InferenceRun string representation."""
    inference = InferenceRun(
        id=1,
        prompt="Test",
        response="Response",
        provider="azure",
        created_at=datetime(2024, 1, 1, 12, 0, 0)
    )
    
    repr_str = repr(inference)
    assert "InferenceRun" in repr_str
    assert "id=1" in repr_str
    assert "provider=azure" in repr_str


def test_inference_run_to_dict():
    """Test InferenceRun to_dict method."""
    created_at = datetime(2024, 1, 1, 12, 0, 0)
    inference = InferenceRun(
        id=1,
        prompt="Test prompt",
        response="Test response",
        provider="azure",
        tokens=100,
        latency_ms=500.0,
        metadata_json={"key": "value"},
        created_at=created_at
    )
    
    result = inference.to_dict()
    
    assert result['id'] == 1
    assert result['prompt'] == "Test prompt"
    assert result['response'] == "Test response"
    assert result['provider'] == "azure"
    assert result['tokens'] == 100
    assert result['latency_ms'] == 500.0
    assert result['metadata'] == {"key": "value"}
    assert result['created_at'] == created_at.isoformat()

