"""
Tests for v2 database models.
"""

import pytest
from datetime import datetime, timezone
from study_query_llm.db.models_v2 import (
    RawCall,
    Group,
    GroupMember,
    CallArtifact,
    EmbeddingVector,
)


def test_raw_call_creation():
    """Test creating a RawCall instance."""
    raw_call = RawCall(
        provider="azure_openai_gpt-4",
        model="gpt-4",
        modality="text",
        status="success",
        request_json={"prompt": "Test prompt"},
        response_json={"text": "Test response"},
        latency_ms=500.0,
        tokens_json={"total": 100},
        metadata_json={"key": "value"},
    )
    
    assert raw_call.provider == "azure_openai_gpt-4"
    assert raw_call.model == "gpt-4"
    assert raw_call.modality == "text"
    assert raw_call.status == "success"
    assert raw_call.request_json == {"prompt": "Test prompt"}
    assert raw_call.response_json == {"text": "Test response"}
    assert raw_call.latency_ms == 500.0
    assert raw_call.tokens_json == {"total": 100}
    assert raw_call.metadata_json == {"key": "value"}


def test_raw_call_with_failure():
    """Test RawCall with failure status."""
    raw_call = RawCall(
        provider="azure_openai_gpt-4",
        modality="text",
        status="failed",
        request_json={"prompt": "Test prompt"},
        response_json=None,
        error_json={"error": "Rate limit exceeded", "code": 429},
    )
    
    assert raw_call.status == "failed"
    assert raw_call.response_json is None
    assert raw_call.error_json == {"error": "Rate limit exceeded", "code": 429}


def test_raw_call_to_dict():
    """Test RawCall to_dict method."""
    created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    raw_call = RawCall(
        id=1,
        provider="azure",
        model="gpt-4",
        modality="text",
        status="success",
        request_json={"prompt": "Test"},
        response_json={"text": "Response"},
        created_at=created_at,
    )
    
    result = raw_call.to_dict()
    
    assert result['id'] == 1
    assert result['provider'] == "azure"
    assert result['model'] == "gpt-4"
    assert result['status'] == "success"
    assert result['created_at'] == created_at.isoformat()


def test_group_creation():
    """Test creating a Group instance."""
    group = Group(
        group_type="batch",
        name="test_batch_1",
        description="Test batch description",
        metadata_json={"key": "value"},
    )
    
    assert group.group_type == "batch"
    assert group.name == "test_batch_1"
    assert group.description == "Test batch description"
    assert group.metadata_json == {"key": "value"}


def test_group_to_dict():
    """Test Group to_dict method."""
    created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    group = Group(
        id=1,
        group_type="experiment",
        name="exp_1",
        description="Experiment 1",
        created_at=created_at,
    )
    
    result = group.to_dict()
    
    assert result['id'] == 1
    assert result['group_type'] == "experiment"
    assert result['name'] == "exp_1"
    assert result['created_at'] == created_at.isoformat()


def test_group_member_creation():
    """Test creating a GroupMember instance."""
    member = GroupMember(
        group_id=1,
        call_id=10,
        position=0,
        role="primary",
    )
    
    assert member.group_id == 1
    assert member.call_id == 10
    assert member.position == 0
    assert member.role == "primary"


def test_call_artifact_creation():
    """Test creating a CallArtifact instance."""
    artifact = CallArtifact(
        call_id=1,
        artifact_type="image",
        uri="s3://bucket/image.jpg",
        content_type="image/jpeg",
        byte_size=1024000,
        metadata_json={"width": 1920, "height": 1080},
    )
    
    assert artifact.call_id == 1
    assert artifact.artifact_type == "image"
    assert artifact.uri == "s3://bucket/image.jpg"
    assert artifact.content_type == "image/jpeg"
    assert artifact.byte_size == 1024000
    assert artifact.metadata_json == {"width": 1920, "height": 1080}


def test_embedding_vector_creation():
    """Test creating an EmbeddingVector instance."""
    vector = EmbeddingVector(
        call_id=1,
        dimension=1536,
        vector=[0.1, 0.2, 0.3] * 512,  # 1536 dimensions
        norm=1.0,
        metadata_json={"model": "text-embedding-3-small"},
    )
    
    assert vector.call_id == 1
    assert vector.dimension == 1536
    assert len(vector.vector) == 1536
    assert vector.norm == 1.0
    assert vector.metadata_json == {"model": "text-embedding-3-small"}


def test_embedding_vector_to_dict():
    """Test EmbeddingVector to_dict method."""
    vector = EmbeddingVector(
        id=1,
        call_id=10,
        dimension=1024,
        vector=[0.5] * 1024,
        norm=0.707,
    )
    
    result = vector.to_dict()
    
    assert result['id'] == 1
    assert result['call_id'] == 10
    assert result['dimension'] == 1024
    assert len(result['vector']) == 1024
    assert result['norm'] == 0.707
