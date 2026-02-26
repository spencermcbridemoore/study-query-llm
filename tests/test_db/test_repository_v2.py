"""
Tests for RawCallRepository (v2 schema).
"""

import pytest
import os
from datetime import datetime, timedelta, timezone
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import RawCall, Group, GroupMember


@pytest.fixture
def v2_db_connection():
    """
    Fixture for v2 database connection.
    
    Tries Postgres from DATABASE_URL env var, falls back to SQLite in-memory
    for basic tests (pgvector features will be skipped).
    """
    v2_db_url = os.environ.get("DATABASE_URL")
    
    if v2_db_url and v2_db_url.startswith("postgresql"):
        # Use Postgres if available
        try:
            db = DatabaseConnectionV2(v2_db_url, enable_pgvector=False)
            db.init_db()
            yield db
            db.drop_all_tables()
        except Exception as e:
            pytest.skip(f"Postgres not available: {str(e)}")
    else:
        # Fall back to SQLite for basic tests
        # Note: Some Postgres-specific features won't work, but basic CRUD will
        db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
        db.init_db()
        yield db
        db.drop_all_tables()


@pytest.fixture
def v2_repository(v2_db_connection):
    """Fixture for v2 repository with session."""
    with v2_db_connection.session_scope() as session:
        yield RawCallRepository(session)


def test_insert_raw_call(v2_db_connection):
    """Test inserting a single raw call."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        call_id = repo.insert_raw_call(
            provider="azure_openai_gpt-4",
            request_json={"prompt": "What is 2+2?"},
            response_json={"text": "4"},
            status="success",
            latency_ms=250.5,
            tokens_json={"total": 10},
        )
        
        assert call_id is not None
        assert call_id > 0


def test_insert_raw_call_with_failure(v2_db_connection):
    """Test inserting a failed raw call."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        call_id = repo.insert_raw_call(
            provider="azure_openai_gpt-4",
            request_json={"prompt": "Test"},
            status="failed",
            error_json={"error": "Rate limit", "code": 429},
        )
        
        assert call_id > 0
        
        # Verify it was stored correctly
        call = repo.get_raw_call_by_id(call_id)
        assert call.status == "failed"
        assert call.response_json is None
        assert call.error_json == {"error": "Rate limit", "code": 429}


def test_get_raw_call_by_id(v2_db_connection):
    """Test retrieving raw call by ID."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Insert
        call_id = repo.insert_raw_call(
            provider="test",
            request_json={"prompt": "Test prompt"},
            response_json={"text": "Test response"},
        )
        
        # Retrieve
        call = repo.get_raw_call_by_id(call_id)
        
        assert call is not None
        assert call.id == call_id
        assert call.provider == "test"
        assert call.request_json == {"prompt": "Test prompt"}
        assert call.response_json == {"text": "Test response"}


def test_query_raw_calls_by_provider(v2_db_connection):
    """Test querying raw calls by provider."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Insert test data
        repo.insert_raw_call("azure", {"prompt": "P1"}, response_json={"text": "R1"})
        repo.insert_raw_call("azure", {"prompt": "P2"}, response_json={"text": "R2"})
        repo.insert_raw_call("openai", {"prompt": "P3"}, response_json={"text": "R3"})
        
        # Query by provider
        azure_calls = repo.query_raw_calls(provider="azure")
        
        assert len(azure_calls) == 2
        assert all(call.provider == "azure" for call in azure_calls)


def test_query_raw_calls_by_status(v2_db_connection):
    """Test querying raw calls by status."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Insert test data
        repo.insert_raw_call("test", {"prompt": "P1"}, status="success", response_json={"text": "R1"})
        repo.insert_raw_call("test", {"prompt": "P2"}, status="failed", error_json={"error": "E1"})
        repo.insert_raw_call("test", {"prompt": "P3"}, status="success", response_json={"text": "R3"})
        
        # Query by status
        failed_calls = repo.query_raw_calls(status="failed")
        
        assert len(failed_calls) == 1
        assert failed_calls[0].status == "failed"


def test_query_raw_calls_by_modality(v2_db_connection):
    """Test querying raw calls by modality."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Insert test data
        repo.insert_raw_call("test", {"prompt": "P1"}, modality="text", response_json={"text": "R1"})
        repo.insert_raw_call("test", {"input": "I1"}, modality="embedding", response_json={"vector": [0.1, 0.2]})
        repo.insert_raw_call("test", {"prompt": "P2"}, modality="text", response_json={"text": "R2"})
        
        # Query by modality
        text_calls = repo.query_raw_calls(modality="text")
        
        assert len(text_calls) == 2
        assert all(call.modality == "text" for call in text_calls)


def test_batch_insert_raw_calls(v2_db_connection):
    """Test batch inserting raw calls."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        calls_data = [
            {
                "provider": "test",
                "request_json": {"prompt": f"Prompt {i}"},
                "response_json": {"text": f"Response {i}"},
            }
            for i in range(5)
        ]
        
        call_ids = repo.batch_insert_raw_calls(calls_data)
        
        assert len(call_ids) == 5
        assert all(cid > 0 for cid in call_ids)


def test_create_group(v2_db_connection):
    """Test creating a group."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        group_id = repo.create_group(
            group_type="batch",
            name="test_batch_1",
            description="Test batch",
        )
        
        assert group_id > 0
        
        # Verify it was created
        group = repo.get_group_by_id(group_id)
        assert group is not None
        assert group.group_type == "batch"
        assert group.name == "test_batch_1"


def test_add_call_to_group(v2_db_connection):
    """Test adding a call to a group."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Create call and group
        call_id = repo.insert_raw_call("test", {"prompt": "P1"}, response_json={"text": "R1"})
        group_id = repo.create_group("batch", "batch_1")
        
        # Add call to group
        member_id = repo.add_call_to_group(group_id, call_id, position=0)
        
        assert member_id > 0
        
        # Verify membership
        calls_in_group = repo.get_calls_in_group(group_id)
        assert len(calls_in_group) == 1
        assert calls_in_group[0].id == call_id


def test_get_calls_in_group(v2_db_connection):
    """Test getting all calls in a group."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Create group
        group_id = repo.create_group("batch", "batch_1")
        
        # Create calls and add to group
        call_ids = []
        for i in range(3):
            call_id = repo.insert_raw_call("test", {"prompt": f"P{i}"}, response_json={"text": f"R{i}"})
            call_ids.append(call_id)
            repo.add_call_to_group(group_id, call_id, position=i)
        
        # Get calls in group
        calls = repo.get_calls_in_group(group_id)
        
        assert len(calls) == 3
        assert set(call.id for call in calls) == set(call_ids)


def test_get_groups_for_call(v2_db_connection):
    """Test getting all groups that contain a call."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Create call
        call_id = repo.insert_raw_call("test", {"prompt": "P1"}, response_json={"text": "R1"})
        
        # Create groups and add call to them
        group1_id = repo.create_group("batch", "batch_1")
        group2_id = repo.create_group("experiment", "exp_1")
        
        repo.add_call_to_group(group1_id, call_id)
        repo.add_call_to_group(group2_id, call_id)
        
        # Get groups for call
        groups = repo.get_groups_for_call(call_id)
        
        assert len(groups) == 2
        assert set(g.id for g in groups) == {group1_id, group2_id}


def test_get_or_create_defective_group(v2_db_connection):
    """Test getting or creating the defective_data group."""
    from study_query_llm.services.data_quality_service import DataQualityService
    
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        quality_service = DataQualityService(repo)
        
        # First call should create the group
        group_id1 = quality_service.get_or_create_defective_group()
        assert group_id1 > 0
        
        # Second call should return the same group
        group_id2 = quality_service.get_or_create_defective_group()
        assert group_id2 == group_id1
        
        # Verify group properties
        group = repo.get_group_by_id(group_id1)
        assert group is not None
        assert group.group_type == "label"
        assert group.name == "defective_data"


def test_is_call_defective(v2_db_connection):
    """Test checking if a call is marked as defective."""
    from study_query_llm.services.data_quality_service import DataQualityService
    
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        quality_service = DataQualityService(repo)
        
        # Create test calls
        call_id1 = repo.insert_raw_call("test", {"prompt": "P1"}, response_json={"text": "R1"})
        call_id2 = repo.insert_raw_call("test", {"prompt": "P2"}, response_json={"text": "R2"})
        
        # Initially, calls should not be defective
        assert not quality_service.is_call_defective(call_id1)
        assert not quality_service.is_call_defective(call_id2)
        
        # Mark call_id1 as defective
        group_id = quality_service.get_or_create_defective_group()
        repo.add_call_to_group(group_id, call_id1, role="bogus_run")
        
        # Now call_id1 should be defective, call_id2 should not
        assert quality_service.is_call_defective(call_id1)
        assert not quality_service.is_call_defective(call_id2)


def test_query_raw_calls_excluding_defective(v2_db_connection):
    """Test querying raw calls with defective exclusion."""
    from study_query_llm.services.data_quality_service import DataQualityService
    
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        quality_service = DataQualityService(repo)
        
        # Create test calls with different modalities
        call_id1 = repo.insert_raw_call("test", {"input": "text1"}, modality="embedding", response_json={"embedding": [0.1, 0.2]})
        call_id2 = repo.insert_raw_call("test", {"input": "text2"}, modality="embedding", response_json={"embedding": [0.3, 0.4]})
        call_id3 = repo.insert_raw_call("test", {"input": "text3"}, modality="text", response_json={"text": "response"})
        
        # Mark call_id2 as defective
        group_id = quality_service.get_or_create_defective_group()
        repo.add_call_to_group(group_id, call_id2, role="bogus_embedding")
        
        # Query all embeddings - should exclude call_id2
        embedding_calls = repo.query_raw_calls_excluding_defective(modality="embedding", limit=100)
        call_ids = [c.id for c in embedding_calls]
        
        assert call_id1 in call_ids
        assert call_id2 not in call_ids  # Should be excluded
        assert len(embedding_calls) == 1
        
        # Query all calls - should still exclude call_id2
        all_calls = repo.query_raw_calls_excluding_defective(limit=100)
        all_call_ids = [c.id for c in all_calls]
        
        assert call_id1 in all_call_ids
        assert call_id2 not in all_call_ids  # Should be excluded
        assert call_id3 in all_call_ids
        assert len(all_calls) == 2


def test_get_provider_stats(v2_db_connection):
    """Test getting provider statistics."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Insert test data
        repo.insert_raw_call("azure", {"prompt": "P1"}, latency_ms=100.0, response_json={"text": "R1"})
        repo.insert_raw_call("azure", {"prompt": "P2"}, latency_ms=200.0, response_json={"text": "R2"})
        repo.insert_raw_call("openai", {"prompt": "P3"}, latency_ms=150.0, response_json={"text": "R3"})
        
        # Get stats
        stats = repo.get_provider_stats()
        
        assert len(stats) == 2
        azure_stats = next(s for s in stats if s['provider'] == 'azure')
        assert azure_stats['count'] == 2
        assert azure_stats['avg_latency_ms'] == 150.0


def test_get_total_count(v2_db_connection):
    """Test getting total count of raw calls."""
    with v2_db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Insert test data
        for i in range(5):
            repo.insert_raw_call("test", {"prompt": f"P{i}"}, response_json={"text": f"R{i}"})
        
        count = repo.get_total_count()
        assert count == 5
