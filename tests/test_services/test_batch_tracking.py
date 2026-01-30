"""
Tests for batch tracking in InferenceService (v2 schema).

Tests that run_sampling_inference and run_batch_inference properly
generate and use batch_id for grouping runs via v2 Group/GroupMember tables.
"""

import pytest
import uuid
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.mark.asyncio
async def test_sampling_inference_generates_batch_id(counting_provider, db_connection):
    """Test that sampling inference generates a batch_id (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_sampling_inference("Test prompt", n=3)
        
        # All results should have the same batch_id
        batch_ids = [r.get('batch_id') for r in results if 'batch_id' in r]
        assert len(batch_ids) == 3
        assert len(set(batch_ids)) == 1  # All the same
        assert batch_ids[0] is not None
        
        # Verify UUID format
        uuid.UUID(batch_ids[0])


@pytest.mark.asyncio
async def test_sampling_inference_with_custom_batch_id(counting_provider, db_connection):
    """Test that sampling inference can use a custom batch_id (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        custom_batch_id = str(uuid.uuid4())
        
        results = await service.run_sampling_inference(
            "Test prompt",
            n=3,
            batch_id=custom_batch_id
        )
        
        # All results should have the custom batch_id
        batch_ids = [r.get('batch_id') for r in results if 'batch_id' in r]
        assert all(bid == custom_batch_id for bid in batch_ids)


@pytest.mark.asyncio
async def test_batch_inference_generates_batch_id(counting_provider, db_connection):
    """Test that batch inference generates a batch_id (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_batch_inference(
            ["Prompt 1", "Prompt 2", "Prompt 3"]
        )
        
        # All results should have the same batch_id
        batch_ids = [r.get('batch_id') for r in results if 'batch_id' in r]
        assert len(batch_ids) == 3
        assert len(set(batch_ids)) == 1  # All the same
        assert batch_ids[0] is not None


@pytest.mark.asyncio
async def test_batch_inference_with_custom_batch_id(counting_provider, db_connection):
    """Test that batch inference can use a custom batch_id (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        custom_batch_id = str(uuid.uuid4())
        
        results = await service.run_batch_inference(
            ["Prompt 1", "Prompt 2"],
            batch_id=custom_batch_id
        )
        
        # All results should have the custom batch_id
        batch_ids = [r.get('batch_id') for r in results if 'batch_id' in r]
        assert all(bid == custom_batch_id for bid in batch_ids)


@pytest.mark.asyncio
async def test_single_inference_without_batch_id(counting_provider, db_connection):
    """Test that single inference doesn't require batch_id (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        result = await service.run_inference("Test prompt")
        
        # Should not have batch_id unless explicitly provided
        assert 'batch_id' not in result or result.get('batch_id') is None


@pytest.mark.asyncio
async def test_single_inference_with_batch_id(counting_provider, db_connection):
    """Test that single inference can accept batch_id (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        batch_id = str(uuid.uuid4())
        
        result = await service.run_inference("Test prompt", batch_id=batch_id)
        
        assert result.get('batch_id') == batch_id


@pytest.mark.asyncio
async def test_sampling_inference_stored_in_database(counting_provider, db_connection):
    """Test that sampling inference runs are stored with batch_id in database (v2)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_sampling_inference("Test prompt", n=3)
        batch_id = results[0].get('batch_id')
        
        session.commit()
    
    # Verify in database via groups
    from study_query_llm.db.models_v2 import Group
    
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        # Find group with this batch_id
        groups = session.query(Group).filter_by(group_type='batch').all()
        # Find the group with matching batch_id in metadata
        matching_group = None
        for group in groups:
            if group.metadata_json and group.metadata_json.get('batch_id') == batch_id:
                matching_group = group
                break
        
        assert matching_group is not None
        batch_calls = repo.get_calls_in_group(matching_group.id)
        assert len(batch_calls) == 3


@pytest.mark.asyncio
async def test_batch_inference_stored_in_database(counting_provider, db_connection):
    """Test that batch inference runs are stored with batch_id in database (v2)."""
    from study_query_llm.db.models_v2 import Group
    
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_batch_inference(
            ["Prompt 1", "Prompt 2", "Prompt 3"]
        )
        batch_id = results[0].get('batch_id')
        
        session.commit()
    
    # Verify in database via groups
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        # Find group with this batch_id
        groups = session.query(Group).filter_by(group_type='batch').all()
        # Find the group with matching batch_id in metadata
        matching_group = None
        for group in groups:
            if group.metadata_json and group.metadata_json.get('batch_id') == batch_id:
                matching_group = group
                break
        
        assert matching_group is not None
        batch_calls = repo.get_calls_in_group(matching_group.id)
        assert len(batch_calls) == 3


@pytest.mark.asyncio
async def test_different_batches_have_different_ids(counting_provider, db_connection):
    """Test that different batch operations generate different batch_ids (v2)."""
    from study_query_llm.db.models_v2 import Group
    
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        # Run two separate sampling inferences
        results1 = await service.run_sampling_inference("Prompt 1", n=2)
        results2 = await service.run_sampling_inference("Prompt 2", n=2)
        
        batch_id_1 = results1[0].get('batch_id')
        batch_id_2 = results2[0].get('batch_id')
        
        # Should be different
        assert batch_id_1 != batch_id_2
        
        session.commit()
    
    # Verify they're separate in database via groups
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        groups = session.query(Group).filter_by(group_type='batch').all()
        
        group_1 = None
        group_2 = None
        for group in groups:
            if group.metadata_json and group.metadata_json.get('batch_id') == batch_id_1:
                group_1 = group
            if group.metadata_json and group.metadata_json.get('batch_id') == batch_id_2:
                group_2 = group
        
        assert group_1 is not None
        assert group_2 is not None
        assert group_1.id != group_2.id
        
        batch_1_calls = repo.get_calls_in_group(group_1.id)
        batch_2_calls = repo.get_calls_in_group(group_2.id)
        
        assert len(batch_1_calls) == 2
        assert len(batch_2_calls) == 2

