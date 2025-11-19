"""
Tests for batch tracking in InferenceService.

Tests that run_repeated_inference and run_batch_inference properly
generate and use batch_id for grouping runs.
"""

import pytest
import uuid
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.inference_repository import InferenceRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database."""
    db = DatabaseConnection("sqlite:///:memory:")
    db.init_db()
    return db


@pytest.mark.asyncio
async def test_repeated_inference_generates_batch_id(counting_provider, db_connection):
    """Test that repeated inference generates a batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_repeated_inference("Test prompt", n=3)
        
        # All results should have the same batch_id
        batch_ids = [r.get('batch_id') for r in results if 'batch_id' in r]
        assert len(batch_ids) == 3
        assert len(set(batch_ids)) == 1  # All the same
        assert batch_ids[0] is not None
        
        # Verify UUID format
        uuid.UUID(batch_ids[0])


@pytest.mark.asyncio
async def test_repeated_inference_with_custom_batch_id(counting_provider, db_connection):
    """Test that repeated inference can use a custom batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        custom_batch_id = str(uuid.uuid4())
        
        results = await service.run_repeated_inference(
            "Test prompt",
            n=3,
            batch_id=custom_batch_id
        )
        
        # All results should have the custom batch_id
        batch_ids = [r.get('batch_id') for r in results if 'batch_id' in r]
        assert all(bid == custom_batch_id for bid in batch_ids)


@pytest.mark.asyncio
async def test_batch_inference_generates_batch_id(counting_provider, db_connection):
    """Test that batch inference generates a batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
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
    """Test that batch inference can use a custom batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
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
    """Test that single inference doesn't require batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        result = await service.run_inference("Test prompt")
        
        # Should not have batch_id unless explicitly provided
        assert 'batch_id' not in result or result.get('batch_id') is None


@pytest.mark.asyncio
async def test_single_inference_with_batch_id(counting_provider, db_connection):
    """Test that single inference can accept batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        batch_id = str(uuid.uuid4())
        
        result = await service.run_inference("Test prompt", batch_id=batch_id)
        
        assert result.get('batch_id') == batch_id


@pytest.mark.asyncio
async def test_repeated_inference_stored_in_database(counting_provider, db_connection):
    """Test that repeated inference runs are stored with batch_id in database."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_repeated_inference("Test prompt", n=3)
        batch_id = results[0].get('batch_id')
        
        session.commit()
    
    # Verify in database
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_runs = repo.get_inferences_by_batch_id(batch_id)
        
        assert len(batch_runs) == 3
        assert all(run.batch_id == batch_id for run in batch_runs)


@pytest.mark.asyncio
async def test_batch_inference_stored_in_database(counting_provider, db_connection):
    """Test that batch inference runs are stored with batch_id in database."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        results = await service.run_batch_inference(
            ["Prompt 1", "Prompt 2", "Prompt 3"]
        )
        batch_id = results[0].get('batch_id')
        
        session.commit()
    
    # Verify in database
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_runs = repo.get_inferences_by_batch_id(batch_id)
        
        assert len(batch_runs) == 3
        assert all(run.batch_id == batch_id for run in batch_runs)


@pytest.mark.asyncio
async def test_different_batches_have_different_ids(counting_provider, db_connection):
    """Test that different batch operations generate different batch_ids."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service = InferenceService(counting_provider, repository=repo)
        
        # Run two separate repeated inferences
        results1 = await service.run_repeated_inference("Prompt 1", n=2)
        results2 = await service.run_repeated_inference("Prompt 2", n=2)
        
        batch_id_1 = results1[0].get('batch_id')
        batch_id_2 = results2[0].get('batch_id')
        
        # Should be different
        assert batch_id_1 != batch_id_2
        
        session.commit()
    
    # Verify they're separate in database
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_1_runs = repo.get_inferences_by_batch_id(batch_id_1)
        batch_2_runs = repo.get_inferences_by_batch_id(batch_id_2)
        
        assert len(batch_1_runs) == 2
        assert len(batch_2_runs) == 2

