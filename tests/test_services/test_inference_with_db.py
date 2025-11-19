"""
Tests for InferenceService with database integration.
"""

import pytest
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
async def test_inference_service_with_repository(mock_provider, db_connection):
    """Test that InferenceService saves to database when repository is provided."""
    service = InferenceService(mock_provider)
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service.repository = repo
        
        result = await service.run_inference("Test prompt")
        
        # Should have an ID
        assert 'id' in result
        assert result['id'] is not None
        
        # Verify it's in the database
        saved = repo.get_inference_by_id(result['id'])
        assert saved is not None
        assert saved.prompt == "Test prompt"
        assert saved.response == result['response']
        assert saved.provider == result['metadata']['provider']


@pytest.mark.asyncio
async def test_inference_service_without_repository(mock_provider):
    """Test that InferenceService works without repository."""
    service = InferenceService(mock_provider)
    
    result = await service.run_inference("Test prompt")
    
    # Should not have an ID
    assert 'id' not in result or result.get('id') is None
    assert 'response' in result
    assert result['response'] is not None


@pytest.mark.asyncio
async def test_inference_service_stores_original_prompt(mock_provider, db_connection):
    """Test that original prompt (before preprocessing) is stored."""
    service = InferenceService(
        mock_provider,
        preprocess=True,
        clean_whitespace=True
    )
    
    messy_prompt = "  hello   world  "
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        service.repository = repo
        
        result = await service.run_inference(messy_prompt)
        
        # Verify original prompt is stored (not processed version)
        saved = repo.get_inference_by_id(result['id'])
        assert saved.prompt == messy_prompt  # Original, not cleaned


@pytest.mark.asyncio
async def test_inference_service_handles_db_errors_gracefully(mock_provider):
    """Test that database errors don't break inference."""
    # Create a mock repository that raises an error
    class FailingRepository:
        def insert_inference_run(self, *args, **kwargs):
            raise Exception("Database error")
    
    service = InferenceService(mock_provider, repository=FailingRepository())
    
    # Should still work even if DB fails
    result = await service.run_inference("Test prompt")
    
    assert 'response' in result
    assert result['response'] is not None
    # ID might not be set if DB failed, but that's OK

