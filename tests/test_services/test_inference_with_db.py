"""
Tests for InferenceService with database integration (v2 schema).
"""

import pytest
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    # Use PostgreSQL connection string format, but SQLite will work for basic tests
    # For full v2 features, use PostgreSQL
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.mark.asyncio
async def test_inference_service_with_repository(mock_provider, db_connection):
    """Test that InferenceService saves to database when repository is provided (v2)."""
    service = InferenceService(mock_provider)
    
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service.repository = repo
        
        result = await service.run_inference("Test prompt")
        
        # Should have an ID
        assert 'id' in result
        assert result['id'] is not None
        
        # Verify it's in the database (v2 schema)
        saved = repo.get_raw_call_by_id(result['id'])
        assert saved is not None
        assert saved.status == 'success'
        assert saved.provider == result['metadata']['provider']
        # Check prompt in request_json
        assert saved.request_json is not None
        assert saved.request_json.get('prompt') == "Test prompt"
        # Check response in response_json
        assert saved.response_json is not None
        assert saved.response_json.get('text') == result['response']


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
    """Test that original prompt (before preprocessing) is stored (v2)."""
    service = InferenceService(
        mock_provider,
        preprocess=True,
        clean_whitespace=True
    )
    
    messy_prompt = "  hello   world  "
    
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service.repository = repo
        
        result = await service.run_inference(messy_prompt)
        
        # Verify original prompt is stored (not processed version) in request_json
        saved = repo.get_raw_call_by_id(result['id'])
        assert saved.request_json is not None
        assert saved.request_json.get('prompt') == messy_prompt  # Original, not cleaned


@pytest.mark.asyncio
async def test_inference_service_handles_db_errors_gracefully(mock_provider):
    """Test that database errors don't break inference."""
    # Create a mock repository that raises an error
    class FailingRepository:
        def insert_raw_call(self, *args, **kwargs):
            raise Exception("Database error")
        def create_group(self, *args, **kwargs):
            raise Exception("Database error")
        def add_call_to_group(self, *args, **kwargs):
            raise Exception("Database error")
    
    service = InferenceService(mock_provider, repository=FailingRepository())
    
    # Should still work even if DB fails
    result = await service.run_inference("Test prompt")
    
    assert 'response' in result
    assert result['response'] is not None
    # ID might not be set if DB failed, but that's OK

