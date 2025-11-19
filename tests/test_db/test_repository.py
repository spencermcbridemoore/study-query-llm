"""
Tests for InferenceRepository.
"""

import pytest
from datetime import datetime, timedelta, timezone
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.inference_repository import InferenceRepository
from study_query_llm.db.models import InferenceRun


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database."""
    db = DatabaseConnection("sqlite:///:memory:")
    db.init_db()
    return db


@pytest.fixture
def repository(db_connection):
    """Fixture for repository with session."""
    with db_connection.session_scope() as session:
        yield InferenceRepository(session)
        session.commit()


def test_insert_inference_run(db_connection):
    """Test inserting a single inference run."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        inference_id = repo.insert_inference_run(
            prompt="What is 2+2?",
            response="4",
            provider="azure_openai_gpt-4",
            tokens=10,
            latency_ms=250.5,
            metadata={"temperature": 0.7}
        )
        
        assert inference_id is not None
        assert inference_id > 0


def test_get_inference_by_id(db_connection):
    """Test retrieving inference by ID."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert
        inference_id = repo.insert_inference_run(
            prompt="Test prompt",
            response="Test response",
            provider="test"
        )
        
        # Retrieve
        inference = repo.get_inference_by_id(inference_id)
        
        assert inference is not None
        assert inference.id == inference_id
        assert inference.prompt == "Test prompt"
        assert inference.response == "Test response"


def test_query_inferences_by_provider(db_connection):
    """Test querying inferences by provider."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert test data
        repo.insert_inference_run("Prompt 1", "Response 1", "azure")
        repo.insert_inference_run("Prompt 2", "Response 2", "azure")
        repo.insert_inference_run("Prompt 3", "Response 3", "openai")
        
        # Query by provider
        azure_runs = repo.query_inferences(provider="azure")
        
        assert len(azure_runs) == 2
        assert all(run.provider == "azure" for run in azure_runs)


def test_query_inferences_with_date_range(db_connection):
    """Test querying inferences with date range."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert with specific dates (by manipulating created_at)
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)
        
        # These will have current timestamp, but we can test the filter
        repo.insert_inference_run("Old", "Response", "test")
        
        # Query with date range
        recent_runs = repo.query_inferences(
            date_range=(yesterday, tomorrow)
        )
        
        assert len(recent_runs) >= 1


def test_query_inferences_with_limit_and_offset(db_connection):
    """Test querying with pagination."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert multiple records
        for i in range(10):
            repo.insert_inference_run(
                f"Prompt {i}",
                f"Response {i}",
                "test"
            )
        
        # Query with limit
        first_page = repo.query_inferences(limit=5)
        assert len(first_page) == 5
        
        # Query with offset
        second_page = repo.query_inferences(limit=5, offset=5)
        assert len(second_page) == 5
        
        # Should be different records
        assert first_page[0].id != second_page[0].id


def test_get_provider_stats(db_connection):
    """Test getting provider statistics."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert test data
        repo.insert_inference_run("P1", "R1", "azure", tokens=100, latency_ms=500.0)
        repo.insert_inference_run("P2", "R2", "azure", tokens=200, latency_ms=1000.0)
        repo.insert_inference_run("P3", "R3", "openai", tokens=150, latency_ms=750.0)
        
        stats = repo.get_provider_stats()
        
        assert len(stats) == 2
        
        # Find azure stats
        azure_stats = next(s for s in stats if s['provider'] == 'azure')
        assert azure_stats['count'] == 2
        assert azure_stats['avg_tokens'] == 150.0
        assert azure_stats['total_tokens'] == 300


def test_search_by_prompt(db_connection):
    """Test searching inferences by prompt content."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert test data
        repo.insert_inference_run("What is Python?", "A programming language", "test")
        repo.insert_inference_run("What is Java?", "Another language", "test")
        repo.insert_inference_run("Tell me about Python", "Python is great", "test")
        
        # Search
        results = repo.search_by_prompt("Python")
        
        assert len(results) == 2
        assert all("Python" in r.prompt for r in results)


def test_get_total_count(db_connection):
    """Test getting total count of inferences."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Initially empty
        assert repo.get_total_count() == 0
        
        # Insert some
        repo.insert_inference_run("P1", "R1", "test")
        repo.insert_inference_run("P2", "R2", "test")
        
        assert repo.get_total_count() == 2


def test_batch_insert_inferences(db_connection):
    """Test batch inserting multiple inferences."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        inferences = [
            {
                "prompt": f"Prompt {i}",
                "response": f"Response {i}",
                "provider": "test",
                "tokens": 10 * i,
                "latency_ms": 100.0 * i
            }
            for i in range(5)
        ]
        
        ids = repo.batch_insert_inferences(inferences)
        
        assert len(ids) == 5
        assert all(id > 0 for id in ids)
        
        # Verify they were inserted
        assert repo.get_total_count() == 5


def test_get_count_by_provider(db_connection):
    """Test getting count by provider."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        repo.insert_inference_run("P1", "R1", "azure")
        repo.insert_inference_run("P2", "R2", "azure")
        repo.insert_inference_run("P3", "R3", "openai")
        
        assert repo.get_count_by_provider("azure") == 2
        assert repo.get_count_by_provider("openai") == 1
        assert repo.get_count_by_provider("nonexistent") == 0

