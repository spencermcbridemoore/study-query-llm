"""
Tests for batch tracking functionality.

Tests that batch_id is properly stored and can be used to group inference runs.
"""

import pytest
import uuid
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.inference_repository import InferenceRepository
from study_query_llm.db.models import InferenceRun


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database."""
    db = DatabaseConnection("sqlite:///:memory:")
    db.init_db()
    return db


def test_inference_run_with_batch_id(db_connection):
    """Test creating InferenceRun with batch_id."""
    with db_connection.session_scope() as session:
        batch_id = str(uuid.uuid4())
        inference = InferenceRun(
            prompt="Test prompt",
            response="Test response",
            provider="azure",
            batch_id=batch_id
        )
        
        assert inference.batch_id == batch_id
        assert inference.prompt == "Test prompt"


def test_inference_run_without_batch_id(db_connection):
    """Test creating InferenceRun without batch_id (nullable)."""
    with db_connection.session_scope() as session:
        inference = InferenceRun(
            prompt="Test prompt",
            response="Test response",
            provider="azure"
        )
        
        assert inference.batch_id is None


def test_to_dict_includes_batch_id(db_connection):
    """Test that to_dict() includes batch_id."""
    batch_id = str(uuid.uuid4())
    inference = InferenceRun(
        id=1,
        prompt="Test",
        response="Response",
        provider="azure",
        batch_id=batch_id
    )
    
    result = inference.to_dict()
    assert result['batch_id'] == batch_id


def test_insert_inference_with_batch_id(db_connection):
    """Test inserting inference with batch_id."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id = str(uuid.uuid4())
        
        inference_id = repo.insert_inference_run(
            prompt="Test",
            response="Response",
            provider="azure",
            batch_id=batch_id
        )
        
        assert inference_id is not None
        
        # Verify it was stored
        inference = repo.get_inference_by_id(inference_id)
        assert inference.batch_id == batch_id


def test_get_inferences_by_batch_id(db_connection):
    """Test retrieving all inferences in a batch."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id = str(uuid.uuid4())
        
        # Insert 3 inferences with same batch_id
        for i in range(3):
            repo.insert_inference_run(
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                provider="azure",
                batch_id=batch_id,
                tokens=100 + i,
                latency_ms=500.0 + i
            )
        
        # Insert 1 inference without batch_id
        repo.insert_inference_run(
            prompt="Other prompt",
            response="Other response",
            provider="azure"
        )
        
        session.commit()
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_runs = repo.get_inferences_by_batch_id(batch_id)
        
        assert len(batch_runs) == 3
        assert all(run.batch_id == batch_id for run in batch_runs)
        assert all(f"Prompt {i}" in [run.prompt for run in batch_runs] for i in range(3))


def test_get_inferences_by_batch_id_empty(db_connection):
    """Test retrieving batch with no runs."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id = str(uuid.uuid4())
        
        runs = repo.get_inferences_by_batch_id(batch_id)
        assert len(runs) == 0


def test_get_batch_summary(db_connection):
    """Test getting batch summary statistics."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id = str(uuid.uuid4())
        
        # Insert multiple runs with different metrics
        repo.insert_inference_run(
            prompt="Test 1",
            response="Response 1",
            provider="azure",
            batch_id=batch_id,
            tokens=100,
            latency_ms=500.0
        )
        repo.insert_inference_run(
            prompt="Test 2",
            response="Response 2",
            provider="openai",
            batch_id=batch_id,
            tokens=200,
            latency_ms=1000.0
        )
        repo.insert_inference_run(
            prompt="Test 3",
            response="Response 3",
            provider="azure",
            batch_id=batch_id,
            tokens=150,
            latency_ms=750.0
        )
        
        session.commit()
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        summary = repo.get_batch_summary(batch_id)
        
        assert summary['batch_id'] == batch_id
        assert summary['total_runs'] == 3
        assert set(summary['providers']) == {'azure', 'openai'}
        assert summary['total_tokens'] == 450
        assert summary['avg_tokens'] == 150.0
        assert summary['avg_latency_ms'] == 750.0
        assert summary['min_latency_ms'] == 500.0
        assert summary['max_latency_ms'] == 1000.0
        assert summary['created_at_range'][0] is not None
        assert summary['created_at_range'][1] is not None


def test_get_batch_summary_empty(db_connection):
    """Test getting summary for non-existent batch."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id = str(uuid.uuid4())
        
        summary = repo.get_batch_summary(batch_id)
        
        assert summary['batch_id'] == batch_id
        assert summary['total_runs'] == 0
        assert summary['providers'] == []
        assert summary['total_tokens'] == 0
        assert summary['avg_tokens'] == 0.0
        assert summary['avg_latency_ms'] == 0.0


def test_get_batch_summary_with_null_values(db_connection):
    """Test batch summary when some runs have null tokens/latency."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id = str(uuid.uuid4())
        
        # Insert runs with and without metrics
        repo.insert_inference_run(
            prompt="Test 1",
            response="Response 1",
            provider="azure",
            batch_id=batch_id,
            tokens=100,
            latency_ms=500.0
        )
        repo.insert_inference_run(
            prompt="Test 2",
            response="Response 2",
            provider="azure",
            batch_id=batch_id
            # No tokens or latency
        )
        
        session.commit()
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        summary = repo.get_batch_summary(batch_id)
        
        assert summary['total_runs'] == 2
        assert summary['total_tokens'] == 100
        assert summary['avg_tokens'] == 100.0
        assert summary['avg_latency_ms'] == 500.0


def test_multiple_batches_dont_interfere(db_connection):
    """Test that different batches don't interfere with each other."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        batch_id_1 = str(uuid.uuid4())
        batch_id_2 = str(uuid.uuid4())
        
        # Insert runs in different batches
        repo.insert_inference_run("Prompt 1", "Response 1", "azure", batch_id=batch_id_1)
        repo.insert_inference_run("Prompt 2", "Response 2", "azure", batch_id=batch_id_1)
        repo.insert_inference_run("Prompt 3", "Response 3", "azure", batch_id=batch_id_2)
        
        session.commit()
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        batch_1_runs = repo.get_inferences_by_batch_id(batch_id_1)
        batch_2_runs = repo.get_inferences_by_batch_id(batch_id_2)
        
        assert len(batch_1_runs) == 2
        assert len(batch_2_runs) == 1
        assert all(r.batch_id == batch_id_1 for r in batch_1_runs)
        assert all(r.batch_id == batch_id_2 for r in batch_2_runs)

