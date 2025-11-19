"""
Tests for database connection management.
"""

import pytest
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.models import InferenceRun


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database connection."""
    return DatabaseConnection("sqlite:///:memory:")


def test_database_connection_initialization(db_connection):
    """Test creating a database connection."""
    assert db_connection is not None
    assert db_connection.engine is not None
    assert db_connection.SessionLocal is not None


def test_init_db(db_connection):
    """Test initializing database tables."""
    db_connection.init_db()
    
    # Verify tables were created by trying to query
    with db_connection.session_scope() as session:
        count = session.query(InferenceRun).count()
        assert count == 0  # Should be empty but table exists


def test_session_scope(db_connection):
    """Test session context manager."""
    db_connection.init_db()
    
    with db_connection.session_scope() as session:
        # Should be able to create a session
        assert session is not None
        
        # Should be able to add and query
        inference = InferenceRun(
            prompt="Test",
            response="Response",
            provider="test"
        )
        session.add(inference)
        # Commit happens automatically on exit


def test_session_scope_rollback_on_error(db_connection):
    """Test that session rolls back on error."""
    db_connection.init_db()
    
    try:
        with db_connection.session_scope() as session:
            inference = InferenceRun(
                prompt="Test",
                response="Response",
                provider="test"
            )
            session.add(inference)
            # Force an error
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Verify rollback - data should not be persisted
    with db_connection.session_scope() as session:
        count = session.query(InferenceRun).count()
        assert count == 0


def test_get_session(db_connection):
    """Test getting a session directly."""
    db_connection.init_db()
    
    session = db_connection.get_session()
    assert session is not None
    
    # Should manually close
    session.close()


def test_recreate_db(db_connection):
    """Test recreating database."""
    db_connection.init_db()
    
    # Add some data
    with db_connection.session_scope() as session:
        inference = InferenceRun(
            prompt="Test",
            response="Response",
            provider="test"
        )
        session.add(inference)
    
    # Verify data exists
    with db_connection.session_scope() as session:
        assert session.query(InferenceRun).count() == 1
    
    # Recreate
    db_connection.recreate_db()
    
    # Verify data is gone
    with db_connection.session_scope() as session:
        assert session.query(InferenceRun).count() == 0

