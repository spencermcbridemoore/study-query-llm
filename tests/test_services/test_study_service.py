"""
Tests for Phase 4 - Study Service (Analytics).

Tests the StudyService for analyzing stored inference data.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from study_query_llm.services.study_service import StudyService
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.inference_repository import InferenceRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database."""
    db = DatabaseConnection("sqlite:///:memory:")
    db.init_db()
    return db


@pytest.fixture
def study_service(db_connection):
    """Fixture for StudyService with test data."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        
        # Insert test data
        repo.insert_inference_run("What is Python?", "A programming language", "azure", tokens=50, latency_ms=500.0)
        repo.insert_inference_run("What is Java?", "Another language", "azure", tokens=45, latency_ms=450.0)
        repo.insert_inference_run("Explain Python", "Python is great", "openai", tokens=60, latency_ms=600.0)
        repo.insert_inference_run("Tell me about Python", "Python is versatile", "openai", tokens=55, latency_ms=550.0)
        
        session.commit()
        
        # Return service with new session for queries
        new_session = db_connection.get_session()
        new_repo = InferenceRepository(new_session)
        yield StudyService(new_repo)
        new_session.close()


def test_get_provider_comparison(study_service):
    """Test provider comparison."""
    df = study_service.get_provider_comparison()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'provider' in df.columns
    assert 'count' in df.columns
    assert 'avg_tokens' in df.columns
    assert 'avg_latency_ms' in df.columns
    assert 'total_tokens' in df.columns
    assert 'avg_cost_estimate' in df.columns
    
    # Should have 2 providers
    assert len(df) == 2


def test_get_provider_comparison_empty(db_connection):
    """Test provider comparison with empty database."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        study = StudyService(repo)
        
        df = study.get_provider_comparison()
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty


def test_get_recent_inferences(study_service):
    """Test getting recent inferences."""
    df = study_service.get_recent_inferences(limit=10)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 10
    assert 'id' in df.columns
    assert 'prompt' in df.columns
    assert 'response' in df.columns
    assert 'provider' in df.columns
    assert 'tokens' in df.columns
    assert 'latency_ms' in df.columns
    assert 'created_at' in df.columns


def test_get_recent_inferences_with_provider_filter(study_service):
    """Test getting recent inferences filtered by provider."""
    df = study_service.get_recent_inferences(limit=10, provider="azure")
    
    assert isinstance(df, pd.DataFrame)
    assert all(df['provider'] == 'azure')


def test_get_recent_inferences_truncation(study_service, db_connection):
    """Test that long prompts/responses are truncated."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        repo.insert_inference_run(
            "A" * 200,  # Long prompt
            "B" * 200,  # Long response
            "test"
        )
        session.commit()
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        study = StudyService(repo)
        df = study.get_recent_inferences(limit=1)
        
        # Should be truncated
        assert len(df.iloc[0]['prompt']) <= 103  # 100 chars + "..."
        assert len(df.iloc[0]['response']) <= 103


def test_get_time_series_data(study_service):
    """Test time-series data aggregation."""
    df = study_service.get_time_series_data(days=7, group_by='day')
    
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert 'created_at' in df.columns
        assert 'provider' in df.columns
        assert 'tokens_sum' in df.columns
        assert 'tokens_mean' in df.columns
        assert 'latency_ms_mean' in df.columns


def test_get_time_series_data_different_groupings(study_service):
    """Test time-series with different group_by values."""
    for group_by in ['day', 'hour', 'minute']:
        df = study_service.get_time_series_data(days=7, group_by=group_by)
        assert isinstance(df, pd.DataFrame)


def test_get_time_series_data_empty(db_connection):
    """Test time-series with empty database."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        study = StudyService(repo)
        
        df = study.get_time_series_data(days=7)
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty


def test_search_prompts(study_service):
    """Test searching prompts."""
    df = study_service.search_prompts("Python")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Search is case-insensitive, so check that all results contain the search term (case-insensitive)
    assert all("python" in prompt.lower() for prompt in df['prompt'])


def test_search_prompts_case_insensitive(study_service):
    """Test that search is case-insensitive."""
    df_lower = study_service.search_prompts("python")
    df_upper = study_service.search_prompts("PYTHON")
    
    # Should return same results
    assert len(df_lower) == len(df_upper)


def test_search_prompts_no_results(study_service):
    """Test search with no matching results."""
    df = study_service.search_prompts("nonexistent_term_xyz")
    
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_search_prompts_response_truncation(study_service, db_connection):
    """Test that long responses are truncated in search results."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        repo.insert_inference_run(
            "Test prompt",
            "X" * 300,  # Long response
            "test"
        )
        session.commit()
    
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        study = StudyService(repo)
        df = study.search_prompts("Test")
        
        if not df.empty:
            assert len(df.iloc[0]['response']) <= 203  # 200 chars + "..."


def test_get_summary_stats(study_service):
    """Test getting summary statistics."""
    stats = study_service.get_summary_stats()
    
    assert isinstance(stats, dict)
    assert 'total_inferences' in stats
    assert 'total_tokens' in stats
    assert 'unique_providers' in stats
    assert 'provider_breakdown' in stats
    
    assert stats['total_inferences'] > 0
    assert stats['total_tokens'] > 0
    assert stats['unique_providers'] > 0
    assert isinstance(stats['provider_breakdown'], list)


def test_get_summary_stats_empty(db_connection):
    """Test summary stats with empty database."""
    with db_connection.session_scope() as session:
        repo = InferenceRepository(session)
        study = StudyService(repo)
        
        stats = study.get_summary_stats()
        
        assert stats['total_inferences'] == 0
        assert stats['total_tokens'] == 0
        assert stats['unique_providers'] == 0
        assert stats['provider_breakdown'] == []


def test_dataframe_types(study_service):
    """Test that DataFrames have correct data types."""
    # Test provider comparison
    comparison = study_service.get_provider_comparison()
    assert isinstance(comparison['count'].dtype, (pd.Int64Dtype, type(pd.Series([1]).dtype)))
    
    # Test recent inferences
    recent = study_service.get_recent_inferences()
    if not recent.empty:
        assert pd.api.types.is_datetime64_any_dtype(recent['created_at'])

