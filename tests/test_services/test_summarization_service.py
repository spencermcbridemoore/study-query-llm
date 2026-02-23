"""
Tests for SummarizationService.

Tests LLM-based summarization with RawCall logging and group integration.
"""

import pytest
from unittest.mock import AsyncMock, patch
from study_query_llm.services.summarization_service import (
    SummarizationService,
    SummarizationRequest,
    SummarizationResponse,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.fixture
def mock_inference_service():
    """Fixture for a mock InferenceService."""
    service = AsyncMock()
    service.run_inference = AsyncMock(
        return_value={"response": "Mock summary", "metadata": {}}
    )
    return service


@pytest.mark.asyncio
async def test_summarize_batch_basic(mock_inference_service, db_connection):
    """Test basic batch summarization."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        request = SummarizationRequest(
            texts=["Text 1", "Text 2"],
            llm_deployment="gpt-4",
            validate_deployment=False,  # Skip validation in test
        )

        with patch.object(
            service, "_validate_deployment", return_value=True
        ), patch(
            "study_query_llm.services.summarization_service.ProviderFactory"
        ) as mock_factory, patch(
            "study_query_llm.services.summarization_service.InferenceService",
            return_value=mock_inference_service,
        ):
            result = await service.summarize_batch(request)

            assert len(result.summaries) == 2
            assert all(s == "Mock summary" for s in result.summaries)
            assert len(result.raw_call_ids) == 2
            assert all(call_id > 0 for call_id in result.raw_call_ids)


@pytest.mark.asyncio
async def test_summarize_batch_logs_to_rawcall(mock_inference_service, db_connection):
    """Test that summarization calls are logged to RawCall."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        request = SummarizationRequest(
            texts=["Test text"],
            llm_deployment="gpt-4",
            validate_deployment=False,
        )

        with patch.object(
            service, "_validate_deployment", return_value=True
        ), patch(
            "study_query_llm.services.summarization_service.ProviderFactory"
        ), patch(
            "study_query_llm.services.summarization_service.InferenceService",
            return_value=mock_inference_service,
        ):
            result = await service.summarize_batch(request)

            # Verify RawCall was created
            assert len(result.raw_call_ids) == 1
            call_id = result.raw_call_ids[0]

            saved_call = repo.get_raw_call_by_id(call_id)
            assert saved_call is not None
            assert saved_call.status == "success"
            assert saved_call.modality == "text"
            assert saved_call.model == "gpt-4"
            assert saved_call.response_json is not None
            assert saved_call.response_json["text"] == "Mock summary"


@pytest.mark.asyncio
async def test_summarize_batch_with_group_id(mock_inference_service, db_connection):
    """Test that group_id is stored in metadata."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        # Create a group
        group_id = repo.create_group(
            group_type="clustering_run",
            name="test_run",
            description="Test run",
        )

        request = SummarizationRequest(
            texts=["Test text"],
            llm_deployment="gpt-4",
            group_id=group_id,
            validate_deployment=False,
        )

        with patch.object(
            service, "_validate_deployment", return_value=True
        ), patch(
            "study_query_llm.services.summarization_service.ProviderFactory"
        ), patch(
            "study_query_llm.services.summarization_service.InferenceService",
            return_value=mock_inference_service,
        ):
            result = await service.summarize_batch(request)

            # Verify group_id is in metadata
            call_id = result.raw_call_ids[0]
            saved_call = repo.get_raw_call_by_id(call_id)
            assert saved_call.metadata_json["group_id"] == group_id


@pytest.mark.asyncio
async def test_summarize_batch_logs_failures(mock_inference_service, db_connection):
    """Test that failed calls are logged with status='failed'."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        # Make mock service raise an error
        mock_inference_service.run_inference = AsyncMock(
            side_effect=Exception("API error")
        )

        request = SummarizationRequest(
            texts=["Test text"],
            llm_deployment="gpt-4",
            validate_deployment=False,
        )

        with patch.object(
            service, "_validate_deployment", return_value=True
        ), patch(
            "study_query_llm.services.summarization_service.ProviderFactory"
        ), patch(
            "study_query_llm.services.summarization_service.InferenceService",
            return_value=mock_inference_service,
        ):
            with pytest.raises(RuntimeError, match="All 1 summarizations failed"):
                await service.summarize_batch(request)

            # Verify failure was logged
            failed_calls = repo.query_raw_calls(status="failed", limit=10)
            assert len(failed_calls) > 0
            failed_call = failed_calls[0]
            assert failed_call.status == "failed"
            assert failed_call.error_json is not None
            assert "error_type" in failed_call.error_json
            assert "error_message" in failed_call.error_json


@pytest.mark.asyncio
async def test_summarize_batch_invalid_deployment(db_connection):
    """Test that invalid deployment is caught and logged."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        request = SummarizationRequest(
            texts=["Test text"],
            llm_deployment="invalid-deployment",
            validate_deployment=True,
        )

        with patch.object(service, "_validate_deployment", return_value=False):
            with pytest.raises(ValueError, match="Invalid deployment"):
                await service.summarize_batch(request)

            # Verify all texts were logged as failed
            failed_calls = repo.query_raw_calls(status="failed", limit=10)
            assert len(failed_calls) >= 1  # At least one failure logged


@pytest.mark.asyncio
async def test_summarize_batch_partial_failures(mock_inference_service, db_connection):
    """Test that partial failures are handled correctly."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        # Make first call succeed, second fail
        call_count = 0

        async def mock_run_inference(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"response": "Success", "metadata": {}}
            else:
                raise Exception("API error")

        mock_inference_service.run_inference = AsyncMock(side_effect=mock_run_inference)

        request = SummarizationRequest(
            texts=["Text 1", "Text 2"],
            llm_deployment="gpt-4",
            validate_deployment=False,
        )

        with patch.object(
            service, "_validate_deployment", return_value=True
        ), patch(
            "study_query_llm.services.summarization_service.ProviderFactory"
        ), patch(
            "study_query_llm.services.summarization_service.InferenceService",
            return_value=mock_inference_service,
        ):
            result = await service.summarize_batch(request)

            # Should have one success, one failure
            assert len(result.summaries) == 2
            assert result.summaries[0] == "Success"
            assert result.summaries[1] == ""  # Placeholder for failure
            assert result.metadata["successful"] == 1
            assert result.metadata["failed"] == 1


def test_create_paraphraser_for_llm(mock_inference_service, db_connection):
    """Test creating a paraphraser callable."""
    # This test runs in sync context (not async) to properly test the paraphraser
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        with patch.object(
            service, "_validate_deployment", return_value=True
        ), patch(
            "study_query_llm.services.summarization_service.ProviderFactory"
        ), patch(
            "study_query_llm.services.summarization_service.InferenceService",
            return_value=mock_inference_service,
        ):
            paraphraser = service.create_paraphraser_for_llm("gpt-4")

            assert paraphraser is not None
            assert callable(paraphraser)

            # Test the paraphraser (will use asyncio.run internally)
            # Note: This will work in a non-async context, but may fail in async test context
            # The paraphraser is designed for use in sync algorithms
            try:
                summaries = paraphraser(["Text 1", "Text 2"])
                assert len(summaries) == 2
                assert all(s == "Mock summary" for s in summaries)
            except RuntimeError as e:
                # If we're in an async context (pytest-asyncio), this is expected
                # The paraphraser is meant for sync contexts
                if "Cannot run paraphraser in async context" in str(e):
                    pytest.skip("Paraphraser requires sync context (not async test)")
                else:
                    raise


@pytest.mark.asyncio
async def test_summarize_batch_without_repository(mock_inference_service):
    """Test that service works without repository (no persistence)."""
    service = SummarizationService(repository=None)

    request = SummarizationRequest(
        texts=["Test text"],
        llm_deployment="gpt-4",
        validate_deployment=False,
    )

    with patch.object(
        service, "_validate_deployment", return_value=True
    ), patch(
        "study_query_llm.services.summarization_service.ProviderFactory"
    ), patch(
        "study_query_llm.services.summarization_service.InferenceService",
        return_value=mock_inference_service,
    ):
        result = await service.summarize_batch(request)

        assert len(result.summaries) == 1
        assert result.raw_call_ids == [0]  # No persistence, so 0 placeholder
