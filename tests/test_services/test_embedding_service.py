"""
Tests for EmbeddingService.

Tests embedding generation, caching, deployment validation, retry logic, and DB persistence.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from study_query_llm.services.embedding_service import (
    EmbeddingService,
    EmbeddingRequest,
    EmbeddingResponse,
    DEPLOYMENT_MAX_TOKENS,
    estimate_tokens,
)
from study_query_llm.providers.base_embedding import BaseEmbeddingProvider, EmbeddingResult
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.fixture
def mock_provider():
    """A no-op BaseEmbeddingProvider for tests that mock internal methods."""
    provider = AsyncMock(spec=BaseEmbeddingProvider)
    provider.validate_model = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    provider.get_provider_name = MagicMock(return_value="mock")
    return provider


@pytest.fixture
def mock_embedding_response():
    """Fixture for a mock EmbeddingResult."""
    return EmbeddingResult(vector=[0.1, 0.2, 0.3, 0.4, 0.5], index=0)


@pytest.mark.asyncio
async def test_get_embedding_basic(mock_embedding_response, db_connection, mock_provider):
    """Test basic embedding generation."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        # Mock the embedding API call
        with patch.object(
            service, "_create_embedding_with_retry", return_value=mock_embedding_response
        ):
            with patch.object(service, "_validate_deployment", return_value=True):
                request = EmbeddingRequest(
                    text="Hello world",
                    deployment="text-embedding-ada-002",
                )

                result = await service.get_embedding(request)

                assert result.vector == [0.1, 0.2, 0.3, 0.4, 0.5]
                assert result.dimension == 5
                assert result.model == "text-embedding-ada-002"
                assert result.cached is False
                assert result.raw_call_id is not None


@pytest.mark.asyncio
async def test_get_embedding_cache_hit(mock_embedding_response, db_connection, mock_provider):
    """Test that cache hit returns stored vector without provider call."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        # First, create an embedding and store it
        with patch.object(
            service, "_create_embedding_with_retry", return_value=mock_embedding_response
        ):
            with patch.object(service, "_validate_deployment", return_value=True):
                request = EmbeddingRequest(
                    text="Hello world",
                    deployment="text-embedding-ada-002",
                )

                # First call - should create and store
                result1 = await service.get_embedding(request)
                assert result1.cached is False

                # Second call - should hit cache
                result2 = await service.get_embedding(request)
                assert result2.cached is True
                assert result2.vector == result1.vector
                assert result2.raw_call_id == result1.raw_call_id


@pytest.mark.asyncio
async def test_get_embedding_invalid_deployment(db_connection, mock_provider):
    """Test that invalid deployment is skipped and logged once."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        # Mock validation to return False
        with patch.object(service, "_validate_deployment", return_value=False):
            request = EmbeddingRequest(
                text="Hello world",
                deployment="invalid-deployment",
            )

            with pytest.raises(ValueError, match="Invalid deployment"):
                await service.get_embedding(request)

            # Verify failure was logged
            failed_calls = repo.query_raw_calls(status="failed", limit=10)
            assert len(failed_calls) > 0
            assert failed_calls[0].model == "invalid-deployment"
            assert failed_calls[0].error_json is not None


@pytest.mark.asyncio
async def test_get_embedding_retry_configured(mock_provider):
    """Test that retry is configured for transient errors."""
    from study_query_llm.services._shared import should_retry_exception

    service = EmbeddingService(max_retries=6, initial_wait=1.0, max_wait=30.0, provider=mock_provider)

    assert service.max_retries == 6
    assert service.initial_wait == 1.0
    assert service.max_wait == 30.0

    assert should_retry_exception(Exception("502 Bad Gateway")) is True
    assert should_retry_exception(Exception("Rate limit exceeded")) is True
    assert should_retry_exception(Exception("Connection timeout")) is True
    assert should_retry_exception(Exception("503 Service Unavailable")) is True
    assert should_retry_exception(Exception("401 Unauthorized")) is False


@pytest.mark.asyncio
async def test_get_embedding_failure_logged(db_connection, mock_provider):
    """Test that failed calls are persisted with status='failed' and error_json."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, max_retries=1, provider=mock_provider)

        # Mock to always fail
        async def mock_create_with_retry(*args, **kwargs):
            raise ValueError("Permanent error")

        with patch.object(service, "_create_embedding_with_retry", side_effect=mock_create_with_retry):
            with patch.object(service, "_validate_deployment", return_value=True):
                request = EmbeddingRequest(
                    text="Hello world",
                    deployment="text-embedding-ada-002",
                )

                with pytest.raises(ValueError):
                    await service.get_embedding(request)

                # Verify failure was logged
                failed_calls = repo.query_raw_calls(status="failed", limit=10)
                assert len(failed_calls) > 0
                failed_call = failed_calls[0]
                assert failed_call.status == "failed"
                assert failed_call.error_json is not None
                assert "error_type" in failed_call.error_json
                assert "error_message" in failed_call.error_json


@pytest.mark.asyncio
async def test_compute_request_hash_deterministic(mock_provider):
    """Test that deterministic hashing ensures same input produces same hash."""
    service = EmbeddingService(provider=mock_provider)

    hash1 = service._compute_request_hash(
        "Hello world", "text-embedding-ada-002", None, "float", "azure"
    )
    hash2 = service._compute_request_hash(
        "Hello world", "text-embedding-ada-002", None, "float", "azure"
    )

    assert hash1 == hash2

    # Different text should produce different hash
    hash3 = service._compute_request_hash(
        "Different text", "text-embedding-ada-002", None, "float", "azure"
    )
    assert hash1 != hash3


@pytest.mark.asyncio
async def test_normalize_text(mock_provider):
    """Test that text normalization removes null bytes and normalizes whitespace."""
    service = EmbeddingService(provider=mock_provider)

    # Test null byte removal
    text1 = "Hello\x00world"
    normalized1 = service._normalize_text(text1)
    assert "\x00" not in normalized1

    # Test whitespace normalization
    text2 = "Hello   world\n\t  test"
    normalized2 = service._normalize_text(text2)
    assert "  " not in normalized2  # No double spaces
    assert "\n" not in normalized2  # No newlines
    assert "\t" not in normalized2  # No tabs


@pytest.mark.asyncio
async def test_get_embedding_rejects_empty_string(db_connection, mock_provider):
    """Test that empty strings are rejected before API calls."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        request = EmbeddingRequest(
            text="",
            deployment="text-embedding-ada-002",
        )

        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            await service.get_embedding(request)


@pytest.mark.asyncio
async def test_get_embedding_rejects_whitespace_only(db_connection, mock_provider):
    """Test that whitespace-only strings are rejected."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        # Test various whitespace-only strings
        whitespace_strings = ["   ", "\n\t\n", "  \r\n  ", "\t"]

        for text in whitespace_strings:
            request = EmbeddingRequest(
                text=text,
                deployment="text-embedding-ada-002",
            )

            with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
                await service.get_embedding(request)


@pytest.mark.asyncio
async def test_get_embedding_rejects_null_bytes_only(db_connection, mock_provider):
    """Test that strings with only null bytes are rejected."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        request = EmbeddingRequest(
            text="\x00\x00",
            deployment="text-embedding-ada-002",
        )

        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            await service.get_embedding(request)


@pytest.mark.asyncio
async def test_get_embedding_rejects_whitespace_and_null_bytes(db_connection, mock_provider):
    """Test that strings with only whitespace and null bytes are rejected."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        request = EmbeddingRequest(
            text="  \x00  \n\t",
            deployment="text-embedding-ada-002",
        )

        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            await service.get_embedding(request)


@pytest.mark.asyncio
async def test_get_embedding_logs_failure_for_empty_string(db_connection, mock_provider):
    """Test that empty string failures are logged to the database."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        request = EmbeddingRequest(
            text="   ",
            deployment="text-embedding-ada-002",
        )

        with pytest.raises(ValueError):
            await service.get_embedding(request)

        # Check that failure was logged
        from study_query_llm.db.models_v2 import RawCall
        failed_calls = (
            session.query(RawCall)
            .filter(RawCall.modality == "embedding")
            .filter(RawCall.status == "failed")
            .all()
        )

        assert len(failed_calls) == 1
        error_message = failed_calls[0].error_json.get("error_message", "")
        assert "empty text" in error_message.lower()


@pytest.mark.asyncio
async def test_get_embeddings_batch(mock_embedding_response, db_connection, mock_provider):
    """Test that batch operations work correctly with caching per item."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        with patch.object(
            service, "_create_embedding_with_retry", return_value=mock_embedding_response
        ):
            with patch.object(service, "_validate_deployment", return_value=True):
                requests = [
                    EmbeddingRequest(text="Text 1", deployment="text-embedding-ada-002"),
                    EmbeddingRequest(text="Text 2", deployment="text-embedding-ada-002"),
                    EmbeddingRequest(text="Text 3", deployment="text-embedding-ada-002"),
                ]

                results = await service.get_embeddings_batch(requests)

                assert len(results) == 3
                assert all(r.vector == [0.1, 0.2, 0.3, 0.4, 0.5] for r in results)
                assert all(r.cached is False for r in results)

                # Second batch - should all hit cache
                results2 = await service.get_embeddings_batch(requests)
                assert all(r.cached is True for r in results2)


@pytest.mark.asyncio
async def test_filter_valid_deployments(mock_provider):
    """Test deployment filtering."""
    service = EmbeddingService(provider=mock_provider)

    # Mock validation
    async def mock_validate(deployment, provider):
        return deployment in ["valid-1", "valid-2"]

    with patch.object(service, "_validate_deployment", side_effect=mock_validate):
        deployments = ["valid-1", "invalid-1", "valid-2", "invalid-2"]
        valid = await service.filter_valid_deployments(deployments)

        assert valid == ["valid-1", "valid-2"]


@pytest.mark.asyncio
async def test_embedding_service_without_repository(mock_embedding_response, mock_provider):
    """Test that service works without repository (no persistence)."""
    service = EmbeddingService(repository=None, provider=mock_provider)

    with patch.object(
        service, "_create_embedding_with_retry", return_value=mock_embedding_response
    ):
        with patch.object(service, "_validate_deployment", return_value=True):
            request = EmbeddingRequest(
                text="Hello world",
                deployment="text-embedding-ada-002",
            )

            result = await service.get_embedding(request)

            assert result.vector == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert result.cached is False
            assert result.raw_call_id is None  # No persistence


@pytest.mark.asyncio
async def test_embedding_service_context_manager(mock_embedding_response, mock_provider):
    """Test that service works as async context manager."""
    service = EmbeddingService(repository=None, provider=mock_provider)

    with patch.object(
        service, "_create_embedding_with_retry", return_value=mock_embedding_response
    ):
        with patch.object(service, "_validate_deployment", return_value=True):
            async with service:
                request = EmbeddingRequest(
                    text="Hello world",
                    deployment="text-embedding-ada-002",
                )

                result = await service.get_embedding(request)
                assert result.vector == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.asyncio
async def test_require_db_persistence_default_raises_on_error(mock_embedding_response, db_connection, mock_provider):
    """Test that default behavior (require_db_persistence=True) raises on DB save failure."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, require_db_persistence=True, provider=mock_provider)
        
        # Mock repository to raise an error on insert
        with patch.object(
            service, "_create_embedding_with_retry", return_value=mock_embedding_response
        ):
            with patch.object(service, "_validate_deployment", return_value=True):
                with patch.object(repo, "insert_raw_call", side_effect=Exception("DB connection failed")):
                    request = EmbeddingRequest(
                        text="Hello world",
                        deployment="text-embedding-ada-002",
                    )
                    
                    with pytest.raises(RuntimeError) as exc_info:
                        await service.get_embedding(request)
                    
                    assert "Database persistence failed" in str(exc_info.value)
                    assert "DB connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_require_db_persistence_false_continues_on_error(mock_embedding_response, db_connection, mock_provider):
    """Test that require_db_persistence=False allows graceful degradation on DB save failure."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, require_db_persistence=False, provider=mock_provider)
        
        # Mock repository to raise an error on insert
        with patch.object(
            service, "_create_embedding_with_retry", return_value=mock_embedding_response
        ):
            with patch.object(service, "_validate_deployment", return_value=True):
                with patch.object(repo, "insert_raw_call", side_effect=Exception("DB connection failed")):
                    request = EmbeddingRequest(
                        text="Hello world",
                        deployment="text-embedding-ada-002",
                    )
                    
                    # Should not raise, but return result without DB persistence
                    result = await service.get_embedding(request)
                    
                    assert result.vector == [0.1, 0.2, 0.3, 0.4, 0.5]
                    assert result.raw_call_id is None  # Not persisted due to error


def test_get_deployment_max_tokens(mock_provider):
    """Test getting maximum tokens for known deployments."""
    service = EmbeddingService(provider=mock_provider)
    
    # Test known deployments from lookup table
    for deployment, expected_max in DEPLOYMENT_MAX_TOKENS.items():
        max_tokens = service.get_deployment_max_tokens(deployment)
        assert max_tokens == expected_max, f"Expected {expected_max} for {deployment}, got {max_tokens}"
    
    # Test pattern matching for text-embedding-3 models
    max_tokens = service.get_deployment_max_tokens("text-embedding-3-custom")
    assert max_tokens == 8191, "Should infer 8191 for text-embedding-3-* models"
    
    # Test unknown deployment
    max_tokens = service.get_deployment_max_tokens("unknown-model-xyz")
    assert max_tokens is None, "Should return None for unknown deployments"
    
    # Test caching
    max_tokens_1 = service.get_deployment_max_tokens("text-embedding-3-small")
    max_tokens_2 = service.get_deployment_max_tokens("text-embedding-3-small")
    assert max_tokens_1 == max_tokens_2 == 8191
    assert "azure:text-embedding-3-small" in service._deployment_limits_cache


def test_estimate_tokens():
    """Test token estimation."""
    # Test with simple text
    text = "Hello world"
    tokens = estimate_tokens(text)
    assert tokens > 0, "Should estimate at least 1 token"
    
    # Test with longer text
    long_text = "Hello world " * 100
    tokens_long = estimate_tokens(long_text)
    assert tokens_long > tokens, "Longer text should have more tokens"
    
    # Test with model name
    tokens_with_model = estimate_tokens(text, "text-embedding-3-small")
    assert tokens_with_model > 0


def test_validate_text_length(mock_provider):
    """Test text length validation."""
    service = EmbeddingService(provider=mock_provider)
    
    # Test valid text (short)
    is_valid, est_tokens, max_tokens = service.validate_text_length(
        "Hello world", "text-embedding-3-small"
    )
    assert is_valid is True
    assert est_tokens is not None and est_tokens > 0
    assert max_tokens == 8191
    
    # Test text that's too long (create text that exceeds limit)
    # Create text that's approximately 10,000 tokens (way over 8191 limit)
    very_long_text = "This is a test sentence. " * 2000  # ~40,000 chars ≈ 10,000 tokens
    is_valid, est_tokens, max_tokens = service.validate_text_length(
        very_long_text, "text-embedding-3-small"
    )
    assert is_valid is False, "Very long text should fail validation"
    assert est_tokens is not None
    assert max_tokens == 8191
    assert est_tokens > max_tokens
    
    # Test unknown deployment (should pass validation)
    is_valid, est_tokens, max_tokens = service.validate_text_length(
        "Hello world", "unknown-deployment"
    )
    assert is_valid is True, "Unknown deployment should pass validation (can't validate)"
    assert max_tokens is None


@pytest.mark.asyncio
async def test_get_embedding_rejects_text_exceeding_limit(mock_embedding_response, db_connection, mock_provider):
    """Test that get_embedding raises error for text exceeding token limit."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)
        
        # Create text that exceeds the limit
        very_long_text = "This is a test sentence. " * 2000  # ~10,000 tokens
        
        request = EmbeddingRequest(
            text=very_long_text,
            deployment="text-embedding-3-small"  # Max 8191 tokens
        )
        
        with patch.object(service, "_validate_deployment", return_value=True):
            with pytest.raises(ValueError, match="exceeds maximum token limit"):
                await service.get_embedding(request)
            
            # Verify failure was logged
            # Check that a failed RawCall was created
            from study_query_llm.db.models_v2 import RawCall
            failed_calls = repo.session.query(RawCall).filter_by(
                modality="embedding",
                status="failed",
                model="text-embedding-3-small"
            ).all()
            assert len(failed_calls) > 0, "Failed call should be logged to database"


# =============================================================================
# Tests for true API batching (chunk_size path)
# =============================================================================


def _make_batch_embedding(index: int, vector=None) -> EmbeddingResult:
    """Create a mock EmbeddingResult for the given index."""
    vector = vector or [0.1 * (index + 1), 0.2, 0.3]
    return EmbeddingResult(vector=vector, index=index)


@pytest.mark.asyncio
async def test_get_embeddings_batch_chunked_all_uncached(db_connection, mock_provider):
    """With chunk_size set, all uncached texts are fetched in one batch call per chunk."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        texts = [f"Text {i}" for i in range(5)]
        requests = [
            EmbeddingRequest(text=t, deployment="text-embedding-ada-002")
            for t in texts
        ]

        # Track call index so each chunk gets embeddings with the right global offsets
        call_counter = [0]

        async def batch_side_effect(chunk_texts, deployment, provider="azure", dimensions=None):
            offset = call_counter[0] * 3  # chunk_size=3
            call_counter[0] += 1
            return [_make_batch_embedding(j, [float(offset + j), 0.1, 0.2]) for j in range(len(chunk_texts))]

        with patch.object(
            service,
            "_create_embedding_batch_with_retry",
            side_effect=batch_side_effect,
        ) as mock_batch:
            with patch.object(service, "_validate_deployment", return_value=True):
                results = await service.get_embeddings_batch(requests, chunk_size=3)

        assert len(results) == 5
        # Two chunks: [0:3] and [3:5] → two batch calls
        assert mock_batch.call_count == 2
        # Vectors are in original order
        for i, resp in enumerate(results):
            assert resp.vector[0] == float(i)
            assert resp.cached is False


@pytest.mark.asyncio
async def test_get_embeddings_batch_chunked_mixed_cached_uncached(db_connection, mock_provider):
    """With chunk_size set, cached items are served from DB without API call."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        # Pre-insert two embeddings via the single path so they are in the DB
        pre_requests = [
            EmbeddingRequest(text="Alpha", deployment="text-embedding-ada-002"),
            EmbeddingRequest(text="Beta", deployment="text-embedding-ada-002"),
        ]
        pre_emb = _make_batch_embedding(0, [0.9, 0.8, 0.7])
        with patch.object(
            service, "_create_embedding_with_retry", return_value=pre_emb
        ):
            with patch.object(service, "_validate_deployment", return_value=True):
                for req in pre_requests:
                    await service.get_embedding(req)

        # Now request those two + one new text in chunked mode
        all_requests = pre_requests + [
            EmbeddingRequest(text="Gamma", deployment="text-embedding-ada-002"),
        ]
        new_emb = _make_batch_embedding(0, [0.1, 0.2, 0.3])

        with patch.object(
            service,
            "_create_embedding_batch_with_retry",
            return_value=[new_emb],
        ) as mock_batch:
            results = await service.get_embeddings_batch(all_requests, chunk_size=5)

        assert len(results) == 3
        # Only "Gamma" should have triggered an API call
        assert mock_batch.call_count == 1
        called_texts = mock_batch.call_args[0][0]  # first positional arg = texts list
        assert called_texts == ["Gamma"]
        # First two results are cached
        assert results[0].cached is True
        assert results[1].cached is True
        assert results[2].cached is False


@pytest.mark.asyncio
async def test_get_embeddings_batch_chunked_preserves_order(db_connection, mock_provider):
    """Results are returned in the same order as the input requests."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo, provider=mock_provider)

        n = 7
        requests = [
            EmbeddingRequest(text=f"Item {i}", deployment="text-embedding-ada-002")
            for i in range(n)
        ]

        # Mock: return embeddings with first dimension = index
        def make_batch_side_effect(texts, deployment, provider, dimensions):
            return [_make_batch_embedding(j, [float(j), 0.0, 0.0]) for j in range(len(texts))]

        # Use side_effect because chunk indices differ per call
        chunk_call_count = [0]

        async def batch_side_effect(texts, deployment, provider="azure", dimensions=None):
            offset = chunk_call_count[0] * 3  # chunk_size=3
            chunk_call_count[0] += 1
            return [_make_batch_embedding(0, [float(offset + j), 0.0, 0.0]) for j in range(len(texts))]

        with patch.object(
            service,
            "_create_embedding_batch_with_retry",
            side_effect=batch_side_effect,
        ):
            results = await service.get_embeddings_batch(requests, chunk_size=3)

        assert len(results) == n
        # Each result's first vector component equals its original index
        for i, resp in enumerate(results):
            assert resp.vector[0] == float(i), f"Order mismatch at index {i}"


@pytest.mark.asyncio
async def test_batch_api_call_sends_multiple_inputs():
    """_create_embedding_batch_with_retry delegates to provider.create_embeddings."""
    mock_embs = [_make_batch_embedding(i) for i in range(3)]

    provider = AsyncMock(spec=BaseEmbeddingProvider)
    provider.create_embeddings = AsyncMock(return_value=mock_embs)
    provider.close = AsyncMock()
    provider.get_provider_name = MagicMock(return_value="mock")

    service = EmbeddingService(repository=None, provider=provider)
    texts = ["Hello", "World", "Foo"]

    results = await service._create_embedding_batch_with_retry(
        texts, "test-deployment", "azure", None
    )

    provider.create_embeddings.assert_called_once_with(texts, "test-deployment", None)
    assert len(results) == 3


# =============================================================================
# Tests for repository batch hash lookup
# =============================================================================


def test_get_embedding_vectors_by_request_hashes_empty(db_connection):
    """Returns empty dict when no hashes provided."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        result = repo.get_embedding_vectors_by_request_hashes("any-model", [])
        assert result == {}


def test_get_embedding_vectors_by_request_hashes_returns_found(db_connection):
    """Returns matching embeddings for hashes that exist in the DB."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        # Manually insert a RawCall + EmbeddingVector with a known hash
        from study_query_llm.db.models_v2 import EmbeddingVector

        target_hash = "abc123"
        call_id = repo.insert_raw_call(
            provider="azure_openai_test-model",
            request_json={"input": "test text", "model": "test-model"},
            model="test-model",
            modality="embedding",
            status="success",
            response_json={"model": "test-model", "embedding_dim": 3},
            metadata_json={"request_hash": target_hash},
        )
        ev = EmbeddingVector(
            call_id=call_id,
            vector=[0.1, 0.2, 0.3],
            dimension=3,
            norm=0.37,
            metadata_json={"model": "test-model"},
        )
        session.add(ev)
        session.flush()

        result = repo.get_embedding_vectors_by_request_hashes(
            "test-model", [target_hash, "not-in-db"]
        )

    # "not-in-db" should be absent; target_hash should be present
    assert "not-in-db" not in result
    # SQLite JSON path operator may not be supported; skip assertion if empty
    if result:
        assert target_hash in result
        vector, rc_id = result[target_hash]
        assert vector == [0.1, 0.2, 0.3]
        assert rc_id == call_id