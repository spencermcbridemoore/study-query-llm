"""Tests for AzureEmbeddingProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.embedding import Embedding as EmbeddingObj

from study_query_llm.providers.azure_embedding_provider import AzureEmbeddingProvider
from study_query_llm.providers.base_embedding import EmbeddingResult
from study_query_llm.config import ProviderConfig


def _make_azure_config() -> ProviderConfig:
    return ProviderConfig(
        name="azure",
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        api_version="2024-02-15-preview",
    )


@pytest.fixture
def provider():
    """Create an AzureEmbeddingProvider with mocked client."""
    with patch(
        "study_query_llm.providers.azure_embedding_provider.AsyncAzureOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        prov = AzureEmbeddingProvider(_make_azure_config())
        prov._mock_client = mock_client  # expose for assertions
        yield prov


@pytest.mark.asyncio
async def test_create_embeddings_returns_embedding_results(provider):
    """create_embeddings maps OpenAI SDK Embedding objects to EmbeddingResult."""
    sdk_embeddings = [
        EmbeddingObj(embedding=[0.1, 0.2], index=0, object="embedding"),
        EmbeddingObj(embedding=[0.3, 0.4], index=1, object="embedding"),
    ]
    mock_response = MagicMock()
    mock_response.data = sdk_embeddings
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    results = await provider.create_embeddings(["hello", "world"], "ada-002")

    assert len(results) == 2
    assert isinstance(results[0], EmbeddingResult)
    assert results[0].vector == [0.1, 0.2]
    assert results[0].index == 0
    assert results[1].vector == [0.3, 0.4]
    assert results[1].index == 1

    provider._mock_client.embeddings.create.assert_called_once_with(
        model="ada-002", input=["hello", "world"]
    )


@pytest.mark.asyncio
async def test_create_embeddings_passes_dimensions(provider):
    """dimensions kwarg is forwarded to the SDK when provided."""
    sdk_emb = EmbeddingObj(embedding=[0.5], index=0, object="embedding")
    mock_response = MagicMock()
    mock_response.data = [sdk_emb]
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    await provider.create_embeddings(["text"], "ada-002", dimensions=256)

    provider._mock_client.embeddings.create.assert_called_once_with(
        model="ada-002", input=["text"], dimensions=256
    )


@pytest.mark.asyncio
async def test_create_embeddings_sorts_by_index(provider):
    """Results are returned sorted by index even if the API returns them out of order."""
    sdk_embeddings = [
        EmbeddingObj(embedding=[0.9], index=2, object="embedding"),
        EmbeddingObj(embedding=[0.1], index=0, object="embedding"),
        EmbeddingObj(embedding=[0.5], index=1, object="embedding"),
    ]
    mock_response = MagicMock()
    mock_response.data = sdk_embeddings
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    results = await provider.create_embeddings(["a", "b", "c"], "m")

    assert [r.index for r in results] == [0, 1, 2]
    assert [r.vector for r in results] == [[0.1], [0.5], [0.9]]


@pytest.mark.asyncio
async def test_validate_model_success(provider):
    """validate_model returns True when the probe call succeeds."""
    sdk_emb = EmbeddingObj(embedding=[0.0], index=0, object="embedding")
    mock_response = MagicMock()
    mock_response.data = [sdk_emb]
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    assert await provider.validate_model("good-deployment") is True


@pytest.mark.asyncio
async def test_validate_model_failure(provider):
    """validate_model returns False when the probe call raises."""
    provider._mock_client.embeddings.create = AsyncMock(
        side_effect=Exception("Not found")
    )

    assert await provider.validate_model("bad-deployment") is False


@pytest.mark.asyncio
async def test_close_delegates_to_client(provider):
    """close() calls the underlying client's close()."""
    await provider.close()
    provider._mock_client.close.assert_called_once()


def test_get_provider_name():
    """Provider name is 'azure'."""
    with patch(
        "study_query_llm.providers.azure_embedding_provider.AsyncAzureOpenAI"
    ):
        prov = AzureEmbeddingProvider(_make_azure_config())
        assert prov.get_provider_name() == "azure"


def test_init_requires_endpoint():
    """Missing endpoint raises ValueError."""
    config = ProviderConfig(
        name="azure",
        api_key="key",
        endpoint=None,
        api_version="2024-02-15-preview",
    )
    with pytest.raises(ValueError, match="endpoint"):
        AzureEmbeddingProvider(config)


def test_init_requires_api_version():
    """Missing api_version raises ValueError."""
    config = ProviderConfig(
        name="azure",
        api_key="key",
        endpoint="https://test.openai.azure.com/",
        api_version=None,
    )
    with pytest.raises(ValueError, match="API version"):
        AzureEmbeddingProvider(config)
