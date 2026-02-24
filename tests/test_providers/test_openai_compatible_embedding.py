"""Tests for OpenAICompatibleEmbeddingProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.embedding import Embedding as EmbeddingObj

from study_query_llm.providers.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingProvider,
)
from study_query_llm.providers.base_embedding import EmbeddingResult


@pytest.fixture
def provider():
    """Create an OpenAICompatibleEmbeddingProvider with mocked client."""
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        prov = OpenAICompatibleEmbeddingProvider(
            base_url="http://localhost:8080/v1",
            api_key="not-needed",
            provider_label="test_local",
        )
        prov._mock_client = mock_client
        yield prov


@pytest.mark.asyncio
async def test_create_embeddings_returns_embedding_results(provider):
    """create_embeddings returns EmbeddingResult list."""
    sdk_embeddings = [
        EmbeddingObj(embedding=[1.0, 2.0], index=0, object="embedding"),
        EmbeddingObj(embedding=[3.0, 4.0], index=1, object="embedding"),
    ]
    mock_response = MagicMock()
    mock_response.data = sdk_embeddings
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    results = await provider.create_embeddings(["a", "b"], "bge-m3")

    assert len(results) == 2
    assert isinstance(results[0], EmbeddingResult)
    assert results[0].vector == [1.0, 2.0]
    assert results[1].vector == [3.0, 4.0]


@pytest.mark.asyncio
async def test_create_embeddings_passes_dimensions(provider):
    """dimensions kwarg is forwarded when provided."""
    sdk_emb = EmbeddingObj(embedding=[0.5], index=0, object="embedding")
    mock_response = MagicMock()
    mock_response.data = [sdk_emb]
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    await provider.create_embeddings(["text"], "bge-m3", dimensions=128)

    provider._mock_client.embeddings.create.assert_called_once_with(
        model="bge-m3", input=["text"], dimensions=128
    )


@pytest.mark.asyncio
async def test_create_embeddings_omits_dimensions_when_none(provider):
    """dimensions kwarg is omitted when None (some servers reject it)."""
    sdk_emb = EmbeddingObj(embedding=[0.5], index=0, object="embedding")
    mock_response = MagicMock()
    mock_response.data = [sdk_emb]
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    await provider.create_embeddings(["text"], "bge-m3", dimensions=None)

    provider._mock_client.embeddings.create.assert_called_once_with(
        model="bge-m3", input=["text"]
    )


@pytest.mark.asyncio
async def test_validate_model_defaults_to_true(provider):
    """Default validate_model always returns True."""
    assert await provider.validate_model("any-model") is True


@pytest.mark.asyncio
async def test_close_delegates(provider):
    """close() calls the underlying client's close()."""
    await provider.close()
    provider._mock_client.close.assert_called_once()


def test_get_provider_name_returns_label():
    """Provider name matches the label passed at construction."""
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ):
        prov = OpenAICompatibleEmbeddingProvider(
            base_url="http://localhost:8080/v1",
            provider_label="huggingface",
        )
        assert prov.get_provider_name() == "huggingface"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8080/v1",
        "https://xyz.endpoints.huggingface.cloud/v1",
        "http://192.168.1.42:11434/v1",
    ],
)
def test_init_accepts_various_base_urls(base_url):
    """Provider can be created with any base_url."""
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        prov = OpenAICompatibleEmbeddingProvider(base_url=base_url)
        MockClient.assert_called_once_with(base_url=base_url, api_key="not-needed")
        assert prov.get_provider_name() == "openai_compatible"
