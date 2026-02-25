"""Tests for ACITEIEmbeddingProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.embedding import Embedding as EmbeddingObj

from study_query_llm.providers.aci_tei_embedding_provider import ACITEIEmbeddingProvider
from study_query_llm.providers.base_embedding import EmbeddingResult


def _make_manager(endpoint_url="http://1.2.3.4:80/v1", model_id="BAAI/bge-m3"):
    """Build a minimal mock ACITEIManager with the required attributes."""
    manager = MagicMock()
    manager.endpoint_url = endpoint_url
    manager.model_id = model_id
    manager.ping = MagicMock()
    return manager


@pytest.fixture
def manager():
    return _make_manager()


@pytest.fixture
def provider(manager):
    """Create ACITEIEmbeddingProvider with a mocked AsyncOpenAI client."""
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        prov = ACITEIEmbeddingProvider(manager)
        prov._mock_client = mock_client
        yield prov


@pytest.mark.asyncio
async def test_create_embeddings_pings_manager(provider, manager):
    """create_embeddings resets the idle timer by calling ping() before the API call."""
    sdk_emb = EmbeddingObj(embedding=[0.1, 0.2], index=0, object="embedding")
    mock_response = MagicMock()
    mock_response.data = [sdk_emb]
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    await provider.create_embeddings(["hello"], "BAAI/bge-m3")

    manager.ping.assert_called_once()


@pytest.mark.asyncio
async def test_create_embeddings_returns_embedding_results(provider, manager):
    """create_embeddings delegates to parent and returns EmbeddingResult objects."""
    sdk_embeddings = [
        EmbeddingObj(embedding=[1.0, 2.0], index=0, object="embedding"),
        EmbeddingObj(embedding=[3.0, 4.0], index=1, object="embedding"),
    ]
    mock_response = MagicMock()
    mock_response.data = sdk_embeddings
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    results = await provider.create_embeddings(["a", "b"], "BAAI/bge-m3")

    assert len(results) == 2
    assert isinstance(results[0], EmbeddingResult)
    assert results[0].vector == [1.0, 2.0]
    assert results[1].vector == [3.0, 4.0]


@pytest.mark.asyncio
async def test_create_embeddings_ping_before_api_call(manager):
    """ping() is called before the API call (not after a failure)."""
    call_order = []

    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()

        async def tracked_create(**kwargs):
            call_order.append("api")
            sdk_emb = EmbeddingObj(embedding=[0.5], index=0, object="embedding")
            resp = MagicMock()
            resp.data = [sdk_emb]
            return resp

        mock_client.embeddings.create = tracked_create
        MockClient.return_value = mock_client

        manager.ping.side_effect = lambda: call_order.append("ping")
        prov = ACITEIEmbeddingProvider(manager)

        await prov.create_embeddings(["text"], "BAAI/bge-m3")

    assert call_order == ["ping", "api"], (
        "ping() must be called before the API call so that even a long batch "
        "keeps the container alive."
    )


def test_get_provider_name_returns_aci_tei(manager):
    """Provider name is 'aci_tei'."""
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ):
        prov = ACITEIEmbeddingProvider(manager)
        assert prov.get_provider_name() == "aci_tei"


@pytest.mark.asyncio
async def test_close_does_not_call_manager_delete(provider, manager):
    """close() closes the HTTP client but does NOT delete the ACI container."""
    await provider.close()

    manager.delete.assert_not_called()
    provider._mock_client.close.assert_called_once()


def test_constructor_raises_if_endpoint_not_set():
    """ACITEIEmbeddingProvider raises ValueError if manager.endpoint_url is None."""
    manager = MagicMock()
    manager.endpoint_url = None
    manager.model_id = "BAAI/bge-m3"

    with pytest.raises(ValueError, match="endpoint_url is None"):
        ACITEIEmbeddingProvider(manager)


def test_constructor_passes_endpoint_to_parent(manager):
    """The parent OpenAICompatibleEmbeddingProvider receives the correct base_url."""
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        ACITEIEmbeddingProvider(manager)
        MockClient.assert_called_once_with(
            base_url="http://1.2.3.4:80/v1", api_key="not-needed"
        )


@pytest.mark.asyncio
async def test_validate_model_defaults_to_true(provider):
    """Default validate_model always returns True (inherited from base)."""
    assert await provider.validate_model("any-model") is True
