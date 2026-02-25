"""Tests for ManagedTEIEmbeddingProvider.

The provider accepts any manager that satisfies the duck-typed contract:
    endpoint_url, model_id, provider_label, ping()

Tests run with both an ACI-style mock and a local-Docker-style mock to verify
the provider works correctly regardless of the backing manager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.embedding import Embedding as EmbeddingObj

from study_query_llm.providers.managed_tei_embedding_provider import (
    ManagedTEIEmbeddingProvider,
    get_prompt_name_for_model,
    _INSTRUCT_MODEL_PROMPT_NAMES,
)
from study_query_llm.providers.base_embedding import EmbeddingResult


# ---------------------------------------------------------------------------
# Manager mock factories
# ---------------------------------------------------------------------------

def _make_aci_manager(endpoint_url="http://20.1.2.3:80/v1", model_id="BAAI/bge-m3"):
    """Mock manager that looks like ACITEIManager."""
    manager = MagicMock()
    manager.endpoint_url = endpoint_url
    manager.model_id = model_id
    manager.provider_label = "aci_tei"
    manager.ping = MagicMock()
    return manager


def _make_docker_manager(endpoint_url="http://localhost:8080/v1", model_id="BAAI/bge-m3"):
    """Mock manager that looks like LocalDockerTEIManager."""
    manager = MagicMock()
    manager.endpoint_url = endpoint_url
    manager.model_id = model_id
    manager.provider_label = "local_docker_tei"
    manager.ping = MagicMock()
    return manager


# ---------------------------------------------------------------------------
# Parameterised fixture: run each test with both manager types
# ---------------------------------------------------------------------------

@pytest.fixture(params=["aci", "docker"])
def manager(request):
    if request.param == "aci":
        return _make_aci_manager()
    return _make_docker_manager()


@pytest.fixture
def provider(manager):
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        prov = ManagedTEIEmbeddingProvider(manager)
        prov._mock_client = mock_client
        yield prov


# ---------------------------------------------------------------------------
# provider_label / get_provider_name
# ---------------------------------------------------------------------------

def test_get_provider_name_matches_manager_label():
    """provider_label is taken from the manager, not hardcoded."""
    for mk, expected in [(_make_aci_manager(), "aci_tei"), (_make_docker_manager(), "local_docker_tei")]:
        with patch("study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"):
            prov = ManagedTEIEmbeddingProvider(mk)
            assert prov.get_provider_name() == expected


# ---------------------------------------------------------------------------
# create_embeddings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_embeddings_pings_manager(provider, manager):
    """create_embeddings calls ping() on the manager before the API call."""
    sdk_emb = EmbeddingObj(embedding=[0.1, 0.2], index=0, object="embedding")
    mock_response = MagicMock()
    mock_response.data = [sdk_emb]
    provider._mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    await provider.create_embeddings(["hello"], "BAAI/bge-m3")

    manager.ping.assert_called_once()


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
        prov = ManagedTEIEmbeddingProvider(manager)

        await prov.create_embeddings(["text"], "BAAI/bge-m3")

    assert call_order == ["ping", "api"]


@pytest.mark.asyncio
async def test_create_embeddings_returns_embedding_results(provider, manager):
    """create_embeddings returns EmbeddingResult objects."""
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


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_does_not_stop_container(provider, manager):
    """close() closes the HTTP client but does NOT stop the container."""
    await provider.close()

    manager.stop.assert_not_called()
    manager.delete.assert_not_called()
    provider._mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_raises_if_endpoint_not_set():
    """Raises ValueError when manager.endpoint_url is None."""
    manager = MagicMock()
    manager.endpoint_url = None
    manager.model_id = "BAAI/bge-m3"
    manager.provider_label = "aci_tei"

    with pytest.raises(ValueError, match="endpoint_url is None"):
        ManagedTEIEmbeddingProvider(manager)


def test_constructor_passes_endpoint_to_parent():
    """The OpenAICompatibleEmbeddingProvider receives the correct base_url."""
    manager = _make_aci_manager(endpoint_url="http://1.2.3.4:80/v1")

    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        ManagedTEIEmbeddingProvider(manager)
        MockClient.assert_called_once_with(
            base_url="http://1.2.3.4:80/v1", api_key="not-needed"
        )


def test_constructor_local_docker_endpoint():
    """Works correctly with a localhost endpoint (LocalDockerTEIManager)."""
    manager = _make_docker_manager(endpoint_url="http://localhost:8080/v1")

    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        ManagedTEIEmbeddingProvider(manager)
        MockClient.assert_called_once_with(
            base_url="http://localhost:8080/v1", api_key="not-needed"
        )


# ---------------------------------------------------------------------------
# validate_model default
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_model_defaults_to_true(provider):
    """Default validate_model always returns True (inherited from base)."""
    assert await provider.validate_model("any-model") is True


# ---------------------------------------------------------------------------
# get_prompt_name_for_model helper
# ---------------------------------------------------------------------------

def test_get_prompt_name_for_known_instruct_model():
    """Known instruct models return 'query'."""
    assert get_prompt_name_for_model("Qwen/Qwen3-Embedding-8B") == "query"
    assert get_prompt_name_for_model("Alibaba-NLP/gte-Qwen2-7B-instruct") == "query"
    assert get_prompt_name_for_model("intfloat/multilingual-e5-large-instruct") == "query"


def test_get_prompt_name_for_unknown_model_returns_none():
    """Non-instruct models return None."""
    assert get_prompt_name_for_model("BAAI/bge-m3") is None
    assert get_prompt_name_for_model("nomic-ai/nomic-embed-text-v1.5") is None
    assert get_prompt_name_for_model("not-a-real-model") is None


def test_instruct_model_registry_all_have_string_prompt_names():
    """Every entry in the registry maps to a non-empty string."""
    for model_id, name in _INSTRUCT_MODEL_PROMPT_NAMES.items():
        assert isinstance(name, str) and name, f"{model_id!r} has invalid prompt_name {name!r}"


# ---------------------------------------------------------------------------
# prompt_name auto-detection (prompt_name="auto", the default)
# ---------------------------------------------------------------------------

def test_auto_sets_extra_body_for_instruct_model():
    """Auto-mode sets _extra_body when model_id is in the registry."""
    manager = MagicMock()
    manager.endpoint_url = "http://localhost:8080/v1"
    manager.model_id = "Qwen/Qwen3-Embedding-8B"
    manager.provider_label = "local_docker_tei"

    with patch("study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"):
        prov = ManagedTEIEmbeddingProvider(manager)

    assert prov._extra_body == {"prompt_name": "query"}


def test_auto_leaves_extra_body_none_for_non_instruct_model():
    """Auto-mode does NOT set _extra_body for models not in the registry."""
    manager = MagicMock()
    manager.endpoint_url = "http://localhost:8080/v1"
    manager.model_id = "BAAI/bge-m3"
    manager.provider_label = "local_docker_tei"

    with patch("study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"):
        prov = ManagedTEIEmbeddingProvider(manager)

    assert prov._extra_body is None


# ---------------------------------------------------------------------------
# prompt_name explicit overrides
# ---------------------------------------------------------------------------

def test_explicit_prompt_name_overrides_registry():
    """An explicit prompt_name string is used regardless of the registry."""
    manager = MagicMock()
    manager.endpoint_url = "http://localhost:8080/v1"
    manager.model_id = "BAAI/bge-m3"  # not an instruct model
    manager.provider_label = "local_docker_tei"

    with patch("study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"):
        prov = ManagedTEIEmbeddingProvider(manager, prompt_name="passage")

    assert prov._extra_body == {"prompt_name": "passage"}


def test_prompt_name_none_disables_for_known_instruct_model():
    """prompt_name=None suppresses prompt injection even for a registered model."""
    manager = MagicMock()
    manager.endpoint_url = "http://localhost:8080/v1"
    manager.model_id = "Qwen/Qwen3-Embedding-8B"
    manager.provider_label = "local_docker_tei"

    with patch("study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"):
        prov = ManagedTEIEmbeddingProvider(manager, prompt_name=None)

    assert prov._extra_body is None


# ---------------------------------------------------------------------------
# extra_body is forwarded to the API call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extra_body_passed_to_api_for_instruct_model():
    """For instruct models, extra_body={'prompt_name': 'query'} is sent to the API."""
    manager = MagicMock()
    manager.endpoint_url = "http://localhost:8080/v1"
    manager.model_id = "Qwen/Qwen3-Embedding-4B"
    manager.provider_label = "local_docker_tei"
    manager.ping = MagicMock()

    api_kwargs = {}

    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()

        async def capture_create(**kwargs):
            api_kwargs.update(kwargs)
            sdk_emb = EmbeddingObj(embedding=[0.1, 0.2], index=0, object="embedding")
            resp = MagicMock()
            resp.data = [sdk_emb]
            return resp

        mock_client.embeddings.create = capture_create
        MockClient.return_value = mock_client
        prov = ManagedTEIEmbeddingProvider(manager)
        await prov.create_embeddings(["test"], "Qwen/Qwen3-Embedding-4B")

    assert api_kwargs.get("extra_body") == {"prompt_name": "query"}


@pytest.mark.asyncio
async def test_no_extra_body_for_non_instruct_model():
    """For non-instruct models, extra_body is not included in the API call."""
    manager = MagicMock()
    manager.endpoint_url = "http://localhost:8080/v1"
    manager.model_id = "BAAI/bge-m3"
    manager.provider_label = "local_docker_tei"
    manager.ping = MagicMock()

    api_kwargs = {}

    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()

        async def capture_create(**kwargs):
            api_kwargs.update(kwargs)
            sdk_emb = EmbeddingObj(embedding=[0.1, 0.2], index=0, object="embedding")
            resp = MagicMock()
            resp.data = [sdk_emb]
            return resp

        mock_client.embeddings.create = capture_create
        MockClient.return_value = mock_client
        prov = ManagedTEIEmbeddingProvider(manager)
        await prov.create_embeddings(["test"], "BAAI/bge-m3")

    assert "extra_body" not in api_kwargs
