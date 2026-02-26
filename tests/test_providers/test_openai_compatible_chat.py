"""Tests for OpenAICompatibleChatProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from study_query_llm.providers.openai_compatible_chat_provider import (
    OpenAICompatibleChatProvider,
)
from study_query_llm.providers.base import ProviderResponse


@pytest.fixture
def provider():
    """Create an OpenAICompatibleChatProvider with mocked client."""
    with patch(
        "study_query_llm.providers.openai_compatible_chat_provider.AsyncOpenAI"
    ) as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        prov = OpenAICompatibleChatProvider(
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
            api_key="not-needed",
            provider_label="test_local",
        )
        prov._mock_client = mock_client
        yield prov


def _mock_chat_response(content="Mock summary", finish_reason="stop"):
    """Build a mock ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.total_tokens = 42
    usage.prompt_tokens = 10
    usage.completion_tokens = 32

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.mark.asyncio
async def test_complete_returns_provider_response(provider):
    """complete() returns a ProviderResponse with correct fields."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response("Test reply")
    )

    result = await provider.complete("Hello", temperature=0.5, max_tokens=100)

    assert isinstance(result, ProviderResponse)
    assert result.text == "Test reply"
    assert result.provider == "test_local"
    assert result.tokens == 42
    assert result.latency_ms >= 0
    assert result.metadata["model"] == "llama3.1:8b"
    assert result.metadata["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_complete_passes_model_from_constructor(provider):
    """The model baked in at construction is used in the API call."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response()
    )

    await provider.complete("test prompt")

    call_kwargs = provider._mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "llama3.1:8b"


@pytest.mark.asyncio
async def test_complete_passes_temperature_and_max_tokens(provider):
    """temperature and max_tokens are forwarded to the API."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response()
    )

    await provider.complete("prompt", temperature=0.3, max_tokens=256)

    call_kwargs = provider._mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["temperature"] == 0.3
    assert call_kwargs.kwargs["max_tokens"] == 256


@pytest.mark.asyncio
async def test_complete_omits_max_tokens_when_none(provider):
    """max_tokens is not sent when None (let server use its default)."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response()
    )

    await provider.complete("prompt", max_tokens=None)

    call_kwargs = provider._mock_client.chat.completions.create.call_args
    assert "max_tokens" not in call_kwargs.kwargs


def test_get_provider_name_returns_label():
    """Provider name matches the label passed at construction."""
    with patch(
        "study_query_llm.providers.openai_compatible_chat_provider.AsyncOpenAI"
    ):
        prov = OpenAICompatibleChatProvider(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b",
            provider_label="my_ollama",
        )
        assert prov.get_provider_name() == "my_ollama"


@pytest.mark.asyncio
async def test_close_delegates(provider):
    """close() calls the underlying client's close()."""
    await provider.close()
    provider._mock_client.close.assert_called_once()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "https://my-vllm-server.example.com/v1",
        "http://192.168.1.42:8000/v1",
    ],
)
def test_init_accepts_various_base_urls(base_url):
    """Provider can be created with any base_url."""
    with patch(
        "study_query_llm.providers.openai_compatible_chat_provider.AsyncOpenAI"
    ) as MockClient:
        prov = OpenAICompatibleChatProvider(
            base_url=base_url, model="test-model"
        )
        MockClient.assert_called_once_with(
            base_url=base_url, api_key="not-needed"
        )
        assert prov.get_provider_name() == "local_llm"


@pytest.mark.asyncio
async def test_complete_handles_none_content(provider):
    """Gracefully handles None content from the API (e.g. refusal)."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response(content=None)
    )

    result = await provider.complete("prompt")
    assert result.text == ""
