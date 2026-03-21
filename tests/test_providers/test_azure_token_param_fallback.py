"""Unit tests for Azure token-parameter fallback behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from study_query_llm.config import ProviderConfig
from study_query_llm.providers.azure_provider import AzureOpenAIProvider


def _mock_chat_response(content: str = "ok", model: str = "test-model"):
    """Build a mock ChatCompletion-like response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.total_tokens = 42
    usage.prompt_tokens = 10
    usage.completion_tokens = 32

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


@pytest.fixture
def provider():
    """Create an Azure provider with mocked AsyncAzureOpenAI client."""
    with patch("study_query_llm.providers.azure_provider.AsyncAzureOpenAI") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value = mock_client
        cfg = ProviderConfig(
            name="azure",
            api_key="test-key",
            endpoint="https://example.openai.azure.com/",
            deployment_name="gpt-5.4-nano",
            api_version="2024-02-15-preview",
        )
        prov = AzureOpenAIProvider(cfg)
        prov._mock_client = mock_client
        yield prov


@pytest.mark.asyncio
async def test_complete_uses_max_tokens_when_supported(provider):
    """Provider keeps max_tokens when API accepts it."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response()
    )

    result = await provider.complete("hello", max_tokens=128)

    call_kwargs = provider._mock_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["max_tokens"] == 128
    assert "max_completion_tokens" not in call_kwargs
    assert result.metadata["max_tokens"] == 128
    assert result.metadata["token_limit_param"] == "max_tokens"


@pytest.mark.asyncio
async def test_complete_retries_with_max_completion_tokens_when_required(provider):
    """Provider retries once with max_completion_tokens on unsupported_parameter."""
    unsupported_error = Exception(
        "Error code: 400 - {'error': {'message': \"Unsupported parameter: "
        "'max_tokens' is not supported with this model. Use "
        "'max_completion_tokens' instead.\", 'type': 'invalid_request_error', "
        "'param': 'max_tokens', 'code': 'unsupported_parameter'}}"
    )
    provider._mock_client.chat.completions.create = AsyncMock(
        side_effect=[unsupported_error, _mock_chat_response(model="gpt-5.4-nano")]
    )

    result = await provider.complete("hello", max_tokens=256)

    assert provider._mock_client.chat.completions.create.await_count == 2
    first_call = provider._mock_client.chat.completions.create.await_args_list[0].kwargs
    second_call = provider._mock_client.chat.completions.create.await_args_list[1].kwargs

    assert first_call["max_tokens"] == 256
    assert "max_completion_tokens" not in first_call

    assert second_call["max_completion_tokens"] == 256
    assert "max_tokens" not in second_call

    assert result.metadata["max_tokens"] == 256
    assert result.metadata["token_limit_param"] == "max_completion_tokens"


@pytest.mark.asyncio
async def test_complete_respects_explicit_max_completion_tokens(provider):
    """If caller passes max_completion_tokens, provider does not add max_tokens."""
    provider._mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response()
    )

    result = await provider.complete(
        "hello", max_tokens=512, max_completion_tokens=64
    )

    call_kwargs = provider._mock_client.chat.completions.create.await_args.kwargs
    assert "max_tokens" not in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 64
    assert result.metadata["token_limit_param"] == "max_completion_tokens"


@pytest.mark.asyncio
async def test_complete_caches_preferred_token_limit_param(provider):
    """After first fallback, subsequent calls should use max_completion_tokens directly."""
    unsupported_error = Exception(
        "Error code: 400 - {'error': {'message': \"Unsupported parameter: "
        "'max_tokens' is not supported with this model. Use "
        "'max_completion_tokens' instead.\", 'type': 'invalid_request_error', "
        "'param': 'max_tokens', 'code': 'unsupported_parameter'}}"
    )
    provider._mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            unsupported_error,        # first call attempt 1
            _mock_chat_response(),    # first call fallback
            _mock_chat_response(),    # second call (should use cached preference)
        ]
    )

    # First call learns fallback behavior
    first = await provider.complete("first", max_tokens=32)
    assert first.metadata["token_limit_param"] == "max_completion_tokens"

    # Second call should not attempt max_tokens first
    second = await provider.complete("second", max_tokens=32)
    assert second.metadata["token_limit_param"] == "max_completion_tokens"

    assert provider._mock_client.chat.completions.create.await_count == 3
    first_attempt = provider._mock_client.chat.completions.create.await_args_list[0].kwargs
    first_fallback = provider._mock_client.chat.completions.create.await_args_list[1].kwargs
    second_call = provider._mock_client.chat.completions.create.await_args_list[2].kwargs

    assert "max_tokens" in first_attempt
    assert "max_completion_tokens" not in first_attempt
    assert "max_completion_tokens" in first_fallback
    assert "max_tokens" not in first_fallback
    assert "max_completion_tokens" in second_call
    assert "max_tokens" not in second_call
