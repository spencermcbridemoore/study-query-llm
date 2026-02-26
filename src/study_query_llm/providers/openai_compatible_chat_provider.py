"""
OpenAI-Compatible Chat Provider.

Uses the vanilla ``AsyncOpenAI`` client with a configurable ``base_url``,
making it work with any endpoint that speaks the OpenAI chat completions
protocol: Ollama, vLLM, Together AI, Fireworks, direct OpenAI, etc.

The model is baked in at construction time so each instance is tied to a
specific model -- the factory passes it as an argument rather than reading
it from an env var.  This is intentional: env vars configure infrastructure
(endpoint, api key), while model selection happens at call-site in sweep
configs.
"""

import time
from typing import Any, Optional

from openai import AsyncOpenAI

from .base import BaseLLMProvider, ProviderResponse
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class OpenAICompatibleChatProvider(BaseLLMProvider):
    """Chat provider for any OpenAI-protocol-compatible endpoint.

    Works with Ollama (``http://localhost:11434/v1``), vLLM, Together AI,
    Fireworks, plain OpenAI, and any other server that exposes the
    ``/chat/completions`` endpoint in the OpenAI format.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        provider_label: str = "local_llm",
    ) -> None:
        """
        Args:
            base_url: Root URL of the chat endpoint
                      (e.g. ``http://localhost:11434/v1``).
            model: Model identifier to use in every request
                   (e.g. ``"qwen2.5:32b"``).  This is passed as the
                   ``model`` field in the request body, not read from env.
            api_key: API key / bearer token.  Defaults to ``"not-needed"``
                     for local servers that do not require auth.
            provider_label: Human-readable label returned by
                            ``get_provider_name()``.
        """
        self._base_url = base_url
        self._model = model
        self._provider_label = provider_label
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        logger.info(
            "Initialized OpenAICompatibleChatProvider "
            "(base_url=%s, model=%s, label=%s)",
            base_url,
            model,
            provider_label,
        )

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a chat completion request.

        Args:
            prompt: User message to send.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens to generate.  ``None`` lets the
                        server use its default.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            ``ProviderResponse`` with the assistant reply and metadata.
        """
        start_time = time.time()

        params: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**params)

        latency_ms = (time.time() - start_time) * 1000
        choice = response.choices[0]
        text = choice.message.content or ""

        usage = response.usage
        tokens = usage.total_tokens if usage else None

        return ProviderResponse(
            text=text,
            provider=self._provider_label,
            tokens=tokens,
            latency_ms=latency_ms,
            metadata={
                "model": self._model,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            },
            raw_response=response,
        )

    def get_provider_name(self) -> str:
        return self._provider_label

    async def close(self) -> None:
        await self._client.close()
