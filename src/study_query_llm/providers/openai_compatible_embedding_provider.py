"""
OpenAI-Compatible Embedding Provider.

Uses the vanilla ``AsyncOpenAI`` client with a configurable ``base_url``,
making it work with any endpoint that speaks the OpenAI embeddings protocol:
HuggingFace TEI, Ollama, vLLM, Together AI, Fireworks, direct OpenAI, etc.
"""

from typing import List, Optional

from openai import AsyncOpenAI

from .base_embedding import BaseEmbeddingProvider, EmbeddingResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider for any OpenAI-protocol-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        provider_label: str = "openai_compatible",
    ) -> None:
        """
        Args:
            base_url: Root URL of the embedding endpoint
                      (e.g. ``http://localhost:8080/v1``).
            api_key: API key / bearer token. Defaults to ``"not-needed"``
                     for local servers that don't require auth.
            provider_label: Human-readable label returned by
                            ``get_provider_name()``.
        """
        self._base_url = base_url
        self._provider_label = provider_label
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        logger.info(
            "Initialized OpenAICompatibleEmbeddingProvider (base_url=%s, label=%s)",
            base_url,
            provider_label,
        )

    # Extra fields forwarded verbatim to the request body (e.g. TEI's
    # ``prompt_name``).  Set via subclass __init__, not exposed in the ABC.
    _extra_body: Optional[dict] = None

    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        params: dict = {"model": model, "input": texts}
        if dimensions is not None:
            params["dimensions"] = dimensions
        if self._extra_body:
            params["extra_body"] = self._extra_body

        response = await self._client.embeddings.create(**params)

        sorted_data = sorted(response.data, key=lambda e: e.index)
        return [
            EmbeddingResult(vector=item.embedding, index=item.index)
            for item in sorted_data
        ]

    def get_provider_name(self) -> str:
        return self._provider_label

    async def close(self) -> None:
        await self._client.close()
