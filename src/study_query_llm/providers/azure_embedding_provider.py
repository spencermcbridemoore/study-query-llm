"""
Azure OpenAI Embedding Provider.

Wraps ``AsyncAzureOpenAI`` to implement the ``BaseEmbeddingProvider``
interface for Azure-hosted embedding deployments.
"""

from typing import List, Optional

from openai import AsyncAzureOpenAI

from .base_embedding import BaseEmbeddingProvider, EmbeddingResult
from ..config import ProviderConfig
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by Azure OpenAI Service."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.endpoint:
            raise ValueError("Azure endpoint is required")
        if not config.api_version:
            raise ValueError("Azure API version is required")

        self._config = config
        self._client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
        )
        logger.info(
            "Initialized AzureEmbeddingProvider (endpoint=%s, api_version=%s)",
            config.endpoint,
            config.api_version,
        )

    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        params: dict = {"model": model, "input": texts}
        if dimensions is not None:
            params["dimensions"] = dimensions

        response = await self._client.embeddings.create(**params)

        sorted_data = sorted(response.data, key=lambda e: e.index)
        return [
            EmbeddingResult(vector=item.embedding, index=item.index)
            for item in sorted_data
        ]

    async def validate_model(self, model: str) -> bool:
        """Probe Azure deployment with a trivial embedding call."""
        try:
            await self._client.embeddings.create(model=model, input=["test"])
            return True
        except Exception as exc:
            logger.warning("Azure deployment validation failed for %s: %s", model, exc)
            return False

    def get_provider_name(self) -> str:
        return "azure"

    async def close(self) -> None:
        await self._client.close()
