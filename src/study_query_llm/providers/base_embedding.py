"""
Base classes for embedding provider abstraction.

Defines the interface that all embedding providers must implement,
ensuring consistent behavior whether embeddings come from Azure OpenAI,
HuggingFace TEI, local models, or any OpenAI-compatible endpoint.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingResult:
    """Provider-neutral embedding result.

    Attributes:
        vector: The embedding vector as a list of floats.
        index: Position of this result in the input batch.
    """

    vector: List[float]
    index: int


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.

    All embedding provider implementations must inherit from this class
    and implement its abstract methods. This keeps ``EmbeddingService``
    decoupled from any specific SDK or transport.
    """

    @abstractmethod
    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for one or more texts in a single API call.

        Args:
            texts: Input texts to embed.
            model: Model or deployment name.
            dimensions: Optional output dimension override
                        (only supported by some models).

        Returns:
            List of ``EmbeddingResult`` objects in the same order as *texts*.

        Raises:
            Exception: Provider-specific errors (rate limits, auth, etc.).
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a short identifier for this provider (e.g. ``'azure'``)."""

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (HTTP clients, etc.)."""

    # ------------------------------------------------------------------
    # Optional hooks with sensible defaults
    # ------------------------------------------------------------------

    async def validate_model(self, model: str) -> bool:
        """Check whether *model* is available on this provider.

        The default implementation returns ``True`` (assume valid).
        Override in providers that support model listing / probing.
        """
        return True

    # Context-manager support
    async def __aenter__(self) -> "BaseEmbeddingProvider":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
