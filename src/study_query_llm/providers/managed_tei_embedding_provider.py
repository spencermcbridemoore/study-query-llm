"""
Managed TEI Embedding Provider.

A single embedding provider that works with any TEI lifecycle manager —
``ACITEIManager`` (Azure Container Instances) or ``LocalDockerTEIManager``
(local GPU Docker container) — via duck typing.

The provider does NOT own the container lifecycle.  Creation and teardown
are the caller's responsibility via the manager's context manager:

    # Local Docker (zero cost, uses your RTX 4090)
    with LocalDockerTEIManager(model_id="BAAI/bge-m3") as manager:
        async with ManagedTEIEmbeddingProvider(manager) as provider:
            service = EmbeddingService(repository=repo, provider=provider)
            # ...

    # Azure (cloud, pay-per-second)
    with ACITEIManager(subscription_id=..., model_id="BAAI/bge-m3") as manager:
        async with ManagedTEIEmbeddingProvider(manager) as provider:
            # ...

Manager duck-type contract (both ACITEIManager and LocalDockerTEIManager satisfy this):
    manager.endpoint_url   -- str, set after start()/create(); None otherwise
    manager.model_id       -- str, HuggingFace model ID
    manager.provider_label -- str, short label e.g. "aci_tei" or "local_docker_tei"
    manager.ping()         -- resets the idle timer
"""

from typing import List, Optional

from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider
from .base_embedding import EmbeddingResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ManagedTEIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """
    Embedding provider backed by any managed HuggingFace TEI instance.

    Extends ``OpenAICompatibleEmbeddingProvider`` with one extra behaviour:
    each call to ``create_embeddings`` resets the manager's idle timer so the
    container is not stopped while a sweep is actively running.

    Compatible with both ``ACITEIManager`` (Azure) and ``LocalDockerTEIManager``
    (local Docker).  The ``provider_label`` (and therefore ``get_provider_name()``)
    is taken directly from the manager, so database records correctly identify
    whether embeddings came from ACI or the local GPU.
    """

    def __init__(self, manager) -> None:
        """
        Args:
            manager: A started manager instance (``ACITEIManager`` or
                ``LocalDockerTEIManager``).  ``manager.endpoint_url`` must
                already be set — call ``manager.start()`` / ``manager.create()``
                or use the context manager before constructing this provider.
        """
        if manager.endpoint_url is None:
            raise ValueError(
                f"{type(manager).__name__}.endpoint_url is None. "
                "Call start()/create() (or use 'with manager:') before "
                "constructing ManagedTEIEmbeddingProvider."
            )

        super().__init__(
            base_url=manager.endpoint_url,
            api_key="not-needed",
            provider_label=manager.provider_label,
        )
        self._manager = manager
        logger.info(
            "Initialized ManagedTEIEmbeddingProvider "
            "(label=%s, endpoint=%s, model=%s)",
            manager.provider_label,
            manager.endpoint_url,
            manager.model_id,
        )

    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings and reset the manager's idle timer.

        The idle timer is reset *before* the API call so that even a
        long-running batch keeps the container alive.
        """
        self._manager.ping()
        return await super().create_embeddings(texts, model, dimensions)

    async def close(self) -> None:
        """
        Close the underlying HTTP client.

        Does NOT stop the container — that is the manager's responsibility
        via its context manager.
        """
        await super().close()
        logger.debug(
            "ManagedTEIEmbeddingProvider closed HTTP client "
            "(container lifecycle managed by %s).",
            type(self._manager).__name__,
        )
