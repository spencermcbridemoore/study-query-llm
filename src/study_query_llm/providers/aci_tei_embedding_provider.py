"""
ACI TEI Embedding Provider.

Wraps ``OpenAICompatibleEmbeddingProvider`` so that every embedding call also
pings the ``ACITEIManager`` idle timer, ensuring the underlying Azure Container
Instance is not deleted while the provider is actively being used.

The provider does NOT own the ACI lifecycle -- creation and deletion are the
caller's responsibility via the ``ACITEIManager`` context manager:

    manager = ACITEIManager(...)
    with manager:                       # creates ACI, deletes on exit
        provider = ACITEIEmbeddingProvider(manager)
        async with provider:            # just closes the HTTP client on exit
            service = EmbeddingService(repository=repo, provider=provider)
            # ... run embeddings ...
"""

from typing import List, Optional

from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider
from .base_embedding import EmbeddingResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ACITEIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """
    Embedding provider backed by an ACI-hosted HuggingFace TEI instance.

    Extends ``OpenAICompatibleEmbeddingProvider`` with one extra behaviour:
    each call to ``create_embeddings`` resets the ``ACITEIManager`` idle timer
    so the container is not deleted while a sweep is actively running.

    Construction requires a fully started ``ACITEIManager`` (i.e. after
    ``manager.create()`` has been called, so that ``manager.endpoint_url`` is
    set).
    """

    def __init__(self, aci_manager) -> None:
        """
        Args:
            aci_manager: A started ``ACITEIManager`` instance.  Its
                ``endpoint_url`` must already be set (call ``manager.create()``
                or use the context manager before constructing this provider).
        """
        if aci_manager.endpoint_url is None:
            raise ValueError(
                "ACITEIManager.endpoint_url is None. "
                "Call manager.create() (or use 'with manager:') before "
                "constructing ACITEIEmbeddingProvider."
            )

        super().__init__(
            base_url=aci_manager.endpoint_url,
            api_key="not-needed",
            provider_label="aci_tei",
        )
        self._aci_manager = aci_manager
        logger.info(
            "Initialized ACITEIEmbeddingProvider (endpoint=%s, model=%s)",
            aci_manager.endpoint_url,
            aci_manager.model_id,
        )

    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings and reset the ACI idle timer.

        The idle timer reset happens *before* the API call so that even a
        long-running batch call keeps the container alive.
        """
        self._aci_manager.ping()
        return await super().create_embeddings(texts, model, dimensions)

    async def close(self) -> None:
        """
        Close the underlying HTTP client.

        Does NOT delete the ACI container -- that is the ``ACITEIManager``
        context manager's responsibility.
        """
        await super().close()
        logger.debug(
            "ACITEIEmbeddingProvider closed HTTP client "
            "(ACI container lifecycle managed by ACITEIManager)."
        )
