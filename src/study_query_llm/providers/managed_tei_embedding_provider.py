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

Instruct-model prompt injection:
    Some models (Qwen3-Embedding, GTE-Qwen2-instruct, multilingual-E5-instruct)
    produce significantly better embeddings when a task-specific prompt prefix is
    prepended to the input.  TEI handles this server-side via the ``prompt_name``
    field in the request body — it looks up the named template from the model's
    ``sentence_transformers_config.json`` and prepends it before encoding.

    ``ManagedTEIEmbeddingProvider`` auto-detects known instruct models from a
    registry and injects ``prompt_name`` automatically.  Pass ``prompt_name=None``
    to suppress for a known model, or pass a custom string to override.
"""

from typing import Dict, List, Optional

from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider
from .base_embedding import EmbeddingResult
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Instruct-model registry
# ---------------------------------------------------------------------------
# Maps HuggingFace model IDs to the TEI ``prompt_name`` that should be used
# when calling /v1/embeddings.  TEI resolves the name to the actual prompt
# text from the model's sentence_transformers_config.json and prepends it
# server-side before encoding — no changes needed in the calling code.
#
# "query" is the standard retrieval/clustering prompt for all listed models.
# For symmetric use cases (same prompt on both sides) "query" remains correct.
#
# Add new instruct models here as the MTEB leaderboard evolves.
_INSTRUCT_MODEL_PROMPT_NAMES: Dict[str, str] = {
    # Qwen3 Embedding series (MTEB #2, #3, #4)
    "Qwen/Qwen3-Embedding-0.6B": "query",
    "Qwen/Qwen3-Embedding-4B": "query",
    "Qwen/Qwen3-Embedding-8B": "query",
    # Alibaba GTE Qwen2 instruct (MTEB #6, #15)
    "Alibaba-NLP/gte-Qwen2-7B-instruct": "query",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": "query",
    # E5 instruct variants (MTEB #7, #18)
    "intfloat/multilingual-e5-large-instruct": "query",
    "intfloat/e5-mistral-7b-instruct": "query",
    "intfloat/e5-large-instruct": "query",
}


def get_prompt_name_for_model(model_id: str) -> Optional[str]:
    """Return the TEI ``prompt_name`` for *model_id*, or ``None`` if not an instruct model."""
    return _INSTRUCT_MODEL_PROMPT_NAMES.get(model_id)


class ManagedTEIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """
    Embedding provider backed by any managed HuggingFace TEI instance.

    Extends ``OpenAICompatibleEmbeddingProvider`` with two extra behaviours:

    1. **Idle-timer ping**: ``create_embeddings`` calls ``manager.ping()``
       before every request so the container is not stopped mid-sweep.

    2. **Instruct-model prompt injection**: models in ``_INSTRUCT_MODEL_PROMPT_NAMES``
       (Qwen3-Embedding, GTE-Qwen2-instruct, multilingual-E5-instruct, …) receive a
       ``prompt_name`` field in every request body so TEI prepends the correct
       task prefix server-side — no changes needed in sweep scripts.

    Compatible with both ``ACITEIManager`` (Azure) and ``LocalDockerTEIManager``
    (local Docker).  The ``provider_label`` (and therefore ``get_provider_name()``)
    is taken directly from the manager, so database records correctly reflect the
    backend and model used.
    """

    def __init__(self, manager, prompt_name: Optional[str] = "auto") -> None:
        """
        Args:
            manager: A started manager instance (``ACITEIManager`` or
                ``LocalDockerTEIManager``).  ``manager.endpoint_url`` must
                already be set — call ``manager.start()`` / ``manager.create()``
                or use the context manager before constructing this provider.
            prompt_name: Controls the TEI ``prompt_name`` field injected into
                every request body.

                - ``"auto"`` *(default)* — look up ``manager.model_id`` in
                  ``_INSTRUCT_MODEL_PROMPT_NAMES``; inject if found, omit if not.
                - ``None`` — never inject a prompt (disables for instruct models;
                  useful for symmetric non-retrieval tasks where you don't want
                  the default instruction prefix).
                - Any other string — use that exact value for every request
                  (for custom or future models not yet in the registry).
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

        # Resolve prompt_name and cache as extra_body for the parent class
        if prompt_name == "auto":
            resolved = get_prompt_name_for_model(manager.model_id)
        elif prompt_name is None:
            resolved = None
        else:
            resolved = prompt_name

        self._extra_body = {"prompt_name": resolved} if resolved is not None else None

        if resolved is not None:
            logger.info(
                "Instruct model detected (%s) — will inject prompt_name=%r into every request.",
                manager.model_id,
                resolved,
            )

        logger.info(
            "Initialized ManagedTEIEmbeddingProvider "
            "(label=%s, endpoint=%s, model=%s, prompt_name=%r)",
            manager.provider_label,
            manager.endpoint_url,
            manager.model_id,
            resolved,
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
