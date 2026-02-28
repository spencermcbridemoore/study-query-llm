"""Backward-compat shim — canonical module is study_query_llm.services.embedding_helpers."""
from study_query_llm.services.embedding_helpers import fetch_embeddings_async  # noqa: F401

__all__ = ["fetch_embeddings_async"]
