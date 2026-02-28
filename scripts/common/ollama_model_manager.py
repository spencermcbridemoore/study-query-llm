"""Backward-compat shim — canonical module is study_query_llm.providers.managers.ollama."""
from study_query_llm.providers.managers.ollama import OllamaModelManager  # noqa: F401

__all__ = ["OllamaModelManager"]
