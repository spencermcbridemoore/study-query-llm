"""Backward-compat shim — canonical module is study_query_llm.providers.managers.protocol."""
from study_query_llm.providers.managers.protocol import ModelManager  # noqa: F401

__all__ = ["ModelManager"]
