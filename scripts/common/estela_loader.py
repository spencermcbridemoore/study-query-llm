"""Backward-compat shim — canonical module is study_query_llm.utils.estela_loader."""
from study_query_llm.utils.estela_loader import load_estela_dict  # noqa: F401

__all__ = ["load_estela_dict"]
