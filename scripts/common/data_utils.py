"""Backward-compat shim — canonical module is study_query_llm.utils.text_utils."""
from study_query_llm.utils.text_utils import (  # noqa: F401
    is_prompt_key,
    flatten_prompt_dict,
    clean_texts,
)

__all__ = ["is_prompt_key", "flatten_prompt_dict", "clean_texts"]
