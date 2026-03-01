"""Backward-compat shim — sweep_utils functions have moved to src modules.

Canonical locations:
- study_query_llm.experiments.sweep_io (serialize, save)
- study_query_llm.experiments.ingestion (ingest_result_to_db)
- study_query_llm.services.paraphraser_factory (create_paraphraser_for_llm)
- study_query_llm.providers.managers.ollama (ollama_vram_scope)
"""
from study_query_llm.experiments.sweep_io import (  # noqa: F401
    get_output_dir,
    serialize_sweep_result,
    save_single_sweep_result,
    save_batch_sweep_results,
)
from study_query_llm.experiments.ingestion import ingest_result_to_db  # noqa: F401
from study_query_llm.services.paraphraser_factory import create_paraphraser_for_llm  # noqa: F401
from study_query_llm.providers.managers.ollama import ollama_vram_scope  # noqa: F401

OUTPUT_DIR = get_output_dir()

__all__ = [
    "OUTPUT_DIR",
    "ollama_vram_scope",
    "create_paraphraser_for_llm",
    "serialize_sweep_result",
    "save_single_sweep_result",
    "save_batch_sweep_results",
    "ingest_result_to_db",
]
