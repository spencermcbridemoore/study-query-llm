"""Backward-compat shim — canonical module is study_query_llm.providers.managers.local_docker_tei."""
from study_query_llm.providers.managers.local_docker_tei import (  # noqa: F401
    LocalDockerTEIManager,
    _TEI_GPU_IMAGE,
    _TEI_CPU_IMAGE,
    _DEFAULT_HF_CACHE,
)

__all__ = ["LocalDockerTEIManager"]
