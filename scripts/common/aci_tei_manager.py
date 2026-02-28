"""Backward-compat shim — canonical module is study_query_llm.providers.managers.aci_tei."""
from study_query_llm.providers.managers.aci_tei import (  # noqa: F401
    ACITEIManager,
    manager_from_env,
    _TEI_CPU_IMAGE,
    _TEI_GPU_IMAGE,
)

__all__ = ["ACITEIManager", "manager_from_env"]
