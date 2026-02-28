"""
Model lifecycle managers for TEI containers, Ollama, and ACI.

Provides the ``ModelManager`` protocol and concrete implementations that
manage GPU/container lifecycle for embedding and LLM inference backends.
"""

from .protocol import ModelManager
from .aci_tei import ACITEIManager, manager_from_env
from .local_docker_tei import LocalDockerTEIManager
from .ollama import OllamaModelManager

__all__ = [
    "ModelManager",
    "ACITEIManager",
    "manager_from_env",
    "LocalDockerTEIManager",
    "OllamaModelManager",
]
