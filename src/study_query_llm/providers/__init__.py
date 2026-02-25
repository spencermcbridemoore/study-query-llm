"""
LLM Provider Abstraction Layer

This package provides a unified interface for interacting with different
LLM providers (Azure OpenAI, OpenAI, Hyperbolic, etc.).

Chat providers implement the ``BaseLLMProvider`` interface; embedding
providers implement ``BaseEmbeddingProvider``.
"""

from .base import BaseLLMProvider, DeploymentInfo, ProviderResponse
from .base_embedding import BaseEmbeddingProvider, EmbeddingResult
from .azure_provider import AzureOpenAIProvider
from .azure_embedding_provider import AzureEmbeddingProvider
from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider
from .aci_tei_embedding_provider import ACITEIEmbeddingProvider
from .factory import ProviderFactory

__all__ = [
    "BaseLLMProvider",
    "DeploymentInfo",
    "ProviderResponse",
    "BaseEmbeddingProvider",
    "EmbeddingResult",
    "AzureOpenAIProvider",
    "AzureEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "ACITEIEmbeddingProvider",
    "ProviderFactory",
]
