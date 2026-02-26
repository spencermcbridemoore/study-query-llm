"""
Provider Factory - Creates LLM provider instances by name.

This factory enables dynamic provider creation and switching without
needing to import provider classes directly.
"""

from typing import Optional
from .base import BaseLLMProvider, DeploymentInfo
from .base_embedding import BaseEmbeddingProvider
from .azure_provider import AzureOpenAIProvider
from .azure_embedding_provider import AzureEmbeddingProvider
from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider
from .openai_compatible_chat_provider import OpenAICompatibleChatProvider
from ..config import ProviderConfig, Config


class ProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    This factory provides a unified way to create providers by name,
    either from configuration or with explicit parameters.
    
    Usage:
        # From config
        factory = ProviderFactory()
        provider = factory.create_from_config("azure")
        
        # With explicit config
        config = ProviderConfig(name="azure", api_key="...", ...)
        provider = factory.create("azure", config)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the factory.
        
        Args:
            config: Optional Config instance. If not provided, creates a new one.
        """
        self.config = config or Config()

    def create(
        self,
        provider_name: str,
        provider_config: ProviderConfig
    ) -> BaseLLMProvider:
        """
        Create a provider instance from a ProviderConfig.
        
        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')
            provider_config: ProviderConfig instance with credentials and settings
        
        Returns:
            BaseLLMProvider instance
        
        Raises:
            ValueError: If provider_name is unknown
        """
        provider_name = provider_name.lower()

        if provider_name == "azure":
            return AzureOpenAIProvider(provider_config)
        # TODO: Add other providers as they are implemented
        # elif provider_name == "openai":
        #     return OpenAIProvider(provider_config)
        # elif provider_name == "hyperbolic":
        #     return HyperbolicProvider(provider_config)
        else:
            available = self.get_available_providers()
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {', '.join(available)}"
            )

    def create_from_config(self, provider_name: str) -> BaseLLMProvider:
        """
        Create a provider instance from application configuration.
        
        This method loads the provider configuration from the Config instance
        and creates the appropriate provider.
        
        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')
        
        Returns:
            BaseLLMProvider instance
        
        Raises:
            ValueError: If provider_name is unknown or not configured
        """
        provider_config = self.config.get_provider_config(provider_name)
        return self.create(provider_name, provider_config)

    @staticmethod
    def get_available_providers() -> list[str]:
        """
        Return list of supported provider names.
        
        Returns:
            List of provider names that can be created by this factory
        """
        return ["azure", "local_llm", "ollama"]

    def get_configured_providers(self) -> list[str]:
        """
        Return list of providers that have credentials configured.
        
        This checks which providers have API keys set in the environment.
        
        Returns:
            List of provider names that are configured and ready to use
        """
        return self.config.get_available_providers()
    
    # ------------------------------------------------------------------
    # Embedding provider creation
    # ------------------------------------------------------------------

    def create_embedding_provider(
        self, provider_name: str
    ) -> BaseEmbeddingProvider:
        """Create an embedding provider instance from application config.

        Args:
            provider_name: ``'azure'``, ``'openai'``, ``'huggingface'``,
                ``'local'``, ``'ollama'``, or any OpenAI-compatible label.

        Returns:
            A ``BaseEmbeddingProvider`` ready to use.

        Raises:
            ValueError: If the provider is not configured.
        """
        provider_name = provider_name.lower()
        provider_config = self.config.get_provider_config(
            provider_name if provider_name != "ollama" else "local"
        )

        if provider_name == "azure":
            return AzureEmbeddingProvider(provider_config)

        base_url = provider_config.endpoint or ""
        if provider_name == "ollama" and not base_url:
            base_url = "http://localhost:11434/v1"

        return OpenAICompatibleEmbeddingProvider(
            base_url=base_url,
            api_key=provider_config.api_key,
            provider_label=provider_name,
        )

    @staticmethod
    def get_available_embedding_providers() -> list[str]:
        """Return list of supported embedding provider names."""
        return ["azure", "openai", "huggingface", "local", "ollama"]

    def create_chat_provider(
        self, provider_name: str, model: str
    ) -> BaseLLMProvider:
        """Create a chat completion provider, with model specified at call-time.

        For Azure, the ``model`` argument is ignored -- the deployment name
        comes from config/env as usual.  For all other providers (``local_llm``,
        ``ollama``), ``model`` is forwarded verbatim as the request-body
        ``model`` field so one provider instance per model can be created
        without touching env vars.

        Args:
            provider_name: ``'azure'``, ``'local_llm'``, or ``'ollama'``.
            model: Model identifier string (e.g. ``"qwen2.5:32b"``).
                   Ignored for Azure.

        Returns:
            A ``BaseLLMProvider`` ready to use.
        """
        # Normalise 'ollama' alias to the 'local_llm' config key
        config_key = "local_llm" if provider_name == "ollama" else provider_name
        provider_config = self.config.get_provider_config(config_key)

        if provider_name == "azure":
            return AzureOpenAIProvider(provider_config)

        base_url = provider_config.endpoint or "http://localhost:11434/v1"
        return OpenAICompatibleChatProvider(
            base_url=base_url,
            model=model,
            api_key=provider_config.api_key,
            provider_label=provider_name,
        )

    @staticmethod
    def get_available_chat_providers() -> list[str]:
        """Return list of supported chat provider names."""
        return ["azure", "local_llm", "ollama"]

    async def list_provider_deployments(
        self,
        provider_name: str,
        modality: Optional[str] = None,
    ) -> list[DeploymentInfo]:
        """
        List available deployments/models for a provider with optional modality filtering.

        This is provider-specific functionality. Currently only Azure OpenAI
        supports listing deployments. Other providers may return empty list
        or raise NotImplementedError.

        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')
            modality: Optional capability filter. Accepted values:
                - ``"chat"`` -- only deployments that support chat completions
                - ``"embedding"`` -- only deployments that support embeddings
                - ``None`` (default) -- return all deployments unfiltered

        Returns:
            List of DeploymentInfo objects for the provider, optionally filtered
            by modality. Deployments with no capability data are included when
            no modality filter is specified.

        Raises:
            ValueError: If provider is unknown
            NotImplementedError: If provider doesn't support listing deployments
            Exception: If unable to query deployments
        """
        provider_name = provider_name.lower()

        if provider_name == "azure":
            provider_config = self.config.get_provider_config("azure")
            deployments = await AzureOpenAIProvider.list_deployments(provider_config)
        elif provider_name in ["openai", "hyperbolic"]:
            raise NotImplementedError(
                f"Provider '{provider_name}' does not support listing deployments. "
                f"Use model names directly in configuration."
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Known chat-model ID prefixes (Azure catalog uses model-version IDs like "gpt-4o-2024-05-13")
        _CHAT_PREFIXES = ("gpt-4", "gpt-35", "gpt-3.5", "o1", "o3")
        _EMBED_PREFIXES = ("text-embedding",)

        if modality == "chat":
            # Prefer deployments that explicitly report chat support.
            # If the API returns no capability data (capabilities={}), fall back
            # to name-based heuristics so the dropdown is never empty.
            explicit = [d for d in deployments if d.supports_chat]
            if explicit:
                deployments = explicit
            else:
                deployments = [
                    d for d in deployments
                    if any(d.id.lower().startswith(p) for p in _CHAT_PREFIXES)
                ]
        elif modality == "embedding":
            explicit = [d for d in deployments if d.supports_embeddings]
            if explicit:
                deployments = explicit
            else:
                deployments = [
                    d for d in deployments
                    if any(d.id.lower().startswith(p) for p in _EMBED_PREFIXES)
                ]

        return deployments

