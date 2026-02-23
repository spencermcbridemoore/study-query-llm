"""
Provider Factory - Creates LLM provider instances by name.

This factory enables dynamic provider creation and switching without
needing to import provider classes directly.
"""

from typing import Optional
from .base import BaseLLMProvider, DeploymentInfo
from .azure_provider import AzureOpenAIProvider
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
        return ["azure"]  # TODO: Add "openai", "hyperbolic" as they are implemented

    def get_configured_providers(self) -> list[str]:
        """
        Return list of providers that have credentials configured.
        
        This checks which providers have API keys set in the environment.
        
        Returns:
            List of provider names that are configured and ready to use
        """
        return self.config.get_available_providers()
    
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

