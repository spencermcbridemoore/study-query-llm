"""
Provider Factory - Creates LLM provider instances by name.

This factory enables dynamic provider creation and switching without
needing to import provider classes directly.
"""

from typing import Optional
from .base import BaseLLMProvider
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

