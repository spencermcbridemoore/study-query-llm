"""
Configuration management for Study Query LLM.

Loads configuration from environment variables (typically from a .env file).
Uses python-dotenv to load .env automatically.

Usage:
    from study_query_llm.config import config

    # Access provider configs
    azure_config = config.get_provider_config("azure")

    # Access database config
    db_url = config.database.connection_string
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# Try to load .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env in project root (parent of src/)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed - will use system environment variables
    pass


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    name: str
    api_key: str
    endpoint: Optional[str] = None
    model: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None

    def __post_init__(self):
        """Validate that required fields are present."""
        if not self.api_key:
            raise ValueError(
                f"API key not set for {self.name} provider. "
                f"Please set the appropriate environment variable."
            )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str

    def __post_init__(self):
        """Provide default if not set."""
        if not self.connection_string:
            self.connection_string = "sqlite:///study_query_llm.db"


class Config:
    """
    Application configuration loaded from environment variables.

    Environment variables can be set:
    1. In a .env file in the project root
    2. In the system environment
    3. In a container/deployment environment
    """

    def __init__(self):
        """Load configuration from environment."""
        self.database = DatabaseConfig(
            connection_string=os.getenv("DATABASE_URL", "sqlite:///study_query_llm.db")
        )

        # Provider configurations (lazy-loaded to avoid requiring all keys)
        self._provider_configs = {}

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')

        Returns:
            ProviderConfig with credentials and settings

        Raises:
            ValueError: If required environment variables are missing
        """
        provider_name = provider_name.lower()

        # Return cached config if already loaded
        if provider_name in self._provider_configs:
            return self._provider_configs[provider_name]

        # Load provider-specific config
        if provider_name == "azure":
            config = ProviderConfig(
                name="azure",
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
        elif provider_name == "openai":
            config = ProviderConfig(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
            )
        elif provider_name == "hyperbolic":
            config = ProviderConfig(
                name="hyperbolic",
                api_key=os.getenv("HYPERBOLIC_API_KEY", ""),
                endpoint=os.getenv("HYPERBOLIC_ENDPOINT", "https://api.hyperbolic.xyz"),
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Cache for future use
        self._provider_configs[provider_name] = config
        return config

    def get_available_providers(self) -> list[str]:
        """
        Get list of providers that have API keys configured.

        Returns:
            List of provider names with credentials set
        """
        available = []
        for provider in ["azure", "openai", "hyperbolic"]:
            try:
                self.get_provider_config(provider)
                available.append(provider)
            except ValueError:
                # API key not set - skip this provider
                pass
        return available


# Global config instance
config = Config()


def require_provider(provider_name: str) -> ProviderConfig:
    """
    Get provider config, raising helpful error if not configured.

    Args:
        provider_name: Name of provider to load

    Returns:
        ProviderConfig

    Raises:
        ValueError: With instructions on how to configure the provider
    """
    try:
        return config.get_provider_config(provider_name)
    except ValueError as e:
        raise ValueError(
            f"\n{'='*60}\n"
            f"Provider '{provider_name}' is not configured.\n\n"
            f"To use {provider_name}, set these environment variables:\n"
            f"{_get_provider_instructions(provider_name)}\n"
            f"You can set these in a .env file in the project root.\n"
            f"See .env.example for a template.\n"
            f"{'='*60}\n"
        ) from e


def _get_provider_instructions(provider_name: str) -> str:
    """Get environment variable instructions for a provider."""
    instructions = {
        "azure": """
  AZURE_OPENAI_API_KEY=your-api-key
  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT=gpt-4
  AZURE_OPENAI_API_VERSION=2024-02-15-preview
        """,
        "openai": """
  OPENAI_API_KEY=your-api-key
  OPENAI_MODEL=gpt-4
        """,
        "hyperbolic": """
  HYPERBOLIC_API_KEY=your-api-key
  HYPERBOLIC_ENDPOINT=https://api.hyperbolic.xyz
        """,
    }
    return instructions.get(provider_name, "  (Unknown provider)")
