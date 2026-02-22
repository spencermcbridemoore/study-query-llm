"""
Tests for Phase 1.5 - Provider Factory.

Tests the ProviderFactory for creating provider instances.
"""

import pytest
from unittest.mock import AsyncMock, patch
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.providers.azure_provider import AzureOpenAIProvider
from study_query_llm.providers.base import DeploymentInfo
from study_query_llm.config import ProviderConfig, Config


def test_factory_get_available_providers():
    """Test that factory returns list of available providers."""
    factory = ProviderFactory()
    providers = factory.get_available_providers()
    
    assert isinstance(providers, list)
    assert "azure" in providers
    # Note: "openai" and "hyperbolic" will be added when implemented


def test_factory_create_azure_provider():
    """Test creating Azure provider via factory."""
    factory = ProviderFactory()
    
    config = ProviderConfig(
        name="azure",
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        deployment_name="gpt-4",
        api_version="2024-02-15-preview"
    )
    
    provider = factory.create("azure", config)
    
    assert isinstance(provider, AzureOpenAIProvider)
    assert provider.get_provider_name().startswith("azure_openai")


def test_factory_create_unknown_provider():
    """Test that factory raises error for unknown provider."""
    factory = ProviderFactory()
    
    config = ProviderConfig(
        name="unknown",
        api_key="test-key"
    )
    
    with pytest.raises(ValueError, match="Unknown provider"):
        factory.create("unknown", config)


def test_factory_create_case_insensitive():
    """Test that provider name is case-insensitive."""
    factory = ProviderFactory()
    
    config = ProviderConfig(
        name="azure",
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        deployment_name="gpt-4",
        api_version="2024-02-15-preview"
    )
    
    # Test various case combinations
    provider1 = factory.create("AZURE", config)
    provider2 = factory.create("Azure", config)
    provider3 = factory.create("azure", config)
    
    assert isinstance(provider1, AzureOpenAIProvider)
    assert isinstance(provider2, AzureOpenAIProvider)
    assert isinstance(provider3, AzureOpenAIProvider)


def test_factory_create_from_config(azure_config):
    """Test creating provider from application config."""
    factory = ProviderFactory()
    
    provider = factory.create_from_config("azure")
    
    assert isinstance(provider, AzureOpenAIProvider)
    assert provider.get_provider_name().startswith("azure_openai")


def test_factory_create_from_config_missing():
    """Test that create_from_config raises error for unconfigured provider."""
    factory = ProviderFactory()
    
    with pytest.raises(ValueError):
        # Assuming OpenAI is not configured
        factory.create_from_config("openai")


def test_factory_get_configured_providers():
    """Test getting list of configured providers."""
    factory = ProviderFactory()
    
    configured = factory.get_configured_providers()
    
    assert isinstance(configured, list)
    # Should return providers that have API keys set
    # This will vary based on environment


def test_factory_with_custom_config():
    """Test factory with custom Config instance."""
    custom_config = Config()
    factory = ProviderFactory(config=custom_config)
    
    # Should work the same as default
    providers = factory.get_available_providers()
    assert "azure" in providers


@pytest.mark.asyncio
async def test_list_provider_deployments_no_filter():
    """list_provider_deployments with no modality returns all deployments."""
    mock_deployments = [
        DeploymentInfo(id="gpt-4o", provider="azure", capabilities={"chat_completion": True, "embeddings": False}),
        DeploymentInfo(id="embed-v4", provider="azure", capabilities={"chat_completion": False, "embeddings": True}),
    ]
    factory = ProviderFactory()
    with patch.object(AzureOpenAIProvider, "list_deployments", new_callable=AsyncMock, return_value=mock_deployments):
        result = await factory.list_provider_deployments("azure")
    assert len(result) == 2
    assert {d.id for d in result} == {"gpt-4o", "embed-v4"}


@pytest.mark.asyncio
async def test_list_provider_deployments_chat_filter():
    """list_provider_deployments with modality='chat' returns only chat models."""
    mock_deployments = [
        DeploymentInfo(id="gpt-4o", provider="azure", capabilities={"chat_completion": True, "embeddings": False}),
        DeploymentInfo(id="embed-v4", provider="azure", capabilities={"chat_completion": False, "embeddings": True}),
    ]
    factory = ProviderFactory()
    with patch.object(AzureOpenAIProvider, "list_deployments", new_callable=AsyncMock, return_value=mock_deployments):
        result = await factory.list_provider_deployments("azure", modality="chat")
    assert len(result) == 1
    assert result[0].id == "gpt-4o"


@pytest.mark.asyncio
async def test_list_provider_deployments_embedding_filter():
    """list_provider_deployments with modality='embedding' returns only embedding models."""
    mock_deployments = [
        DeploymentInfo(id="gpt-4o", provider="azure", capabilities={"chat_completion": True, "embeddings": False}),
        DeploymentInfo(id="embed-v4", provider="azure", capabilities={"chat_completion": False, "embeddings": True}),
    ]
    factory = ProviderFactory()
    with patch.object(AzureOpenAIProvider, "list_deployments", new_callable=AsyncMock, return_value=mock_deployments):
        result = await factory.list_provider_deployments("azure", modality="embedding")
    assert len(result) == 1
    assert result[0].id == "embed-v4"


@pytest.mark.asyncio
async def test_list_provider_deployments_returns_deployment_info():
    """list_provider_deployments returns DeploymentInfo objects, not plain strings."""
    mock_deployments = [
        DeploymentInfo(id="gpt-4o", provider="azure", capabilities={"chat_completion": True}),
    ]
    factory = ProviderFactory()
    with patch.object(AzureOpenAIProvider, "list_deployments", new_callable=AsyncMock, return_value=mock_deployments):
        result = await factory.list_provider_deployments("azure")
    assert all(isinstance(d, DeploymentInfo) for d in result)

