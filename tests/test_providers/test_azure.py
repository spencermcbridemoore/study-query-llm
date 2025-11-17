"""
Tests for Phase 1.2 - Azure OpenAI Provider.

Tests the Azure OpenAI provider implementation with real API calls.
Requires Azure credentials to be configured in .env file.
"""

import pytest
from study_query_llm.providers.azure_provider import AzureOpenAIProvider


@pytest.mark.asyncio
@pytest.mark.requires_api
async def test_azure_provider_initialization(azure_config):
    """Test that Azure provider can be initialized with config."""
    provider = AzureOpenAIProvider(azure_config)
    
    assert provider is not None
    assert provider.get_provider_name().startswith("azure_openai")
    assert provider.deployment_name == azure_config.deployment_name


@pytest.mark.asyncio
@pytest.mark.requires_api
async def test_azure_basic_completion(azure_config):
    """Test basic completion with Azure OpenAI."""
    provider = AzureOpenAIProvider(azure_config)
    
    async with provider:
        response = await provider.complete("What is 2+2? Answer in one word.")
    
    assert response.text is not None
    assert len(response.text) > 0
    assert response.provider.startswith("azure_openai")
    assert response.tokens is not None
    assert response.tokens > 0
    assert response.latency_ms is not None
    assert response.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.requires_api
async def test_azure_response_format(azure_config):
    """Test that Azure response has correct format."""
    provider = AzureOpenAIProvider(azure_config)
    
    async with provider:
        response = await provider.complete("Say hello in 5 words")
    
    # Check required fields
    assert hasattr(response, 'text')
    assert hasattr(response, 'provider')
    assert hasattr(response, 'tokens')
    assert hasattr(response, 'latency_ms')
    assert hasattr(response, 'metadata')
    assert hasattr(response, 'raw_response')
    
    # Check metadata
    assert "model" in response.metadata
    assert "deployment" in response.metadata
    assert "finish_reason" in response.metadata
    assert "prompt_tokens" in response.metadata
    assert "completion_tokens" in response.metadata


@pytest.mark.asyncio
@pytest.mark.requires_api
async def test_azure_with_parameters(azure_config):
    """Test Azure provider with custom parameters."""
    provider = AzureOpenAIProvider(azure_config)
    
    async with provider:
        response = await provider.complete(
            "Say 'hello' in French.",
            temperature=0.5,
            max_tokens=10
        )
    
    assert response.text is not None
    assert response.metadata["temperature"] == 0.5
    assert response.metadata["max_tokens"] == 10


@pytest.mark.asyncio
@pytest.mark.requires_api
async def test_azure_context_manager(azure_config):
    """Test that Azure provider works as async context manager."""
    async with AzureOpenAIProvider(azure_config) as provider:
        response = await provider.complete("Test")
    
    assert response.text is not None
    # Context manager should have closed the connection


def test_azure_missing_endpoint():
    """Test that Azure provider raises error when endpoint is missing."""
    from study_query_llm.config import ProviderConfig
    
    config = ProviderConfig(
        name="azure",
        api_key="test-key",
        # endpoint is missing
        deployment_name="test-deployment",
        api_version="2024-02-15-preview"
    )
    
    with pytest.raises(ValueError, match="endpoint"):
        AzureOpenAIProvider(config)


def test_azure_missing_deployment():
    """Test that Azure provider raises error when deployment is missing."""
    from study_query_llm.config import ProviderConfig
    
    config = ProviderConfig(
        name="azure",
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        # deployment_name is missing
        api_version="2024-02-15-preview"
    )
    
    with pytest.raises(ValueError, match="deployment"):
        AzureOpenAIProvider(config)


def test_azure_missing_api_version():
    """Test that Azure provider raises error when API version is missing."""
    from study_query_llm.config import ProviderConfig
    
    config = ProviderConfig(
        name="azure",
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        deployment_name="test-deployment",
        # api_version is missing
    )
    
    with pytest.raises(ValueError, match="API version"):
        AzureOpenAIProvider(config)

