"""
Tests for Phase 1.5 - Provider Factory.

Tests the ProviderFactory for creating provider instances.
"""

import pytest
from unittest.mock import AsyncMock, patch
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.providers.azure_provider import AzureOpenAIProvider
from study_query_llm.providers.openai_compatible_chat_provider import (
    OpenAICompatibleChatProvider,
)
from study_query_llm.providers.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingProvider,
)
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


# ---------------------------------------------------------------------------
# create_chat_provider tests
# ---------------------------------------------------------------------------


def test_create_chat_provider_local_llm():
    """create_chat_provider('local_llm', model) returns OpenAICompatibleChatProvider."""
    with patch(
        "study_query_llm.providers.openai_compatible_chat_provider.AsyncOpenAI"
    ):
        factory = ProviderFactory()
        provider = factory.create_chat_provider("local_llm", "qwen2.5:32b")

        assert isinstance(provider, OpenAICompatibleChatProvider)
        assert provider._model == "qwen2.5:32b"
        assert provider.get_provider_name() == "local_llm"


def test_create_chat_provider_ollama_alias():
    """'ollama' is an alias that maps to the local_llm config."""
    with patch(
        "study_query_llm.providers.openai_compatible_chat_provider.AsyncOpenAI"
    ):
        factory = ProviderFactory()
        provider = factory.create_chat_provider("ollama", "llama3.1:8b")

        assert isinstance(provider, OpenAICompatibleChatProvider)
        assert provider._model == "llama3.1:8b"


def test_create_chat_provider_azure():
    """create_chat_provider('azure', ...) returns AzureOpenAIProvider."""
    factory = ProviderFactory()
    # Need Azure env to be configured; test with explicit config
    config = Config()
    try:
        config.get_provider_config("azure")
    except ValueError:
        pytest.skip("Azure credentials not configured")

    provider = factory.create_chat_provider("azure", "gpt-4o")
    assert isinstance(provider, AzureOpenAIProvider)


def test_get_available_chat_providers():
    """get_available_chat_providers includes azure, local_llm, ollama."""
    providers = ProviderFactory.get_available_chat_providers()
    assert "azure" in providers
    assert "local_llm" in providers
    assert "ollama" in providers


def test_get_available_embedding_providers():
    """Embedding providers include OpenRouter as first-class option."""
    providers = ProviderFactory.get_available_embedding_providers()
    assert "azure" in providers
    assert "openrouter" in providers


def test_create_embedding_provider_openrouter(monkeypatch):
    """create_embedding_provider('openrouter') returns OpenAI-compatible wrapper."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
    with patch(
        "study_query_llm.providers.openai_compatible_embedding_provider.AsyncOpenAI"
    ):
        factory = ProviderFactory()
        provider = factory.create_embedding_provider("openrouter")
    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.get_provider_name() == "openrouter"


@pytest.mark.asyncio
async def test_list_provider_deployments_openrouter_embedding_filter(monkeypatch):
    """OpenRouter embedding listing is discoverable and mapped to capabilities."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
    factory = ProviderFactory()

    async def _fake_fetch(_endpoint, _api_key, *, embeddings_only):
        assert embeddings_only is True
        return [
            {
                "id": "openai/text-embedding-3-small",
                "created": 123,
                "context_length": 8192,
                "architecture": {
                    "modality": "text->embeddings",
                    "input_modalities": ["text"],
                    "output_modalities": ["embeddings"],
                    "tokenizer": "cl100k_base",
                },
                "pricing": {"prompt": "0.00000002", "completion": "0"},
                "supported_parameters": ["max_tokens", "temperature"],
            }
        ]

    with patch.object(factory, "_fetch_openrouter_models_json", side_effect=_fake_fetch):
        result = await factory.list_provider_deployments(
            "openrouter", modality="embedding"
        )

    assert len(result) == 1
    dep = result[0]
    assert dep.id == "openai/text-embedding-3-small"
    assert dep.supports_embeddings is True
    assert dep.supports_chat is False
    assert dep.context_length == 8192
    assert dep.output_modalities == ["embeddings"]
    assert dep.pricing.get("prompt") == "0.00000002"


@pytest.mark.asyncio
async def test_list_provider_deployments_openrouter_chat_filter(monkeypatch):
    """OpenRouter chat listing filters to text-output models."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
    factory = ProviderFactory()

    async def _fake_fetch(_endpoint, _api_key, *, embeddings_only):
        assert embeddings_only is False
        return [
            {
                "id": "openai/gpt-4o-mini",
                "architecture": {"modality": "text+image->text"},
            },
            {
                "id": "openai/text-embedding-3-small",
                "architecture": {"modality": "text->embeddings"},
            },
        ]

    with patch.object(factory, "_fetch_openrouter_models_json", side_effect=_fake_fetch):
        result = await factory.list_provider_deployments("openrouter", modality="chat")

    assert [d.id for d in result] == ["openai/gpt-4o-mini"]
    assert result[0].supports_chat is True


@pytest.mark.asyncio
async def test_list_provider_deployments_openrouter_skips_malformed_rows(monkeypatch):
    """Malformed model rows are ignored instead of raising."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
    factory = ProviderFactory()

    async def _fake_fetch(_endpoint, _api_key, *, embeddings_only):
        return [
            {"name": "missing-id"},
            {
                "id": "valid/model",
                "architecture": {"output_modalities": ["text"]},
            },
        ]

    with patch.object(factory, "_fetch_openrouter_models_json", side_effect=_fake_fetch):
        result = await factory.list_provider_deployments("openrouter")

    assert [d.id for d in result] == ["valid/model"]

