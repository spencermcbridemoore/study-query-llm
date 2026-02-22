"""
Tests for Phase 1.1 - Base Provider Interface.

Tests the abstract base class and ProviderResponse dataclass.
"""

import pytest
from study_query_llm.providers.base import BaseLLMProvider, DeploymentInfo, ProviderResponse


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing the base interface."""

    def __init__(self, name: str = "mock"):
        self.name = name

    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Simulate an LLM response."""
        import asyncio
        await asyncio.sleep(0.05)

        response_text = f"Mock response to: '{prompt[:50]}...'"

        return ProviderResponse(
            text=response_text,
            provider=self.name,
            tokens=len(prompt.split()) + len(response_text.split()),
            latency_ms=50.0,
            metadata={
                "model": "mock-model-v1",
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            },
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return self.name


@pytest.mark.asyncio
async def test_basic_completion():
    """Test basic completion functionality."""
    provider = MockProvider("test-provider")
    
    response = await provider.complete("What is the capital of France?")
    
    assert response.provider == "test-provider"
    assert response.text is not None
    assert response.tokens > 0
    assert response.latency_ms == 50.0
    assert "metadata" in response.__dict__


@pytest.mark.asyncio
async def test_completion_with_parameters():
    """Test completion with custom parameters."""
    provider = MockProvider()
    
    response = await provider.complete(
        "Write a haiku about coding",
        temperature=0.9,
        max_tokens=50,
    )
    
    assert response.metadata["temperature"] == 0.9
    assert response.metadata["max_tokens"] == 50


@pytest.mark.asyncio
async def test_provider_response_repr():
    """Test ProviderResponse string representation."""
    response = ProviderResponse(
        text="Test response",
        provider="test",
        tokens=100,
        latency_ms=250.5,
    )
    
    repr_str = repr(response)
    assert "ProviderResponse" in repr_str
    assert "test" in repr_str
    assert "100 tokens" in repr_str
    assert "250.50ms" in repr_str


def test_abstract_class_enforcement():
    """Test that abstract methods must be implemented."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseLLMProvider()  # Should fail - can't instantiate abstract class


@pytest.mark.asyncio
async def test_provider_response_optional_fields():
    """Test that ProviderResponse optional fields work correctly."""
    # Minimal response
    minimal = ProviderResponse(
        text="Test",
        provider="test"
    )
    assert minimal.tokens is None
    assert minimal.latency_ms is None
    assert minimal.metadata == {}
    assert minimal.raw_response is None
    
    # Full response
    full = ProviderResponse(
        text="Test",
        provider="test",
        tokens=100,
        latency_ms=50.0,
        metadata={"key": "value"},
        raw_response={"raw": "data"}
    )
    assert full.tokens == 100
    assert full.latency_ms == 50.0
    assert full.metadata == {"key": "value"}
    assert full.raw_response == {"raw": "data"}


def test_deployment_info_defaults():
    """DeploymentInfo with no capabilities should have all supports_* as False."""
    info = DeploymentInfo(id="gpt-4o", provider="azure")
    assert info.id == "gpt-4o"
    assert info.provider == "azure"
    assert info.capabilities == {}
    assert info.lifecycle_status is None
    assert info.created_at is None
    assert info.supports_chat is False
    assert info.supports_embeddings is False
    assert info.supports_completion is False


def test_deployment_info_chat_model():
    """Chat completion model should have supports_chat True."""
    info = DeploymentInfo(
        id="gpt-4o",
        provider="azure",
        capabilities={"chat_completion": True, "completion": True, "embeddings": False},
    )
    assert info.supports_chat is True
    assert info.supports_completion is True
    assert info.supports_embeddings is False


def test_deployment_info_embedding_model():
    """Embedding model should have supports_embeddings True and supports_chat False."""
    info = DeploymentInfo(
        id="text-embedding-ada-002",
        provider="azure",
        capabilities={"chat_completion": False, "embeddings": True, "completion": False},
    )
    assert info.supports_chat is False
    assert info.supports_embeddings is True
    assert info.supports_completion is False


def test_deployment_info_lifecycle_and_created_at():
    """Optional metadata fields should round-trip correctly."""
    info = DeploymentInfo(
        id="gpt-4o",
        provider="azure",
        capabilities={"chat_completion": True},
        lifecycle_status="generally-available",
        created_at=1646126127,
    )
    assert info.lifecycle_status == "generally-available"
    assert info.created_at == 1646126127


@pytest.mark.asyncio
async def test_provider_name_consistency():
    """Test that get_provider_name() returns consistent values."""
    provider = MockProvider("consistent-name")
    
    name1 = provider.get_provider_name()
    name2 = provider.get_provider_name()
    
    assert name1 == name2 == "consistent-name"
    
    # Also verify it matches the response provider field
    response = await provider.complete("test")
    assert response.provider == "consistent-name"

