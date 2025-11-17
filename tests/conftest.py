"""
Pytest configuration and shared fixtures.

This file is automatically discovered by pytest and provides fixtures
available to all test modules.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.config import Config


# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    
    This ensures all async tests use the same event loop.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_provider():
    """
    Fixture for a mock LLM provider.
    
    Returns a simple mock provider that can be used for testing
    without requiring actual API calls.
    """
    class MockProvider(BaseLLMProvider):
        def __init__(self, name: str = "mock"):
            self.name = name
            self.call_count = 0

        async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
            """Simulate an LLM response."""
            self.call_count += 1
            await asyncio.sleep(0.01)  # Simulate some processing time
            
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
            return self.name
    
    return MockProvider()


@pytest.fixture
def echo_provider():
    """
    Fixture for a provider that echoes back the prompt.
    
    Useful for testing preprocessing - you can verify what prompt
    was actually sent to the provider.
    """
    class EchoProvider(BaseLLMProvider):
        async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
            """Echo the prompt back so we can verify preprocessing."""
            return ProviderResponse(
                text=f"Echoed: {prompt}",
                provider="echo_provider",
                tokens=len(prompt.split()),
                latency_ms=10.0,
            )

        def get_provider_name(self) -> str:
            return "echo_provider"
    
    return EchoProvider()


@pytest.fixture
def failing_provider():
    """
    Fixture for a provider that fails a specified number of times.
    
    Used for testing retry logic. Configure with fail_count parameter.
    """
    class FailingProvider(BaseLLMProvider):
        def __init__(self, fail_count: int = 2, error_type: str = "timeout"):
            self.fail_count = fail_count
            self.attempts = 0
            self.error_type = error_type

        async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
            """Fail fail_count times, then succeed."""
            self.attempts += 1

            if self.attempts <= self.fail_count:
                # Raise different types of errors based on error_type
                if self.error_type == "timeout":
                    raise TimeoutError(f"Simulated timeout (attempt {self.attempts})")
                elif self.error_type == "rate_limit":
                    raise Exception(f"Rate limit exceeded (attempt {self.attempts})")
                elif self.error_type == "503":
                    raise Exception(f"503 Service Unavailable (attempt {self.attempts})")
                elif self.error_type == "connection":
                    raise ConnectionError(f"Connection failed (attempt {self.attempts})")

            # Success on final attempt
            return ProviderResponse(
                text="Success after retries!",
                provider="failing_provider",
                tokens=10,
                latency_ms=50.0,
            )

        def get_provider_name(self) -> str:
            return "failing_provider"
    
    return FailingProvider


@pytest.fixture
def azure_config():
    """
    Fixture for Azure provider configuration.
    
    Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc. in .env.
    Skips tests if not configured.
    """
    config = Config()
    try:
        return config.get_provider_config("azure")
    except ValueError:
        pytest.skip("Azure credentials not configured. Set AZURE_OPENAI_API_KEY, etc. in .env")


@pytest.fixture
def openai_config():
    """
    Fixture for OpenAI provider configuration.
    
    Requires OPENAI_API_KEY in .env.
    Skips tests if not configured.
    """
    config = Config()
    try:
        return config.get_provider_config("openai")
    except ValueError:
        pytest.skip("OpenAI credentials not configured. Set OPENAI_API_KEY in .env")

