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
    Fixture factory for a provider that fails a specified number of times.
    
    Used for testing retry logic. Returns a class that can be instantiated
    with fail_count and error_type parameters.
    
    Usage:
        provider = failing_provider(fail_count=2, error_type="timeout")
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
def permanently_failing_provider():
    """
    Fixture for a provider that always fails with a permanent error.
    
    Used to test that permanent errors (like 401) are not retried.
    """
    class PermanentlyFailingProvider(BaseLLMProvider):
        def __init__(self):
            self.attempts = 0

        async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
            self.attempts += 1
            # 401 Unauthorized is a permanent error - should not retry
            raise Exception("401 Unauthorized: Invalid API key")

        def get_provider_name(self) -> str:
            return "permanently_failing_provider"
    
    return PermanentlyFailingProvider()


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


@pytest.fixture
def counting_provider():
    """
    Fixture for a provider that includes a counter in each response.
    
    Useful for testing repeated inference - you can verify each call
    was made separately by checking the call_count.
    """
    class CountingProvider(BaseLLMProvider):
        def __init__(self):
            self.call_count = 0

        async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
            """Return a response with a counter to verify each call is separate."""
            self.call_count += 1
            temperature = kwargs.get('temperature', 0.7)

            return ProviderResponse(
                text=f"Response #{self.call_count} to: {prompt} (temp={temperature})",
                provider="counting_provider",
                tokens=10,
                latency_ms=50.0,
            )

        def get_provider_name(self) -> str:
            return "counting_provider"
    
    return CountingProvider()


@pytest.fixture
def variable_provider():
    """
    Fixture for a provider that returns different responses.
    
    Useful for testing response variability in repeated inference.
    """
    class VariableProvider(BaseLLMProvider):
        def __init__(self):
            self.call_count = 0
            self.responses = [
                "The sky is blue",
                "The ocean is vast",
                "Mountains are tall",
                "Rivers flow freely",
                "Stars shine bright"
            ]

        async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
            """Return varied responses to simulate temperature-based variability."""
            response_text = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1

            return ProviderResponse(
                text=response_text,
                provider="variable_provider",
                tokens=5,
                latency_ms=50.0,
            )

        def get_provider_name(self) -> str:
            return "variable_provider"
    
    return VariableProvider()

