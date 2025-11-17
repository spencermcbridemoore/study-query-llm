"""
Base classes for LLM provider abstraction.

This module defines the interface that all LLM providers must implement,
ensuring consistent behavior across different provider implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProviderResponse:
    """
    Standardized response from any LLM provider.

    This class normalizes responses from different LLM APIs into a
    consistent format that can be used by the rest of the application.

    Attributes:
        text: The generated text response from the LLM
        provider: Name of the provider that generated this response
        tokens: Total tokens used (prompt + completion), if available
        latency_ms: Response latency in milliseconds, if measured
        metadata: Provider-specific metadata (model name, token breakdown, etc.)
        raw_response: The original, unprocessed response object from the provider
    """
    text: str
    provider: str
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        tokens_str = f"{self.tokens} tokens" if self.tokens else "unknown tokens"
        latency_str = f"{self.latency_ms:.2f}ms" if self.latency_ms else "unknown latency"
        return (
            f"ProviderResponse(provider={self.provider}, "
            f"{tokens_str}, {latency_str})"
        )


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All provider implementations (Azure, OpenAI, Hyperbolic, etc.) must
    inherit from this class and implement its abstract methods.

    This ensures a consistent interface regardless of which LLM provider
    is being used, making it easy to switch providers or compare results
    across different providers.
    """

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Send a completion request to the LLM provider.

        This is the core method that all providers must implement. It sends
        a prompt to the LLM and returns the response in a standardized format.

        Args:
            prompt: The input prompt/question for the LLM
            **kwargs: Provider-specific parameters such as:
                - temperature: Controls randomness (0.0 to 1.0)
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - model: Specific model to use (if provider supports multiple)
                - messages: For chat-based APIs (list of message dicts)
                - Any other provider-specific parameters

        Returns:
            ProviderResponse object with standardized response data

        Raises:
            Exception: Provider-specific exceptions (rate limits, auth errors, etc.)
                       Note: Retry logic is handled by the service layer, not here

        Example:
            >>> provider = SomeProvider(api_key="...")
            >>> response = await provider.complete(
            ...     "What is the capital of France?",
            ...     temperature=0.7,
            ...     max_tokens=100
            ... )
            >>> print(response.text)
            "The capital of France is Paris."
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the name of this provider.

        Returns:
            String identifier for the provider (e.g., 'azure', 'openai', 'hyperbolic')

        Example:
            >>> provider = AzureProvider(...)
            >>> provider.get_provider_name()
            'azure'
        """
        pass
