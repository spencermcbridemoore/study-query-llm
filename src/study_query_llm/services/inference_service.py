"""
Inference Service - Core business logic for LLM inference operations.

This service wraps LLM providers and adds business logic like retry handling,
preprocessing, and optional database logging.

Usage:
    from study_query_llm.providers import AzureOpenAIProvider
    from study_query_llm.services import InferenceService
    from study_query_llm.config import config

    # Create provider
    azure_config = config.get_provider_config("azure")
    provider = AzureOpenAIProvider(azure_config)

    # Create service
    service = InferenceService(provider)

    # Run inference
    result = await service.run_inference("What is the capital of France?")
    print(result['response'])
"""

from typing import Optional, Any
from ..providers.base import BaseLLMProvider, ProviderResponse


class InferenceService:
    """
    Core service for running LLM inferences with business logic.

    This service provides a higher-level interface for LLM operations,
    adding functionality beyond basic provider API calls:
    - Standardized response format
    - Optional database integration (to be added in Phase 3)
    - Foundation for retry logic (Phase 2.2)
    - Foundation for preprocessing (Phase 2.3)

    The service is provider-agnostic - it works with any BaseLLMProvider.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        repository=None,  # Will be InferenceRepository in Phase 3
    ):
        """
        Initialize the inference service.

        Args:
            provider: Any LLM provider implementing BaseLLMProvider
            repository: Optional database repository for logging (Phase 3)
        """
        self.provider = provider
        self.repository = repository

    async def run_inference(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Run a single inference through the LLM provider.

        This is the main method for executing LLM completions. It wraps
        the provider's complete() method and returns a standardized dict.

        Args:
            prompt: The user's prompt/question
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate (None = provider default)
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with:
                - response: The LLM's text response
                - metadata: Dict with provider, tokens, latency, etc.
                - provider_response: Full ProviderResponse object

        Example:
            >>> service = InferenceService(azure_provider)
            >>> result = await service.run_inference(
            ...     "What is 2+2?",
            ...     temperature=0.0,
            ...     max_tokens=10
            ... )
            >>> print(result['response'])
            "4"
            >>> print(result['metadata']['tokens'])
            15
        """
        # Call the provider (retry logic will be added in Phase 2.2)
        provider_response: ProviderResponse = await self.provider.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Build standardized response
        result = {
            'response': provider_response.text,
            'metadata': {
                'provider': provider_response.provider,
                'tokens': provider_response.tokens,
                'latency_ms': provider_response.latency_ms,
                'temperature': temperature,
            },
            'provider_response': provider_response,  # Include full object for advanced use
        }

        if max_tokens is not None:
            result['metadata']['max_tokens'] = max_tokens

        # Database logging will be added here in Phase 3.5
        # if self.repository:
        #     self.repository.insert_inference_run(...)

        return result

    async def run_batch_inference(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[dict]:
        """
        Run multiple inferences concurrently.

        This is a convenience method for running multiple prompts through
        the same provider with the same parameters.

        Args:
            prompts: List of prompts to process
            **kwargs: Parameters to apply to all prompts

        Returns:
            List of result dictionaries (same format as run_inference)

        Example:
            >>> service = InferenceService(provider)
            >>> prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
            >>> results = await service.run_batch_inference(prompts, temperature=0.0)
            >>> for result in results:
            ...     print(result['response'])
        """
        import asyncio

        # Run all prompts concurrently
        tasks = [
            self.run_inference(prompt, **kwargs)
            for prompt in prompts
        ]

        return await asyncio.gather(*tasks)

    def get_provider_name(self) -> str:
        """
        Get the name of the underlying provider.

        Returns:
            Provider name string (e.g., 'azure_openai_gpt-4o')
        """
        return self.provider.get_provider_name()

    async def close(self):
        """
        Close the underlying provider connection.

        Call this when done with the service to clean up resources.
        """
        if hasattr(self.provider, 'close'):
            await self.provider.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes provider."""
        await self.close()
