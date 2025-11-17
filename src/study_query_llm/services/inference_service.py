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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    RetryError,
)
from ..providers.base import BaseLLMProvider, ProviderResponse
from .preprocessors import PromptPreprocessor


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
        max_retries: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 10.0,
        preprocess: bool = False,
        clean_whitespace: bool = True,
        truncate_prompts: bool = True,
        max_prompt_length: int = 10000,
        remove_pii: bool = False,
        strip_control_chars: bool = False,
    ):
        """
        Initialize the inference service.

        Args:
            provider: Any LLM provider implementing BaseLLMProvider
            repository: Optional database repository for logging (Phase 3)
            max_retries: Maximum number of retry attempts (default: 3)
            initial_wait: Initial wait time in seconds for exponential backoff (default: 1.0)
            max_wait: Maximum wait time in seconds between retries (default: 10.0)
            preprocess: Enable prompt preprocessing (default: False)
            clean_whitespace: Normalize whitespace when preprocessing (default: True)
            truncate_prompts: Truncate long prompts when preprocessing (default: True)
            max_prompt_length: Maximum prompt length when truncating (default: 10000)
            remove_pii: Remove PII (emails, phones) when preprocessing (default: False)
            strip_control_chars: Remove control characters when preprocessing (default: False)
        """
        self.provider = provider
        self.repository = repository
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait

        # Preprocessing configuration
        self.preprocess = preprocess
        self.clean_whitespace = clean_whitespace
        self.truncate_prompts = truncate_prompts
        self.max_prompt_length = max_prompt_length
        self.remove_pii = remove_pii
        self.strip_control_chars = strip_control_chars

    async def run_inference(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        template: Optional[str] = None,
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
            template: Optional template to wrap prompt (e.g., "You are a tutor. {user_input}")
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with:
                - response: The LLM's text response
                - metadata: Dict with provider, tokens, latency, etc.
                - provider_response: Full ProviderResponse object
                - original_prompt: Original prompt before preprocessing (if preprocessing enabled)
                - processed_prompt: Prompt after preprocessing (if preprocessing enabled)

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
        # Store original prompt
        original_prompt = prompt

        # Apply preprocessing if enabled
        if self.preprocess:
            prompt = PromptPreprocessor.preprocess(
                prompt,
                clean_whitespace=self.clean_whitespace,
                truncate_prompts=self.truncate_prompts,
                max_length=self.max_prompt_length,
                remove_pii=self.remove_pii,
                strip_control_chars=self.strip_control_chars,
                template=template,
            )

        # Call the provider with retry logic
        provider_response: ProviderResponse = await self._call_with_retry(
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
                'preprocessing_enabled': self.preprocess,
            },
            'provider_response': provider_response,  # Include full object for advanced use
        }

        if max_tokens is not None:
            result['metadata']['max_tokens'] = max_tokens

        # Include preprocessing info if enabled
        if self.preprocess:
            result['original_prompt'] = original_prompt
            result['processed_prompt'] = prompt

        # Database logging will be added here in Phase 3.5
        # if self.repository:
        #     self.repository.insert_inference_run(...)

        return result

    async def _call_with_retry(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Internal method to call provider with retry logic.

        This method uses exponential backoff to retry transient errors like
        network issues, rate limits, or temporary service unavailability.

        Retries on:
        - TimeoutError
        - ConnectionError
        - Exception messages containing 'rate limit', 'timeout', '429', '503', '502'

        Does NOT retry on:
        - Authentication errors (401, 403)
        - Invalid requests (400)
        - Not found errors (404)
        - Permanent failures

        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters for the provider

        Returns:
            ProviderResponse from the provider

        Raises:
            RetryError: If all retry attempts are exhausted
            Exception: For non-retryable errors
        """
        # Define retry strategy with exponential backoff
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.initial_wait,
                min=self.initial_wait,
                max=self.max_wait
            ),
            retry=retry_if_exception(self._should_retry_exception),
            reraise=True,
        )

        # Wrap the provider call with retry logic
        @retry_decorator
        async def _make_call():
            return await self.provider.complete(prompt, **kwargs)

        return await _make_call()

    @staticmethod
    def _should_retry_exception(exception: BaseException) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception to evaluate

        Returns:
            True if the exception is retryable, False otherwise
        """
        if isinstance(exception, (TimeoutError, ConnectionError)):
            return True

        # Check error message for retryable conditions
        error_msg = str(exception).lower()
        retryable_patterns = [
            'rate limit',
            'timeout',
            'timed out',
            '429',  # Too Many Requests
            '503',  # Service Unavailable
            '502',  # Bad Gateway
            '504',  # Gateway Timeout
            'temporary',
            'throttl',
        ]

        return any(pattern in error_msg for pattern in retryable_patterns)

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

    async def run_repeated_inference(
        self,
        prompt: str,
        n: int,
        **kwargs: Any,
    ) -> list[dict]:
        """
        Run the same prompt multiple times to collect varied responses.

        This is useful for sampling LLM output variability, especially with
        higher temperature settings. Each request is independent and may
        produce different results.

        Args:
            prompt: The prompt to repeat
            n: Number of times to run the inference
            **kwargs: Parameters to apply to all runs (e.g., temperature, max_tokens)

        Returns:
            List of result dictionaries (same format as run_inference)

        Example:
            >>> service = InferenceService(provider)
            >>> # Get 5 different creative responses
            >>> results = await service.run_repeated_inference(
            ...     "Write a haiku about coding",
            ...     n=5,
            ...     temperature=1.0
            ... )
            >>> for i, result in enumerate(results, 1):
            ...     print(f"Response {i}: {result['response']}")

            >>> # Sample deterministic output (should be identical)
            >>> results = await service.run_repeated_inference(
            ...     "What is 2+2?",
            ...     n=3,
            ...     temperature=0.0
            ... )
            >>> # All responses should be the same
        """
        import asyncio

        # Run the same prompt n times concurrently
        tasks = [
            self.run_inference(prompt, **kwargs)
            for _ in range(n)
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
