"""
Azure OpenAI Provider Implementation.

This provider uses the Azure OpenAI Service API to generate completions.
It implements the BaseLLMProvider interface for consistent usage across providers.

Usage:
    from study_query_llm.providers.azure_provider import AzureOpenAIProvider
    from study_query_llm.config import config

    azure_config = config.get_provider_config("azure")
    provider = AzureOpenAIProvider(azure_config)

    response = await provider.complete("What is the capital of France?")
    print(response.text)
"""

import time
from typing import Any, Optional

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from .base import BaseLLMProvider, ProviderResponse
from ..config import ProviderConfig


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI Service provider implementation.

    Uses the official OpenAI Python SDK with Azure configuration.
    Supports chat completions using Azure OpenAI deployments.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize Azure OpenAI provider.

        Args:
            config: ProviderConfig with Azure credentials and settings
                - api_key: Azure OpenAI API key
                - endpoint: Azure OpenAI endpoint URL
                - deployment_name: Name of the deployed model
                - api_version: Azure OpenAI API version

        Raises:
            ValueError: If required configuration is missing
        """
        if not config.endpoint:
            raise ValueError("Azure endpoint is required")
        if not config.deployment_name:
            raise ValueError("Azure deployment name is required")
        if not config.api_version:
            raise ValueError("Azure API version is required")

        self.config = config
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
        )
        self.deployment_name = config.deployment_name

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion using Azure OpenAI.

        Args:
            prompt: The input prompt/question
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (None for model default)
            **kwargs: Additional parameters passed to the API

        Returns:
            ProviderResponse with completion text and metadata

        Raises:
            Exception: If the API call fails
        """
        start_time = time.time()

        # Build the messages for chat completion
        messages = [{"role": "user", "content": prompt}]

        # Prepare API parameters
        api_params = {
            "model": self.deployment_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        # Add any additional kwargs
        api_params.update(kwargs)

        # Make the API call
        try:
            completion: ChatCompletion = await self.client.chat.completions.create(
                **api_params
            )
        except Exception as e:
            # Re-raise with context
            raise Exception(f"Azure OpenAI API call failed: {str(e)}") from e

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Extract response data
        choice = completion.choices[0]
        response_text = choice.message.content or ""

        # Calculate token usage
        usage = completion.usage
        total_tokens = usage.total_tokens if usage else None

        # Build metadata
        metadata = {
            "model": completion.model,
            "deployment": self.deployment_name,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "temperature": temperature,
        }

        if max_tokens is not None:
            metadata["max_tokens"] = max_tokens

        return ProviderResponse(
            text=response_text,
            provider=self.get_provider_name(),
            tokens=total_tokens,
            latency_ms=latency_ms,
            metadata=metadata,
            raw_response=completion,
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"azure_openai_{self.deployment_name}"

    @staticmethod
    async def list_deployments(azure_config: ProviderConfig) -> list[str]:
        """
        List available Azure OpenAI deployments.
        
        This is a static method that can be called without creating a provider instance,
        useful for querying available deployments before selecting one.
        
        Args:
            azure_config: ProviderConfig with Azure credentials (endpoint, api_key, api_version)
        
        Returns:
            List of deployment names (model IDs)
        
        Raises:
            Exception: If unable to connect or list deployments
        """
        from typing import List
        
        client = AsyncAzureOpenAI(
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            azure_endpoint=azure_config.endpoint,
        )
        
        try:
            models = await client.models.list()
            deployment_names = [model.id for model in models.data]
            return deployment_names
        except Exception as e:
            raise Exception(f"Failed to list Azure deployments: {str(e)}") from e
        finally:
            await client.close()

    async def close(self):
        """Close the Azure OpenAI client connection."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes client."""
        await self.close()
