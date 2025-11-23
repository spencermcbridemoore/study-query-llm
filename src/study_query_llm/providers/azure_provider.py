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
        List available Azure OpenAI model IDs.
        
        NOTE: This returns MODEL IDs, not deployment names!
        Deployment names are custom names you create in Azure Portal.
        Model IDs from this list may or may not work as deployment names.
        
        To find actual deployment names:
        1. Check Azure Portal > Your Resource > Deployments
        2. Or use find_working_deployment.py to test common names
        3. Or use Azure Management API (requires additional credentials)
        
        This is a static method that can be called without creating a provider instance,
        useful for querying available models before selecting one.
        
        Args:
            azure_config: ProviderConfig with Azure credentials (endpoint, api_key, api_version)
        
        Returns:
            List of model IDs (may work as deployment names, but not guaranteed)
        
        Raises:
            Exception: If unable to connect or list models
        """
        from typing import List
        
        client = AsyncAzureOpenAI(
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            azure_endpoint=azure_config.endpoint,
        )
        
        try:
            models = await client.models.list()
            model_ids = [model.id for model in models.data]
            return model_ids
        except Exception as e:
            raise Exception(f"Failed to list Azure models: {str(e)}") from e
        finally:
            await client.close()
    
    @staticmethod
    async def find_working_deployment(azure_config: ProviderConfig, model_ids: Optional[list[str]] = None) -> Optional[str]:
        """
        Test model IDs to find one that works as a deployment name.
        
        This tries each model ID to see if it works as a deployment name.
        This is a workaround since we can't query actual deployment names via the API.
        
        Args:
            azure_config: ProviderConfig with Azure credentials
            model_ids: Optional list of model IDs to test. If None, will list and test chat models.
        
        Returns:
            First working deployment name, or None if none found
        """
        if model_ids is None:
            model_ids = await AzureOpenAIProvider.list_deployments(azure_config)
        
        # Filter to likely chat completion models
        chat_models = [m for m in model_ids if any(
            m.startswith(prefix) for prefix in [
                'gpt-4', 'gpt-35-turbo', 'gpt-3.5', 'gpt-4o', 
                'o1', 'o3', 'claude', 'gpt-5', 'gpt-4.1', 'gpt-4.5'
            ]
        )]
        
        # Test each one
        test_client = AsyncAzureOpenAI(
            api_key=azure_config.api_key,
            api_version=azure_config.api_version,
            azure_endpoint=azure_config.endpoint,
        )
        
        try:
            for model_id in chat_models[:10]:  # Limit to first 10 to avoid too many API calls
                try:
                    # Try a minimal completion to test if this works as a deployment
                    await test_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=1
                    )
                    return model_id  # Found a working one!
                except Exception:
                    continue  # Try next one
        finally:
            await test_client.close()
        
        return None  # None found

    async def close(self):
        """Close the Azure OpenAI client connection."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes client."""
        await self.close()
