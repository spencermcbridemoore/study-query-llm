"""
Azure OpenAI utility functions.

Helper functions for querying Azure OpenAI deployments and models.
"""

import asyncio
from typing import List, Optional
from openai import AsyncAzureOpenAI
from ..config import ProviderConfig


async def list_azure_deployments(azure_config: ProviderConfig) -> List[str]:
    """
    List available Azure OpenAI deployments.
    
    Args:
        azure_config: ProviderConfig with Azure credentials
        
    Returns:
        List of deployment names (model IDs)
        
    Raises:
        Exception: If unable to connect or list deployments
    """
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
        # If listing fails, return empty list
        # The error will be handled by the caller
        raise Exception(f"Failed to list Azure deployments: {str(e)}") from e
    finally:
        await client.close()


async def test_azure_deployment(
    azure_config: ProviderConfig,
    deployment_name: str
) -> tuple[bool, Optional[str]]:
    """
    Test if a deployment name is valid by making a test call.
    
    Args:
        azure_config: ProviderConfig with Azure credentials
        deployment_name: Name of deployment to test
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    client = AsyncAzureOpenAI(
        api_key=azure_config.api_key,
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.endpoint,
    )
    
    try:
        # Make a minimal test call
        await client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "DeploymentNotFound" in error_msg or "deployment not found" in error_msg.lower():
            return False, "Deployment not found"
        return False, error_msg
    finally:
        await client.close()

