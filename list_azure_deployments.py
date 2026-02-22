"""
Helper script to list available Azure OpenAI deployments.

This will help you find the correct deployment name to use in your .env file.

Run: python list_azure_deployments.py
"""

import asyncio
from openai import AsyncAzureOpenAI
from study_query_llm.config import config


async def list_deployments():
    """List available Azure OpenAI deployments."""
    print("="*60)
    print("Azure OpenAI Deployment Checker")
    print("="*60)

    # Load Azure configuration
    print("\nLoading Azure configuration from .env...")
    try:
        azure_config = config.get_provider_config("azure")
        print(f"Endpoint: {azure_config.endpoint}")
        print(f"API Version: {azure_config.api_version}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create client
    client = AsyncAzureOpenAI(
        api_key=azure_config.api_key,
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.endpoint,
    )

    print("\nAttempting to list models/deployments...")
    try:
        # Try to list models
        models = await client.models.list()

        print("\nAvailable deployments:")
        print("-" * 60)

        deployment_found = False
        for model in models.data:
            deployment_found = True
            print(f"  ID: {model.id}")
            if hasattr(model, 'created'):
                print(f"      Created: {model.created}")
            if hasattr(model, 'owned_by'):
                print(f"      Owned by: {model.owned_by}")
            caps = getattr(model, "capabilities", None)
            if caps is not None:
                chat = getattr(caps, "chat_completion", "N/A")
                embeddings = getattr(caps, "embeddings", "N/A")
                completion = getattr(caps, "completion", "N/A")
                print(f"      Chat completion: {chat}")
                print(f"      Embeddings:      {embeddings}")
                print(f"      Completion:      {completion}")
            lifecycle = getattr(model, "lifecycle_status", None)
            if lifecycle:
                print(f"      Lifecycle:       {lifecycle}")
            print()

        if not deployment_found:
            print("  No deployments found.")
            print("\nYou need to create a deployment in Azure AI Foundry:")
            print("  1. Go to Azure AI Foundry in Azure Portal")
            print("  2. Navigate to 'Deployments' or 'Model deployments'")
            print("  3. Create a new deployment (e.g., gpt-4, gpt-35-turbo)")
            print("  4. Use the deployment name in your .env file")
        else:
            print("\nTo use one of these deployments:")
            print("  1. Copy the deployment ID from above")
            print("  2. Update your .env file:")
            print("     AZURE_OPENAI_DEPLOYMENT=<deployment-id>")

    except Exception as e:
        print(f"\nCouldn't list models: {e}")
        print("\nThis API version may not support listing models.")
        print("\nPlease check Azure Portal to find your deployment name:")
        print("  1. Go to Azure AI Foundry resource in Azure Portal")
        print("  2. Click on 'Deployments' or 'Model deployments'")
        print("  3. Look for the 'Deployment name' column")
        print("  4. Update your .env file with that exact name")

    finally:
        await client.close()

    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(list_deployments())
