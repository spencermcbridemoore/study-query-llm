"""
Helper script to find a working deployment by trying common names.
"""

import asyncio
from openai import AsyncAzureOpenAI
from study_query_llm.config import config


async def try_deployment(client, deployment_name):
    """Try to make a simple completion with a deployment name."""
    try:
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5
        )
        return True, response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "DeploymentNotFound" in error_msg:
            return False, "Not deployed"
        else:
            return False, f"Error: {error_msg[:100]}"


async def find_deployment():
    """Try common deployment names to find one that works."""
    print("="*60)
    print("Finding Working Deployment")
    print("="*60)

    # Load Azure configuration
    azure_config = config.get_provider_config("azure")

    client = AsyncAzureOpenAI(
        api_key=azure_config.api_key,
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.endpoint,
    )

    # Common deployment names to try
    deployment_names = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-35-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt4o",
        "gpt4",
        "gpt35turbo",
    ]

    print("\nTrying common deployment names...")
    print("-" * 60)

    for deployment in deployment_names:
        print(f"Trying '{deployment}'... ", end="", flush=True)
        success, result = await try_deployment(client, deployment)

        if success:
            print(f"SUCCESS! Response: {result}")
            print("\n" + "="*60)
            print(f"FOUND WORKING DEPLOYMENT: {deployment}")
            print("="*60)
            print(f"\nUpdate your .env file:")
            print(f"  AZURE_OPENAI_DEPLOYMENT={deployment}")
            await client.close()
            return
        else:
            print(f"{result}")

    print("\n" + "="*60)
    print("No working deployments found")
    print("="*60)
    print("\nYou need to create a deployment in Azure Portal:")
    print("  1. Go to your Azure AI Foundry resource")
    print("  2. Click 'Deployments' in the left sidebar")
    print("  3. Click '+ Create new deployment'")
    print("  4. Select a model (e.g., gpt-4o, gpt-35-turbo)")
    print("  5. Give it a name and click Deploy")
    print("  6. Update your .env with the deployment name")

    await client.close()


if __name__ == "__main__":
    asyncio.run(find_deployment())
