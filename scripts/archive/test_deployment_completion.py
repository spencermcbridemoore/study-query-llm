#!/usr/bin/env python3
"""
Simple test script to verify text completion works with specific deployments.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from study_query_llm.config import Config
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.inference_service import InferenceService

async def test_deployment(deployment_name: str):
    """Test if a deployment can do text completion."""
    print(f"\n{'='*60}")
    print(f"Testing deployment: {deployment_name}")
    print(f"{'='*60}")
    
    # Set the deployment in environment
    original_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment_name
    
    try:
        # Create config and provider
        config = Config()
        # Clear cache to force re-reading
        if "azure" in config._provider_configs:
            del config._provider_configs["azure"]
        
        provider_config = config.get_provider_config("azure")
        print(f"Config deployment_name: {provider_config.deployment_name}")
        print(f"Config endpoint: {provider_config.endpoint}")
        print(f"Config api_version: {provider_config.api_version}")
        print(f"Config has api_key: {bool(provider_config.api_key)}")
        
        # Create provider
        factory = ProviderFactory(config)
        provider = factory.create_from_config("azure")
        print(f"Provider deployment_name: {provider.deployment_name}")
        
        # Create inference service
        service = InferenceService(provider, repository=None)
        
        # Try a simple completion
        print(f"\nAttempting text completion with prompt: 'Hello'")
        try:
            result = await service.run_inference(
                "Hello",
                temperature=0.0,
                max_tokens=5
            )
            # Check what type result is
            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                text = result.get('text', '')
                if not text and 'response' in result:
                    text = result['response'].get('text', '') if isinstance(result['response'], dict) else str(result['response'])
                print(f"[OK] Success! Response: {text[:100] if text else 'Empty response'}")
                print(f"     Result keys: {list(result.keys())}")
                print(f"     Finish reason: {result.get('finish_reason', 'N/A')}")
            else:
                print(f"[OK] Success! Response: {result.text[:100]}")
                print(f"     Finish reason: {result.finish_reason}")
            return True
        except Exception as e:
            print(f"[ERROR] Completion failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await service.close()
            
    except Exception as e:
        print(f"[ERROR] Setup failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original deployment
        if original_deployment:
            os.environ["AZURE_OPENAI_DEPLOYMENT"] = original_deployment
        elif "AZURE_OPENAI_DEPLOYMENT" in os.environ:
            del os.environ["AZURE_OPENAI_DEPLOYMENT"]

async def main():
    """Test multiple deployments."""
    deployments = ["gpt-4o-mini", "gpt-4o"]
    
    print("Testing Azure OpenAI deployments for text completion...")
    print(f"Current AZURE_OPENAI_DEPLOYMENT env: {os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'NOT SET')}")
    
    results = {}
    for deployment in deployments:
        results[deployment] = await test_deployment(deployment)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for deployment, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {deployment}")

if __name__ == "__main__":
    asyncio.run(main())
