#!/usr/bin/env python3
"""
Test the exact validation logic used in SummarizationService.
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

async def test_validation_logic(deployment: str):
    """Test the exact validation logic from SummarizationService._validate_deployment."""
    print(f"\n{'='*60}")
    print(f"Testing validation logic for: {deployment}")
    print(f"{'='*60}")
    
    try:
        # Temporarily override deployment BEFORE creating config
        original_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment
        print(f"Set AZURE_OPENAI_DEPLOYMENT to: {os.environ.get('AZURE_OPENAI_DEPLOYMENT')}")
        
        # Create fresh Config to pick up environment changes (now with correct deployment)
        fresh_config = Config()
        # Clear cache to force re-reading environment variable
        if "azure" in fresh_config._provider_configs:
            del fresh_config._provider_configs["azure"]
        provider_config = fresh_config.get_provider_config("azure")
        print(f"Config deployment_name: {provider_config.deployment_name}")
        
        try:
            # Create provider and service
            factory = ProviderFactory(fresh_config)
            provider_instance = factory.create_from_config("azure")
            print(f"Provider deployment_name: {provider_instance.deployment_name}")
            
            service = InferenceService(provider_instance, repository=None)
            
            # Try a minimal completion call (exactly as in validation)
            print(f"\nAttempting validation call with prompt: 'ping'")
            result = await service.run_inference(
                "ping", temperature=0.0, max_tokens=1
            )
            
            print(f"[OK] Validation succeeded!")
            print(f"     Response: {result.get('response', 'N/A')[:50]}")
            return True
            
        finally:
            await service.close()
            if "azure" in fresh_config._provider_configs:
                del fresh_config._provider_configs["azure"]
            
    except Exception as e:
        print(f"[ERROR] Validation failed: {type(e).__name__}: {str(e)}")
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
    """Test validation for both deployments."""
    deployments = ["gpt-4o-mini", "gpt-4o"]
    
    print("Testing validation logic (exact match to SummarizationService)...")
    print(f"Current AZURE_OPENAI_DEPLOYMENT env: {os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'NOT SET')}")
    
    results = {}
    for deployment in deployments:
        results[deployment] = await test_validation_logic(deployment)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for deployment, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {deployment}")

if __name__ == "__main__":
    asyncio.run(main())
