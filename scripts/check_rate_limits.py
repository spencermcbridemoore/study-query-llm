"""
Check OpenAI API rate limits by making a test embedding request.

Captures and displays rate limit headers from the API response.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Get API key
api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")

if not api_key:
    print("[ERROR] No API key found in environment")
    sys.exit(1)

print("=" * 80)
print("OpenAI API Rate Limits Check")
print("=" * 80)

# Models to check
models_to_check = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

for model in models_to_check:
    print(f"\n{'='*80}")
    print(f"Model: {model}")
    print("="*80)
    
    try:
        # For Azure OpenAI
        if api_base:
            url = f"{api_base}/openai/deployments/{model}/embeddings?api-version=2023-05-15"
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
        else:
            # For standard OpenAI
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        
        # Make minimal request
        payload = {
            "input": "test query for rate limits",
            "model": model if not api_base else None  # Azure uses deployment name in URL
        }
        
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        
        # Extract rate limit headers
        print(f"\nAPI Response: {response.status_code}")
        
        if response.status_code == 200:
            print("\nRate Limit Headers:")
            rate_limit_headers = {
                k: v for k, v in response.headers.items() 
                if 'ratelimit' in k.lower() or 'limit' in k.lower()
            }
            
            if rate_limit_headers:
                for header, value in sorted(rate_limit_headers.items()):
                    print(f"  {header}: {value}")
            else:
                print("  No rate limit headers found in response")
                print("\n  All response headers:")
                for k, v in sorted(response.headers.items()):
                    print(f"    {k}: {v}")
            
            # Parse the data
            data = response.json()
            if 'usage' in data:
                print(f"\nUsage for this request:")
                print(f"  Tokens: {data['usage'].get('total_tokens', 'N/A')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"[ERROR] Failed to check {model}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("Rate Limit Check Complete")
print("="*80)
print("\nNote: For complete rate limit details, visit:")
print("  https://platform.openai.com/settings/organization/limits")
