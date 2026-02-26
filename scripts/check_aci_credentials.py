"""
Quick connectivity check for ACI + HuggingFace credentials.
Read-only -- does NOT create any Azure resources.

Run with:
    python scripts/check_aci_credentials.py
"""

import os
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
rg = os.environ.get("AZURE_RESOURCE_GROUP", "")
hf_token = os.environ.get("HF_API_TOKEN", "")

print("=" * 60)
print("ACI + HuggingFace Credential Check")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Env vars
# ------------------------------------------------------------------
print("\n[1] Environment variables")
for key, val in [("AZURE_SUBSCRIPTION_ID", sub_id), ("AZURE_RESOURCE_GROUP", rg), ("HF_API_TOKEN", hf_token)]:
    if val:
        masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "(set)"
        print(f"    {key}: {masked}  OK")
    else:
        print(f"    {key}: NOT SET  FAIL")

if not sub_id or not rg:
    print("\nAzure env vars missing -- cannot continue Azure checks.")
    sys.exit(1)

# ------------------------------------------------------------------
# 2. Azure authentication
# ------------------------------------------------------------------
print("\n[2] Azure authentication (DefaultAzureCredential)")
try:
    from azure.identity import DefaultAzureCredential
    cred = DefaultAzureCredential()
    # Force a token fetch to verify the credential works
    token = cred.get_token("https://management.azure.com/.default")
    print(f"    Token acquired  OK  (expires in ~{(token.expires_on - __import__('time').time()) / 60:.0f} min)")
except Exception as exc:
    print(f"    FAIL: {exc}")
    print("    Run 'az login' and try again.")
    sys.exit(1)

# ------------------------------------------------------------------
# 3. ACI management client + list container groups (read-only)
# ------------------------------------------------------------------
print("\n[3] ACI management API (list container groups -- read-only)")
try:
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    client = ContainerInstanceManagementClient(cred, sub_id)
    groups = list(client.container_groups.list_by_resource_group(rg))
    print(f"    Resource group accessible  OK")
    if groups:
        print(f"    Existing container groups ({len(groups)}):")
        for g in groups:
            print(f"      - {g.name}  (state={g.provisioning_state})")
    else:
        print(f"    No existing container groups in '{rg}'")
except Exception as exc:
    print(f"    FAIL: {exc}")
    sys.exit(1)

# ------------------------------------------------------------------
# 4. HuggingFace token
# ------------------------------------------------------------------
print("\n[4] HuggingFace token")
if not hf_token:
    print("    HF_API_TOKEN not set -- skipped (fine for public models)")
else:
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {hf_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        print(f"    Token valid  OK  (HF user: {data.get('name', 'unknown')})")
    except urllib.error.HTTPError as exc:
        print(f"    FAIL (HTTP {exc.code}): invalid or expired token")
    except Exception as exc:
        print(f"    FAIL: {exc}")

print("\n" + "=" * 60)
print("All checks complete.")
print("=" * 60)
