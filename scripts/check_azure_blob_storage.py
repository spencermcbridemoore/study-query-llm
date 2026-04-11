"""
Quick connectivity check for Azure Blob Storage (artifact backend).
Writes a test blob, reads it back, verifies checksum, then deletes it.

Run with:
    python scripts/check_azure_blob_storage.py
"""

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TEST_PATH = "_test_azure_blob_check/hello.txt"
TEST_CONTENT = b"Azure Blob storage check OK"


def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


print("=" * 60)
print("Azure Blob Storage Check")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Env vars and policy diagnostics
# ------------------------------------------------------------------
backend_type = (os.environ.get("ARTIFACT_STORAGE_BACKEND") or "local").strip().lower()
runtime_env = (os.environ.get("ARTIFACT_RUNTIME_ENV") or "dev").strip().lower()
auth_mode = (os.environ.get("ARTIFACT_AUTH_MODE") or "connection_string").strip().lower()
strict_mode = _is_truthy(os.environ.get("ARTIFACT_STORAGE_STRICT_MODE"), default=False)

base = (os.environ.get("AZURE_STORAGE_CONTAINER") or "artifacts").strip()
explicit = os.environ.get(f"AZURE_STORAGE_CONTAINER_{runtime_env.upper()}")
if explicit and explicit.strip():
    container = explicit.strip()
elif runtime_env in ("dev", "stage", "prod"):
    container = f"{base}-{runtime_env}"
else:
    container = base
prefix = (os.environ.get("AZURE_STORAGE_PREFIX") or runtime_env).strip("/")
conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "")
managed_identity_client_id = os.environ.get("AZURE_STORAGE_MANAGED_IDENTITY_CLIENT_ID", "")
max_retries = int(os.environ.get("AZURE_STORAGE_MAX_RETRIES") or "3")
retry_backoff_seconds = float(os.environ.get("AZURE_STORAGE_RETRY_BACKOFF_SECONDS") or "0.5")
verify_uploads = _is_truthy(os.environ.get("AZURE_STORAGE_VERIFY_UPLOADS"), default=True)

print("\n[1] Environment and policy")
print(f"    ARTIFACT_STORAGE_BACKEND: {backend_type}")
print(f"    ARTIFACT_RUNTIME_ENV: {runtime_env}")
print(f"    ARTIFACT_AUTH_MODE: {auth_mode}")
print(f"    ARTIFACT_STORAGE_STRICT_MODE: {strict_mode}")
print(f"    AZURE_STORAGE_CONTAINER(resolved): {container}")
print(f"    AZURE_STORAGE_PREFIX(resolved): {prefix}")
print(f"    AZURE_STORAGE_MAX_RETRIES: {max_retries}")
print(f"    AZURE_STORAGE_RETRY_BACKOFF_SECONDS: {retry_backoff_seconds}")
print(f"    AZURE_STORAGE_VERIFY_UPLOADS: {verify_uploads}")

if auth_mode == "connection_string":
    if conn_str:
        masked = conn_str[:30] + "..." + conn_str[-10:] if len(conn_str) > 45 else "(set)"
        print(f"    AZURE_STORAGE_CONNECTION_STRING: {masked}  OK")
    else:
        print("    AZURE_STORAGE_CONNECTION_STRING: NOT SET  FAIL")
elif auth_mode == "managed_identity":
    if account_url:
        print(f"    AZURE_STORAGE_ACCOUNT_URL: {account_url}  OK")
    else:
        print("    AZURE_STORAGE_ACCOUNT_URL: NOT SET  FAIL")
    if managed_identity_client_id:
        print(f"    AZURE_STORAGE_MANAGED_IDENTITY_CLIENT_ID: {managed_identity_client_id}")
else:
    print("    ARTIFACT_AUTH_MODE must be 'connection_string' or 'managed_identity'  FAIL")
    sys.exit(1)

if backend_type != "azure_blob":
    print("\nSet ARTIFACT_STORAGE_BACKEND=azure_blob before running this check.")
    sys.exit(1)

if auth_mode == "connection_string" and not conn_str:
    print("\nSet AZURE_STORAGE_CONNECTION_STRING in .env and try again.")
    sys.exit(1)

if auth_mode == "managed_identity" and not account_url:
    print("\nSet AZURE_STORAGE_ACCOUNT_URL in .env and try again.")
    sys.exit(1)

# ------------------------------------------------------------------
# 2. Create backend and test write/read/delete
# ------------------------------------------------------------------
print("\n[2] Azure Blob Storage connectivity")
try:
    from study_query_llm.storage import StorageBackendFactory

    storage = StorageBackendFactory.create(
        "azure_blob",
        container_name=container,
        auth_mode=auth_mode,
        account_url=account_url or None,
        managed_identity_client_id=managed_identity_client_id or None,
        blob_prefix=prefix,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        verify_uploads=verify_uploads,
        runtime_env=runtime_env,
    )
    blob_prefix = getattr(storage, "blob_prefix", None) or getattr(
        storage, "_blob_prefix", None
    )
    print(
        f"    Backend: {storage.backend_type} "
        f"(container={storage.container_name}, auth_mode={storage.auth_mode}, "
        f"prefix={blob_prefix or '(none)'})"
    )

    # Write
    uri = storage.write(TEST_PATH, TEST_CONTENT, content_type="text/plain")
    expected_sha = hashlib.sha256(TEST_CONTENT).hexdigest()
    print(f"    Write: OK -> {uri}")
    print(f"    Expected sha256: {expected_sha}")

    # Read + checksum verification
    data = storage.read(TEST_PATH)
    actual_sha = hashlib.sha256(data).hexdigest()
    if data == TEST_CONTENT:
        print("    Read: OK (content matches)")
    else:
        print(f"    Read: FAIL (got {len(data)} bytes, expected {len(TEST_CONTENT)})")
        sys.exit(1)
    if actual_sha != expected_sha:
        print(f"    Checksum: FAIL (actual={actual_sha}, expected={expected_sha})")
        sys.exit(1)
    print(f"    Checksum: OK ({actual_sha})")

    # Exists
    assert storage.exists(TEST_PATH), "exists() should return True"
    print("    Exists: OK")

    # Delete (cleanup)
    storage.delete(TEST_PATH)
    assert not storage.exists(TEST_PATH), "exists() should return False after delete"
    print("    Delete: OK (test blob removed)")

except ImportError as e:
    print(f"    Import FAIL: {e}")
    print("    Install with: pip install azure-storage-blob azure-identity")
    sys.exit(1)
except Exception as e:
    print(f"    FAIL: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Azure Blob Storage: SUCCESS")
print("=" * 60)
