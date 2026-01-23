"""
Smoke test Azure OpenAI embedding deployments using env configuration.

Reads from .env (repo root) and validates each embedding deployment by
requesting vectors for a small input batch.

Required env vars:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY

Optional env vars:
- AZURE_OPENAI_API_VERSION (default: 2024-02-15-preview)
- AZURE_OPENAI_EMBEDDING_DEPLOYMENTS (comma-separated)
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT (single)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

_DEBUG_LOG_PATH = r"c:\Users\spenc\Cursor Repos\study-query-llm\.cursor\debug.log"


def _log_debug(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "debug-session",
        "runId": "azure-embeddings-smoke",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = Path(_DEBUG_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as exc:
        print(f"[debug] Failed to write log file at {log_path}: {exc}")
        raise


def _load_env() -> dict:
    info = {
        "dotenv_available": False,
        "env_path": None,
        "env_found": False,
        "loaded_from": None,
    }
    try:
        from dotenv import load_dotenv
    except ImportError:
        return info

    info["dotenv_available"] = True
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    info["env_path"] = str(env_path)
    info["env_found"] = env_path.exists()
    if info["env_found"]:
        load_dotenv(env_path)
        info["loaded_from"] = "repo_root"
    else:
        load_dotenv()
        info["loaded_from"] = "default"
    return info


def _get_deployments() -> Tuple[List[str], Optional[str]]:
    deployments_env = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENTS", "")
    deployments = [item.strip() for item in deployments_env.split(",") if item.strip()]
    if deployments:
        return deployments, "AZURE_OPENAI_EMBEDDING_DEPLOYMENTS"

    single = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if single:
        return [single], "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"

    fallback = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if fallback:
        return [fallback], "AZURE_OPENAI_DEPLOYMENT"

    return [], None


def main() -> None:
    load_info = _load_env()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    # region agent log
    _log_debug(
        "H0",
        "scripts/azure_embeddings_smoke.py:module",
        "module_loaded",
        {
            "python_version": os.environ.get("PYTHON_VERSION"),
            "cwd": str(Path.cwd()),
            "script_path": str(Path(__file__).resolve()),
        },
    )
    # endregion

    # region agent log
    _log_debug(
        "H1",
        "scripts/azure_embeddings_smoke.py:main",
        "env_load_status",
        {
            "dotenv_available": load_info.get("dotenv_available"),
            "env_path": load_info.get("env_path"),
            "env_found": load_info.get("env_found"),
            "loaded_from": load_info.get("loaded_from"),
            "endpoint_present": bool(endpoint),
            "api_key_present": bool(api_key),
            "api_version": api_version,
        },
    )
    # endregion

    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")

    deployments, source = _get_deployments()
    # region agent log
    _log_debug(
        "H2",
        "scripts/azure_embeddings_smoke.py:main",
        "deployment_source",
        {
            "deployments_count": len(deployments),
            "deployment_source": source,
        },
    )
    # endregion
    if not deployments:
        raise SystemExit(
            "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENTS (comma-separated)."
        )

    from openai import AzureOpenAI

    endpoint_host = urlparse(endpoint).netloc if endpoint else None
    # region agent log
    _log_debug(
        "H5",
        "scripts/azure_embeddings_smoke.py:main",
        "client_init",
        {"endpoint_host": endpoint_host, "api_version": api_version},
    )
    # endregion
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    if source == "AZURE_OPENAI_DEPLOYMENT":
        print(
            "Using AZURE_OPENAI_DEPLOYMENT as embedding deployment. "
            "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT(S) to be explicit."
        )
    # region agent log
    _log_debug(
        "H5",
        "scripts/azure_embeddings_smoke.py:main",
        "client_ready",
        {"client_created": True, "deployment_count": len(deployments)},
    )
    # endregion

    inputs = ["first phrase", "second phrase", "third phrase"]

    print("Testing embedding deployments:")
    for name in deployments:
        # region agent log
        _log_debug(
            "H4",
            "scripts/azure_embeddings_smoke.py:main",
            "embedding_request_start",
            {"deployment": name, "inputs_count": len(inputs)},
        )
        # endregion
        try:
            response = client.embeddings.create(model=name, input=inputs)
            vector = response.data[0].embedding
            print(f"  {name}: OK (dim={len(vector)})")
            # region agent log
            _log_debug(
                "H4",
                "scripts/azure_embeddings_smoke.py:main",
                "embedding_request_ok",
                {"deployment": name, "dimension": len(vector)},
            )
            # endregion
        except Exception as exc:
            print(f"  {name}: FAIL ({exc})")
            # region agent log
            _log_debug(
                "H4",
                "scripts/azure_embeddings_smoke.py:main",
                "embedding_request_fail",
                {
                    "deployment": name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:200],
                },
            )
            # endregion


if __name__ == "__main__":
    main()
