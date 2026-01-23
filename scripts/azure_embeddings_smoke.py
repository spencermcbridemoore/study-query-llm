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

import os
from pathlib import Path
from typing import List, Optional, Tuple
_DEFAULT_EMBEDDING_DEPLOYMENTS = [
    "text-embedding-3-small",
    "Cohere-embed-v3-english",
    "text-embedding-ada-002",
    "Cohere-embed-v3-multilingual",
    "embed-v-4-0",
    "text-embedding-3-large",
]


def _load_env() -> dict:
    info = {
        "dotenv_available": False,
        "env_path": None,
        "env_found": False,
        "loaded_from": None,
        "manual_loaded_keys": 0,
    }
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    info["dotenv_available"] = True
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    info["env_path"] = str(env_path)
    info["env_found"] = env_path.exists()
    if info["env_found"]:
        if load_dotenv:
            load_dotenv(env_path)
        info["loaded_from"] = "repo_root"
        info["manual_loaded_keys"] = _load_env_file(env_path)
    else:
        if load_dotenv:
            load_dotenv()
        info["loaded_from"] = "default"
    return info


def _load_env_file(env_path: Path) -> int:
    loaded = 0
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    except Exception:
        return loaded
    return loaded


def _apply_env_aliases() -> list[str]:
    aliases = {
        "AZURE_ENDPOINT": "AZURE_OPENAI_ENDPOINT",
        "AZURE_API_KEY": "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_KEY": "AZURE_OPENAI_API_KEY",
    }
    used = []
    for source, target in aliases.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]
            used.append(source)
    return used


def _get_deployments() -> Tuple[List[str], Optional[str]]:
    deployments_env = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENTS", "")
    deployments = [item.strip() for item in deployments_env.split(",") if item.strip()]
    if deployments:
        return deployments, "AZURE_OPENAI_EMBEDDING_DEPLOYMENTS"

    single = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if single:
        return [single], "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"

    return _DEFAULT_EMBEDDING_DEPLOYMENTS, "DEFAULT_EMBEDDING_DEPLOYMENTS"


def main() -> None:
    load_info = _load_env()
    alias_sources = _apply_env_aliases()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

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
    if not deployments:
        raise SystemExit(
            "No embedding deployments configured. Set "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENTS (comma-separated)."
        )

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    if source == "DEFAULT_EMBEDDING_DEPLOYMENTS":
        print(
            "Using default embedding deployments. "
            "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT(S) to override."
        )
    inputs = ["first phrase", "second phrase", "third phrase"]

    print("Testing embedding deployments:")
    for name in deployments:
        try:
            response = client.embeddings.create(model=name, input=inputs)
            vector = response.data[0].embedding
            print(f"  {name}: OK (dim={len(vector)})")
        except Exception as exc:
            print(f"  {name}: FAIL ({exc})")


if __name__ == "__main__":
    main()
