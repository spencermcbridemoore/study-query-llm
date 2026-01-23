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


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _get_deployments() -> tuple[list[str], str | None]:
    deployments_env = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENTS", "")
    deployments = [item.strip() for item in deployments_env.split(",") if item.strip()]
    if deployments:
        return deployments, None

    single = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if single:
        return [single], None

    fallback = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if fallback:
        return [fallback], "AZURE_OPENAI_DEPLOYMENT"

    return [], None


def main() -> None:
    _load_env()

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

    deployments, fallback = _get_deployments()
    if not deployments:
        raise SystemExit(
            "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENTS (comma-separated)."
        )

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    if fallback:
        print(
            f"Using {fallback} as embedding deployment. "
            "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT(S) to be explicit."
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
