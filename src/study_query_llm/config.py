"""
Configuration management for Study Query LLM.

Loads configuration from environment variables (typically from a .env file).
Uses python-dotenv to load .env automatically.

Usage:
    from study_query_llm.config import config

    # Access provider configs
    azure_config = config.get_provider_config("azure")

    # Access database config
    db_url = config.database.connection_string
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse


def _repo_root() -> Path:
    """Project root: parent of ``src/`` (``study_query_llm/config.py`` → repo)."""
    return Path(__file__).resolve().parent.parent.parent


def _dotenv_candidate_paths() -> list[Path]:
    """Ordered list of .env paths; deduped by resolved path."""
    paths: list[Path] = []
    override = os.environ.get("STUDY_QUERY_LLM_DOTENV", "").strip()
    if override:
        paths.append(Path(override))
    paths.append(_repo_root() / ".env")
    paths.append(Path.cwd() / ".env")
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _load_dotenv_files() -> None:
    """
    Load ``.env`` so ``DATABASE_URL`` and other keys are set before ``Config()`` runs.

    If ``DATABASE_URL`` is missing or blank, the first existing candidate file is loaded
    with ``override=True`` so values from disk win over empty shell placeholders.
    Panel / Bokeh often run with a working directory that is not the repo root, so we
    try both the repo-root ``.env`` and ``Path.cwd() / .env``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    existing = [p for p in _dotenv_candidate_paths() if p.is_file()]
    if not existing:
        return
    db_set = bool(str(os.environ.get("DATABASE_URL", "") or "").strip())
    if not db_set:
        load_dotenv(existing[0], encoding="utf-8", override=True)
        for p in existing[1:]:
            load_dotenv(p, encoding="utf-8", override=False)
    else:
        for p in existing:
            load_dotenv(p, encoding="utf-8", override=False)


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    name: str
    api_key: str
    endpoint: Optional[str] = None
    model: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None

    def __post_init__(self):
        """Validate that required fields are present."""
        if not self.api_key:
            raise ValueError(
                f"API key not set for {self.name} provider. "
                f"Please set the appropriate environment variable."
            )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str

    def __post_init__(self):
        """Provide default if not set."""
        if not self.connection_string:
            self.connection_string = "sqlite:///study_query_llm.db"


class Config:
    """
    Application configuration loaded from environment variables.

    Environment variables can be set:
    1. In a .env file in the project root
    2. In the system environment
    3. In a container/deployment environment
    """

    def __init__(self):
        """Load configuration from environment."""
        raw_db = os.getenv("DATABASE_URL", "") or ""
        if not str(raw_db).strip():
            raw_db = "sqlite:///study_query_llm.db"
        self.database = DatabaseConfig(connection_string=str(raw_db).strip())

        # Provider configurations (lazy-loaded to avoid requiring all keys)
        self._provider_configs = {}

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')

        Returns:
            ProviderConfig with credentials and settings

        Raises:
            ValueError: If required environment variables are missing
        """
        provider_name = provider_name.lower()

        # Return cached config if already loaded
        if provider_name in self._provider_configs:
            return self._provider_configs[provider_name]

        # Load provider-specific config
        if provider_name == "azure":
            config = ProviderConfig(
                name="azure",
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
        elif provider_name == "openai":
            config = ProviderConfig(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
            )
        elif provider_name == "hyperbolic":
            config = ProviderConfig(
                name="hyperbolic",
                api_key=os.getenv("HYPERBOLIC_API_KEY", ""),
                endpoint=os.getenv("HYPERBOLIC_ENDPOINT", "https://api.hyperbolic.xyz"),
            )
        elif provider_name == "huggingface":
            config = ProviderConfig(
                name="huggingface",
                api_key=os.getenv("HF_API_TOKEN", "not-needed"),
                endpoint=os.getenv("HF_EMBEDDING_ENDPOINT", ""),
                model=os.getenv("HF_EMBEDDING_MODEL", ""),
            )
        elif provider_name == "local":
            config = ProviderConfig(
                name="local",
                api_key=os.getenv("LOCAL_EMBEDDING_API_KEY", "not-needed"),
                endpoint=os.getenv(
                    "LOCAL_EMBEDDING_ENDPOINT", "http://localhost:8080/v1"
                ),
                model=os.getenv("LOCAL_EMBEDDING_MODEL", ""),
            )
        elif provider_name == "local_llm":
            # Infrastructure-only config: endpoint + auth.
            # Model is NOT read from env vars -- it is specified per-call
            # in the sweep script's summarizer list and passed to
            # ProviderFactory.create_chat_provider(provider, model).
            config = ProviderConfig(
                name="local_llm",
                api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
                endpoint=os.getenv(
                    "LOCAL_LLM_ENDPOINT", "http://localhost:11434/v1"
                ),
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Cache for future use
        self._provider_configs[provider_name] = config
        return config

    def get_available_providers(self) -> list[str]:
        """
        Get list of providers that have API keys configured.

        Returns:
            List of provider names with credentials set
        """
        available = []
        for provider in ["azure", "openai", "hyperbolic", "huggingface", "local", "local_llm"]:
            try:
                self.get_provider_config(provider)
                available.append(provider)
            except ValueError:
                # API key not set - skip this provider
                pass
        return available


_load_dotenv_files()

# Global config instance
config = Config()


def redact_database_url(url: str) -> str:
    """Mask password in a SQLAlchemy URL for logs (best-effort)."""
    if not url or not isinstance(url, str):
        return str(url)
    try:
        p = urlparse(url.replace("postgresql+asyncpg", "postgresql", 1))
        if p.password is not None:
            user = p.username or ""
            host = p.hostname or ""
            port = f":{p.port}" if p.port else ""
            netloc = f"{user}:***@{host}{port}"
            return urlunparse((p.scheme, netloc, p.path or "", "", "", ""))
    except Exception:
        pass
    return url


def database_connection_summary(url: str) -> str:
    """Short label for logging: engine + redacted target."""
    u = (url or "").lower()
    if "postgresql" in u or "postgres" in u:
        kind = "postgresql"
    elif u.startswith("sqlite"):
        kind = "sqlite"
    else:
        kind = "other"
    return f"{kind} {redact_database_url(url)}"


def require_provider(provider_name: str) -> ProviderConfig:
    """
    Get provider config, raising helpful error if not configured.

    Args:
        provider_name: Name of provider to load

    Returns:
        ProviderConfig

    Raises:
        ValueError: With instructions on how to configure the provider
    """
    try:
        return config.get_provider_config(provider_name)
    except ValueError as e:
        raise ValueError(
            f"\n{'='*60}\n"
            f"Provider '{provider_name}' is not configured.\n\n"
            f"To use {provider_name}, set these environment variables:\n"
            f"{_get_provider_instructions(provider_name)}\n"
            f"You can set these in a .env file in the project root.\n"
            f"See .env.example for a template.\n"
            f"{'='*60}\n"
        ) from e


def _get_provider_instructions(provider_name: str) -> str:
    """Get environment variable instructions for a provider."""
    instructions = {
        "azure": """
  AZURE_OPENAI_API_KEY=your-api-key
  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT=gpt-4
  AZURE_OPENAI_API_VERSION=2024-02-15-preview
        """,
        "openai": """
  OPENAI_API_KEY=your-api-key
  OPENAI_MODEL=gpt-4
        """,
        "hyperbolic": """
  HYPERBOLIC_API_KEY=your-api-key
  HYPERBOLIC_ENDPOINT=https://api.hyperbolic.xyz
        """,
        "huggingface": """
  HF_API_TOKEN=your-hf-token          (or "not-needed" for local TEI)
  HF_EMBEDDING_ENDPOINT=https://your-endpoint.endpoints.huggingface.cloud/v1
  HF_EMBEDDING_MODEL=BAAI/bge-m3      (model hosted on the endpoint)
        """,
        "local": """
  LOCAL_EMBEDDING_ENDPOINT=http://localhost:8080/v1   (TEI / Ollama / vLLM)
  LOCAL_EMBEDDING_API_KEY=not-needed                  (override if required)
  LOCAL_EMBEDDING_MODEL=BAAI/bge-m3                   (model loaded on server)
        """,
    }
    return instructions.get(provider_name, "  (Unknown provider)")
