"""
Provider Factory - Creates LLM provider instances by name.

This factory enables dynamic provider creation and switching without
needing to import provider classes directly.
"""

import asyncio
import json
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from .base import BaseLLMProvider, DeploymentInfo
from .base_embedding import BaseEmbeddingProvider
from .azure_provider import AzureOpenAIProvider
from .azure_embedding_provider import AzureEmbeddingProvider
from .openai_compatible_embedding_provider import OpenAICompatibleEmbeddingProvider
from .openai_compatible_chat_provider import OpenAICompatibleChatProvider
from ..config import ProviderConfig, Config


class ProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    This factory provides a unified way to create providers by name,
    either from configuration or with explicit parameters.
    
    Usage:
        # From config
        factory = ProviderFactory()
        provider = factory.create_from_config("azure")
        
        # With explicit config
        config = ProviderConfig(name="azure", api_key="...", ...)
        provider = factory.create("azure", config)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the factory.
        
        Args:
            config: Optional Config instance. If not provided, creates a new one.
        """
        self.config = config or Config()

    def create(
        self,
        provider_name: str,
        provider_config: ProviderConfig
    ) -> BaseLLMProvider:
        """
        Create a provider instance from a ProviderConfig.
        
        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')
            provider_config: ProviderConfig instance with credentials and settings
        
        Returns:
            BaseLLMProvider instance
        
        Raises:
            ValueError: If provider_name is unknown
        """
        provider_name = provider_name.lower()

        if provider_name == "azure":
            return AzureOpenAIProvider(provider_config)
        # TODO: Add other providers as they are implemented
        # elif provider_name == "openai":
        #     return OpenAIProvider(provider_config)
        # elif provider_name == "hyperbolic":
        #     return HyperbolicProvider(provider_config)
        else:
            available = self.get_available_providers()
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {', '.join(available)}"
            )

    def create_from_config(self, provider_name: str) -> BaseLLMProvider:
        """
        Create a provider instance from application configuration.
        
        This method loads the provider configuration from the Config instance
        and creates the appropriate provider.
        
        Args:
            provider_name: Name of provider ('azure', 'openai', 'hyperbolic')
        
        Returns:
            BaseLLMProvider instance
        
        Raises:
            ValueError: If provider_name is unknown or not configured
        """
        provider_config = self.config.get_provider_config(provider_name)
        return self.create(provider_name, provider_config)

    @staticmethod
    def get_available_providers() -> list[str]:
        """
        Return list of supported provider names.
        
        Returns:
            List of provider names that can be created by this factory
        """
        return ["azure", "local_llm", "ollama"]

    def get_configured_providers(self) -> list[str]:
        """
        Return list of providers that have credentials configured.
        
        This checks which providers have API keys set in the environment.
        
        Returns:
            List of provider names that are configured and ready to use
        """
        return self.config.get_available_providers()
    
    # ------------------------------------------------------------------
    # Embedding provider creation
    # ------------------------------------------------------------------

    def create_embedding_provider(
        self, provider_name: str
    ) -> BaseEmbeddingProvider:
        """Create an embedding provider instance from application config.

        Args:
            provider_name: ``'azure'``, ``'openrouter'``, ``'openai'``,
                ``'huggingface'``, ``'local'``, ``'ollama'``, or any
                OpenAI-compatible label.

        Returns:
            A ``BaseEmbeddingProvider`` ready to use.

        Raises:
            ValueError: If the provider is not configured.
        """
        provider_name = provider_name.lower()
        provider_config = self.config.get_provider_config(
            provider_name if provider_name != "ollama" else "local"
        )

        if provider_name == "azure":
            return AzureEmbeddingProvider(provider_config)

        base_url = provider_config.endpoint or ""
        if provider_name == "ollama" and not base_url:
            base_url = "http://localhost:11434/v1"

        return OpenAICompatibleEmbeddingProvider(
            base_url=base_url,
            api_key=provider_config.api_key,
            provider_label=provider_name,
        )

    @staticmethod
    def get_available_embedding_providers() -> list[str]:
        """Return list of supported embedding provider names."""
        return ["azure", "openrouter", "openai", "huggingface", "local", "ollama"]

    @staticmethod
    def _join_api_url(base_url: str, path: str) -> str:
        """Join API base URL with a fixed path suffix."""
        base = str(base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("Provider endpoint is required for model discovery.")
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    @staticmethod
    def _normalize_modalities(raw: Any) -> list[str]:
        """Normalize modality list values to lowercase strings."""
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _parse_modality_side(raw: str) -> list[str]:
        """Parse one side of OpenRouter architecture modality expression."""
        out: list[str] = []
        seen: set[str] = set()
        for part in str(raw or "").split("+"):
            value = part.strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @classmethod
    def _extract_modalities(
        cls, architecture: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Extract input/output modalities from OpenRouter architecture payload."""
        input_modalities = cls._normalize_modalities(
            architecture.get("input_modalities")
        )
        output_modalities = cls._normalize_modalities(
            architecture.get("output_modalities")
        )

        raw_modality = architecture.get("modality")
        if isinstance(raw_modality, str) and "->" in raw_modality:
            left, right = raw_modality.split("->", 1)
            if not input_modalities:
                input_modalities = cls._parse_modality_side(left)
            if not output_modalities:
                output_modalities = cls._parse_modality_side(right)

        return input_modalities, output_modalities

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Best-effort positive integer coercion."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _map_openrouter_model_to_deployment(
        cls, model_row: dict[str, Any]
    ) -> Optional[DeploymentInfo]:
        """Convert one OpenRouter model row into DeploymentInfo."""
        model_id = str(
            model_row.get("id") or model_row.get("canonical_slug") or ""
        ).strip()
        if not model_id:
            return None

        architecture = model_row.get("architecture")
        if not isinstance(architecture, dict):
            architecture = {}
        input_modalities, output_modalities = cls._extract_modalities(architecture)
        supports_embeddings = "embeddings" in output_modalities
        supports_chat = "text" in output_modalities

        pricing = model_row.get("pricing")
        if not isinstance(pricing, dict):
            pricing = {}

        per_request_limits = model_row.get("per_request_limits")
        if per_request_limits is not None and not isinstance(per_request_limits, dict):
            per_request_limits = None

        supported_parameters = cls._normalize_modalities(
            model_row.get("supported_parameters")
        )
        default_parameters = model_row.get("default_parameters")
        if default_parameters is not None and not isinstance(default_parameters, dict):
            default_parameters = None

        return DeploymentInfo(
            id=model_id,
            provider="openrouter",
            capabilities={
                "chat_completion": supports_chat,
                "completion": supports_chat,
                "embeddings": supports_embeddings,
            },
            lifecycle_status=None,
            created_at=cls._safe_int(model_row.get("created")),
            context_length=cls._safe_int(model_row.get("context_length")),
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            tokenizer=str(architecture.get("tokenizer", "")).strip() or None,
            instruct_type=str(architecture.get("instruct_type", "")).strip() or None,
            pricing=pricing,
            per_request_limits=per_request_limits,
            supported_parameters=supported_parameters,
            default_parameters=default_parameters,
            metadata={
                "canonical_slug": model_row.get("canonical_slug"),
                "name": model_row.get("name"),
                "description": model_row.get("description"),
                "knowledge_cutoff": model_row.get("knowledge_cutoff"),
                "expiration_date": model_row.get("expiration_date"),
                "top_provider": model_row.get("top_provider"),
                "links": model_row.get("links"),
                "hugging_face_id": model_row.get("hugging_face_id"),
            },
        )

    async def _fetch_openrouter_models_json(
        self, endpoint: str, api_key: str, *, embeddings_only: bool
    ) -> list[dict[str, Any]]:
        """Fetch OpenRouter model catalog rows from API."""
        path = "/embeddings/models" if embeddings_only else "/models"
        url = self._join_api_url(endpoint, path)
        headers = {"Authorization": f"Bearer {api_key}"}
        req = urllib_request.Request(url, headers=headers, method="GET")

        def _fetch() -> list[dict[str, Any]]:
            with urllib_request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("OpenRouter response payload is not a JSON object.")
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError(
                    "OpenRouter response is missing 'data' list for models catalog."
                )
            return [row for row in data if isinstance(row, dict)]

        try:
            return await asyncio.to_thread(_fetch)
        except urllib_error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = str(exc)
            raise RuntimeError(
                f"OpenRouter model listing failed with status {exc.code}: "
                f"{body[:300]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"OpenRouter model listing failed: {exc}") from exc

    async def _list_openrouter_deployments(
        self, *, endpoint: str, api_key: str, modality: Optional[str]
    ) -> list[DeploymentInfo]:
        """List OpenRouter models and map them to DeploymentInfo objects."""
        rows = await self._fetch_openrouter_models_json(
            endpoint,
            api_key,
            embeddings_only=(modality == "embedding"),
        )
        deployments: list[DeploymentInfo] = []
        for row in rows:
            mapped = self._map_openrouter_model_to_deployment(row)
            if mapped is not None:
                deployments.append(mapped)
        return deployments

    def create_chat_provider(
        self, provider_name: str, model: str
    ) -> BaseLLMProvider:
        """Create a chat completion provider, with model specified at call-time.

        For Azure, the ``model`` argument is ignored -- the deployment name
        comes from config/env as usual.  For all other providers (``local_llm``,
        ``ollama``), ``model`` is forwarded verbatim as the request-body
        ``model`` field so one provider instance per model can be created
        without touching env vars.

        Args:
            provider_name: ``'azure'``, ``'openrouter'``, ``'local_llm'``, or
                ``'ollama'``.
            model: Model identifier string (e.g. ``"qwen2.5:32b"``).
                   Ignored for Azure.

        Returns:
            A ``BaseLLMProvider`` ready to use.
        """
        # Normalise 'ollama' alias to the 'local_llm' config key
        config_key = "local_llm" if provider_name == "ollama" else provider_name
        provider_config = self.config.get_provider_config(config_key)

        if provider_name == "azure":
            return AzureOpenAIProvider(provider_config)

        base_url = provider_config.endpoint or "http://localhost:11434/v1"
        return OpenAICompatibleChatProvider(
            base_url=base_url,
            model=model,
            api_key=provider_config.api_key,
            provider_label=provider_name,
        )

    @staticmethod
    def get_available_chat_providers() -> list[str]:
        """Return list of supported chat provider names."""
        return ["azure", "openrouter", "local_llm", "ollama"]

    async def list_provider_deployments(
        self,
        provider_name: str,
        modality: Optional[str] = None,
    ) -> list[DeploymentInfo]:
        """
        List available deployments/models for a provider with optional modality filtering.

        This is provider-specific functionality. Azure OpenAI and OpenRouter
        currently support listing in this method.

        Args:
            provider_name: Name of provider ('azure', 'openrouter', 'openai', 'hyperbolic')
            modality: Optional capability filter. Accepted values:
                - ``"chat"`` -- only deployments that support chat completions
                - ``"embedding"`` -- only deployments that support embeddings
                - ``None`` (default) -- return all deployments unfiltered

        Returns:
            List of DeploymentInfo objects for the provider, optionally filtered
            by modality. Deployments with no capability data are included when
            no modality filter is specified.

        Raises:
            ValueError: If provider is unknown
            NotImplementedError: If provider doesn't support listing deployments
            Exception: If unable to query deployments
        """
        provider_name = provider_name.lower()

        if provider_name == "azure":
            provider_config = self.config.get_provider_config("azure")
            deployments = await AzureOpenAIProvider.list_deployments(provider_config)
        elif provider_name == "openrouter":
            provider_config = self.config.get_provider_config("openrouter")
            deployments = await self._list_openrouter_deployments(
                endpoint=provider_config.endpoint or "https://openrouter.ai/api/v1",
                api_key=provider_config.api_key,
                modality=modality,
            )
        elif provider_name in ["openai", "hyperbolic"]:
            raise NotImplementedError(
                f"Provider '{provider_name}' does not support listing deployments. "
                f"Use model names directly in configuration."
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Known chat-model ID prefixes (Azure catalog uses model-version IDs like "gpt-4o-2024-05-13")
        _CHAT_PREFIXES = ("gpt-4", "gpt-35", "gpt-3.5", "o1", "o3")
        _EMBED_PREFIXES = ("text-embedding",)

        if modality == "chat":
            # Prefer deployments that explicitly report chat support.
            # If the API returns no capability data (capabilities={}), fall back
            # to name-based heuristics so the dropdown is never empty.
            explicit = [d for d in deployments if d.supports_chat]
            if explicit:
                deployments = explicit
            elif provider_name == "azure":
                deployments = [
                    d for d in deployments
                    if any(d.id.lower().startswith(p) for p in _CHAT_PREFIXES)
                ]
            else:
                deployments = [
                    d for d in deployments
                    if "text" in d.output_modalities or d.capabilities.get("chat_completion", False)
                ]
        elif modality == "embedding":
            explicit = [d for d in deployments if d.supports_embeddings]
            if explicit:
                deployments = explicit
            elif provider_name == "azure":
                deployments = [
                    d for d in deployments
                    if any(d.id.lower().startswith(p) for p in _EMBED_PREFIXES)
                ]
            else:
                deployments = [
                    d for d in deployments
                    if "embeddings" in d.output_modalities
                    or "embed" in d.id.lower()
                ]

        return deployments

