"""
Model Registry Service

Keeps a cached, up-to-date list of available model deployments per provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..providers.factory import ProviderFactory
from ..config import config as default_config

logger = logging.getLogger(__name__)

# Default cache TTL in seconds
DEFAULT_CACHE_TTL_SECONDS = 3600


class ModelRegistry:
    """
    Cache and refresh available model/deployment names for providers.
    """

    def __init__(
        self,
        factory: Optional[ProviderFactory] = None,
        cache_path: Optional[Path] = None,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.factory = factory or ProviderFactory(default_config)
        self.ttl_seconds = ttl_seconds
        if cache_path is None:
            cache_path = Path(".cache") / "available_models.json"
        self.cache_path = cache_path

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {"version": 1, "providers": {}}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "providers": {}}

    def _write_cache(self, data: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)

    def _is_stale(self, provider_data: dict) -> bool:
        updated_at = provider_data.get("updated_at")
        if not updated_at:
            return True
        try:
            parsed = datetime.fromisoformat(updated_at)
        except ValueError:
            return True
        age_seconds = (datetime.now(timezone.utc) - parsed).total_seconds()
        is_stale = age_seconds >= self.ttl_seconds
        if is_stale:
            logger.debug(f"Cache is stale (age: {age_seconds:.0f}s, TTL: {self.ttl_seconds}s)")
        return is_stale

    async def refresh_provider(self, provider_name: str) -> dict:
        cache = self._load_cache()
        provider_key = provider_name.lower()
        provider_data = cache["providers"].get(provider_key, {})

        try:
            deployments = await self.factory.list_provider_deployments(provider_key)
            provider_data = {
                "models": sorted(deployments),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
            logger.info(f"Refreshed cache for provider '{provider_key}': {len(deployments)} models")
        except Exception as exc:  # noqa: BLE001 - surface error in cache
            provider_data = {
                **provider_data,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            logger.warning(f"Failed to refresh cache for provider '{provider_key}': {exc}")

        cache["providers"][provider_key] = provider_data
        self._write_cache(cache)
        return provider_data

    async def list_models(
        self,
        provider_name: str = "azure",
        refresh_if_stale: bool = True,
    ) -> list[str]:
        cache = self._load_cache()
        provider_key = provider_name.lower()
        provider_data = cache["providers"].get(provider_key)

        if provider_data is None or (refresh_if_stale and self._is_stale(provider_data)):
            provider_data = await self.refresh_provider(provider_key)

        return list(provider_data.get("models", []))

    async def refresh_all(self, provider_names: Optional[list[str]] = None) -> dict:
        if provider_names is None:
            provider_names = self.factory.get_available_providers()

        results = {}
        for provider_name in provider_names:
            results[provider_name] = await self.refresh_provider(provider_name)
        return results

    def list_models_sync(
        self,
        provider_name: str = "azure",
        refresh_if_stale: bool = True,
    ) -> list[str]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.list_models(provider_name, refresh_if_stale))
        raise RuntimeError("list_models_sync called from a running event loop.")
