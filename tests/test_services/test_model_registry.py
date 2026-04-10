"""Tests for ModelRegistry provider cache behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import pytest

from study_query_llm.providers.base import DeploymentInfo
from study_query_llm.services.model_registry import ModelRegistry


@pytest.mark.asyncio
async def test_refresh_provider_persists_extended_model_metadata(tmp_path):
    """refresh_provider writes extended metadata fields for each model."""
    deployment_infos = [
        DeploymentInfo(
            id="openai/text-embedding-3-small",
            provider="openrouter",
            capabilities={"embeddings": True, "chat_completion": False},
            context_length=8192,
            input_modalities=["text"],
            output_modalities=["embeddings"],
            tokenizer="cl100k_base",
            instruct_type=None,
            pricing={"prompt": "0.00000002", "completion": "0"},
            per_request_limits={"max_input_tokens": 8192},
            supported_parameters=["max_tokens"],
            default_parameters={"temperature": None},
            metadata={"name": "text-embedding-3-small"},
        )
    ]
    factory = Mock()
    factory.list_provider_deployments = AsyncMock(return_value=deployment_infos)
    factory.get_available_providers = Mock(return_value=["openrouter"])

    cache_path = tmp_path / "available_models.json"
    registry = ModelRegistry(factory=factory, cache_path=cache_path)

    result = await registry.refresh_provider("openrouter")
    details = result["model_details"]["openai/text-embedding-3-small"]

    assert details["context_length"] == 8192
    assert details["input_modalities"] == ["text"]
    assert details["output_modalities"] == ["embeddings"]
    assert details["pricing"]["prompt"] == "0.00000002"
    assert details["metadata"]["name"] == "text-embedding-3-small"

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    persisted = payload["providers"]["openrouter"]["model_details"][
        "openai/text-embedding-3-small"
    ]
    assert persisted["context_length"] == 8192
    assert persisted["supported_parameters"] == ["max_tokens"]


@pytest.mark.asyncio
async def test_list_models_by_modality_uses_cached_capabilities(tmp_path):
    """list_models_by_modality filters from cached model details."""
    deployment_infos = [
        DeploymentInfo(
            id="openai/gpt-4o-mini",
            provider="openrouter",
            capabilities={"chat_completion": True, "embeddings": False},
        ),
        DeploymentInfo(
            id="openai/text-embedding-3-small",
            provider="openrouter",
            capabilities={"chat_completion": False, "embeddings": True},
        ),
    ]
    factory = Mock()
    factory.list_provider_deployments = AsyncMock(return_value=deployment_infos)
    factory.get_available_providers = Mock(return_value=["openrouter"])

    registry = ModelRegistry(factory=factory, cache_path=tmp_path / "models.json")

    embeddings = await registry.list_models_by_modality(
        "openrouter", modality="embedding"
    )
    chats = await registry.list_models_by_modality("openrouter", modality="chat")

    assert embeddings == ["openai/text-embedding-3-small"]
    assert chats == ["openai/gpt-4o-mini"]


@pytest.mark.asyncio
async def test_list_models_refreshes_when_cache_is_stale(tmp_path):
    """With ttl_seconds=0, each read refreshes provider cache."""
    first = [
        DeploymentInfo(
            id="model-a",
            provider="openrouter",
            capabilities={"embeddings": True},
        )
    ]
    second = [
        DeploymentInfo(
            id="model-b",
            provider="openrouter",
            capabilities={"embeddings": True},
        )
    ]
    factory = Mock()
    factory.list_provider_deployments = AsyncMock(side_effect=[first, second])
    factory.get_available_providers = Mock(return_value=["openrouter"])

    registry = ModelRegistry(
        factory=factory,
        cache_path=tmp_path / "stale_models.json",
        ttl_seconds=0,
    )

    models_first = await registry.list_models("openrouter", refresh_if_stale=True)
    models_second = await registry.list_models("openrouter", refresh_if_stale=True)

    assert models_first == ["model-a"]
    assert models_second == ["model-b"]
    assert factory.list_provider_deployments.await_count == 2
