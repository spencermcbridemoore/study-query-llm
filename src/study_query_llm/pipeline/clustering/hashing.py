"""Canonical hashing helpers for clustering effective pipelines."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from study_query_llm.algorithms.recipes import canonical_recipe_hash


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_value(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    normalized = _normalize_value(payload)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return canonical.encode("utf-8")


def effective_stage_entries(
    pipeline_resolved: list[dict[str, Any]],
    pipeline_effective: list[str],
) -> list[dict[str, Any]]:
    """Return ordered effective stage entries with normalized params."""
    params_by_stage = {
        str(entry.get("stage") or ""): dict(entry.get("params") or {})
        for entry in pipeline_resolved
    }
    return [
        {"stage": stage, "params": dict(params_by_stage.get(stage) or {})}
        for stage in pipeline_effective
    ]


def build_effective_recipe_payload(
    pipeline_resolved: list[dict[str, Any]],
    pipeline_effective: list[str],
) -> dict[str, Any]:
    """Build a recipe-like payload that encodes effective pipeline identity."""
    stages = effective_stage_entries(pipeline_resolved, pipeline_effective)
    return {
        "recipe_version": "v1",
        "stages": [
            {
                "name": str(entry["stage"]),
                "version": "1.0",
                "role": "pipeline_stage",
                "params": dict(entry.get("params") or {}),
            }
            for entry in stages
        ],
    }


def build_pipeline_effective_hash(
    pipeline_resolved: list[dict[str, Any]],
    pipeline_effective: list[str],
) -> str:
    """Hash effective stage sequence + bound stage params."""
    recipe_payload = build_effective_recipe_payload(
        pipeline_resolved=pipeline_resolved,
        pipeline_effective=pipeline_effective,
    )
    return canonical_recipe_hash(recipe_payload)
