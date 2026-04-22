"""Core dataclasses for the five-stage data pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

_ALLOWED_CATEGORY_VALUE_TYPES = (str, int, float, bool, type(None))


def _normalize_category_filter(raw: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
    """Validate and canonicalize category_filter for deterministic hashing."""
    if not isinstance(raw, Mapping):
        raise ValueError("category_filter must be a mapping[str, sequence]")
    if len(raw) == 0:
        raise ValueError(
            "category_filter must be non-empty when provided; "
            "pass None (or omit it) for an unconstrained snapshot"
        )

    normalized: dict[str, list[Any]] = {}
    for key in sorted(raw.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"category_filter keys must be non-empty strings, got {key!r}")
        values = raw[key]
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"category_filter[{key!r}] must be a non-empty list")
        for value in values:
            if not isinstance(value, _ALLOWED_CATEGORY_VALUE_TYPES):
                raise ValueError(
                    f"category_filter[{key!r}] values must be str/int/float/bool/None; "
                    f"got {type(value).__name__}"
                )

        deduped: dict[str, Any] = {}
        for value in values:
            deduped.setdefault(json.dumps(value, sort_keys=True), value)
        normalized[key] = sorted(
            deduped.values(),
            key=lambda item: (type(item).__name__, json.dumps(item, sort_keys=True)),
        )

    return normalized


@dataclass(frozen=True)
class SnapshotRow:
    """Normalized row contract emitted by snapshot parsers."""

    position: int
    source_id: str
    text: str
    label: int | None = None
    label_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubquerySpec:
    """Declarative snapshot selection criteria over a canonical dataframe."""

    filter_expr: str | None = None
    category_filter: Mapping[str, Sequence[Any]] | None = None
    sample_fraction: float | None = None
    sample_n: int | None = None
    sampling_seed: int | None = None
    label_mode: str = "all"

    def __post_init__(self) -> None:
        if self.category_filter is None:
            return
        normalized = _normalize_category_filter(self.category_filter)
        frozen: Mapping[str, tuple[Any, ...]] = MappingProxyType(
            {key: tuple(values) for key, values in normalized.items()}
        )
        object.__setattr__(self, "category_filter", frozen)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Serialize to deterministic JSON-compatible structure."""
        normalized_filter = (
            str(self.filter_expr).strip() if self.filter_expr is not None else None
        )
        if normalized_filter == "":
            normalized_filter = None
        payload: dict[str, Any] = {
            "filter_expr": normalized_filter,
            "sample_fraction": (
                float(self.sample_fraction) if self.sample_fraction is not None else None
            ),
            "sample_n": int(self.sample_n) if self.sample_n is not None else None,
            "sampling_seed": (
                int(self.sampling_seed) if self.sampling_seed is not None else None
            ),
            "label_mode": str(self.label_mode or "all").strip().lower(),
        }
        if self.category_filter is not None:
            payload["category_filter"] = {
                key: list(values) for key, values in self.category_filter.items()
            }
        return payload


@dataclass(frozen=True)
class StageResult:
    """Standard return object for stage entrypoints."""

    stage_name: str
    group_id: int
    run_id: int | None
    artifact_uris: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
