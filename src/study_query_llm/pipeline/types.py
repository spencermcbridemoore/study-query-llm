"""Core dataclasses for the five-stage data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    sample_fraction: float | None = None
    sample_n: int | None = None
    sampling_seed: int | None = None
    label_mode: str = "all"

    def to_canonical_dict(self) -> dict[str, Any]:
        """Serialize to deterministic JSON-compatible structure."""
        normalized_filter = (
            str(self.filter_expr).strip() if self.filter_expr is not None else None
        )
        if normalized_filter == "":
            normalized_filter = None
        return {
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


@dataclass(frozen=True)
class StageResult:
    """Standard return object for stage entrypoints."""

    stage_name: str
    group_id: int
    run_id: int | None
    artifact_uris: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
