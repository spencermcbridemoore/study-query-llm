"""Core dataclasses for the four-stage data pipeline."""

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
class StageResult:
    """Standard return object for stage entrypoints."""

    stage_name: str
    group_id: int
    run_id: int | None
    artifact_uris: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
