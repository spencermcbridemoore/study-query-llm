"""Shared parser protocol for dataset snapshot stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow


@dataclass(frozen=True)
class ParserIdentity:
    """Stable parser identity used in parse-stage idempotency keys."""

    parser_id: str
    parser_version: str


@dataclass(frozen=True)
class ParserContext:
    """Context passed to dataset-specific parser callables."""

    dataset_group_id: int
    artifact_uris: dict[str, str]
    artifact_dir_local: Path
    source_metadata: dict[str, Any]
    parser_id: str | None = None
    parser_version: str | None = None


ParserCallable = Callable[[ParserContext], Iterable["SnapshotRow"]]
