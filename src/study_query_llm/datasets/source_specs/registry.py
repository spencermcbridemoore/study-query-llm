"""Registry mapping dataset slug to file specs and source metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.ausem import (
    AUSEM_DATASET_SLUG,
    ausem_file_specs,
    ausem_source_metadata,
)


@dataclass(frozen=True)
class DatasetAcquireConfig:
    slug: str
    file_specs: Callable[[], List[FileFetchSpec]]
    source_metadata: Callable[[], Dict[str, Any]]


ACQUIRE_REGISTRY: Dict[str, DatasetAcquireConfig] = {
    AUSEM_DATASET_SLUG: DatasetAcquireConfig(
        slug=AUSEM_DATASET_SLUG,
        file_specs=ausem_file_specs,
        source_metadata=ausem_source_metadata,
    ),
}
