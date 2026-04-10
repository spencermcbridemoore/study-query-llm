"""Pinned download specs per public dataset."""

from study_query_llm.datasets.source_specs.ausem import (
    AUSEM_DATASET_SLUG,
    ausem_file_specs,
    ausem_source_metadata,
)
from study_query_llm.datasets.source_specs.registry import (
    ACQUIRE_REGISTRY,
    DatasetAcquireConfig,
)

__all__ = [
    "ACQUIRE_REGISTRY",
    "AUSEM_DATASET_SLUG",
    "DatasetAcquireConfig",
    "ausem_file_specs",
    "ausem_source_metadata",
]
