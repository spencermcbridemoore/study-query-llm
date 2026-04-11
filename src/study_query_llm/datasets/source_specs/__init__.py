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
from study_query_llm.datasets.source_specs.semeval2013_sra_5way import (
    SEMEVAL2013_SRA_5WAY_SLUG,
    semeval2013_sra_5way_file_specs,
    semeval2013_sra_5way_source_metadata,
)
from study_query_llm.datasets.source_specs.sources_uncertainty_zenodo import (
    SOURCES_UNCERTAINTY_QC_SLUG,
    sources_uncertainty_file_specs,
    sources_uncertainty_source_metadata,
)

__all__ = [
    "ACQUIRE_REGISTRY",
    "AUSEM_DATASET_SLUG",
    "DatasetAcquireConfig",
    "SEMEVAL2013_SRA_5WAY_SLUG",
    "SOURCES_UNCERTAINTY_QC_SLUG",
    "ausem_file_specs",
    "ausem_source_metadata",
    "semeval2013_sra_5way_file_specs",
    "semeval2013_sra_5way_source_metadata",
    "sources_uncertainty_file_specs",
    "sources_uncertainty_source_metadata",
]
