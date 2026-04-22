"""Registry mapping dataset slug to file specs and source metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.banking77 import (
    BANKING77_DATASET_SLUG,
    BANKING77_DEFAULT_PARSER_ID,
    BANKING77_DEFAULT_PARSER_VERSION,
    banking77_file_specs,
    banking77_source_metadata,
    parse_banking77_snapshot,
)
from study_query_llm.datasets.source_specs.estela import (
    ESTELA_DATASET_SLUG,
    ESTELA_DEFAULT_PARSER_ID,
    ESTELA_DEFAULT_PARSER_VERSION,
    estela_file_specs,
    estela_source_metadata,
    parse_estela_snapshot,
)
from study_query_llm.datasets.source_specs.parser_protocol import ParserCallable
from study_query_llm.datasets.source_specs.ausem import (
    AUSEM_DATASET_SLUG,
    AUSEM_DEFAULT_PARSER_ID,
    AUSEM_DEFAULT_PARSER_VERSION,
    ausem_file_specs,
    parse_ausem_snapshot,
    ausem_source_metadata,
)
from study_query_llm.datasets.source_specs.semeval2013_sra_5way import (
    SEMEVAL2013_SRA_5WAY_SLUG,
    SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_ID,
    SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION,
    parse_semeval2013_sra_5way_snapshot,
    semeval2013_sra_5way_file_specs,
    semeval2013_sra_5way_source_metadata,
)
from study_query_llm.datasets.source_specs.sources_uncertainty_zenodo import (
    SOURCES_UNCERTAINTY_QC_SLUG,
    SOURCES_UNCERTAINTY_DEFAULT_PARSER_ID,
    SOURCES_UNCERTAINTY_DEFAULT_PARSER_VERSION,
    parse_sources_uncertainty_snapshot,
    sources_uncertainty_file_specs,
    sources_uncertainty_source_metadata,
)
from study_query_llm.datasets.source_specs.twenty_newsgroups import (
    TWENTY_NEWSGROUPS_DATASET_SLUG,
    TWENTY_NEWSGROUPS_DEFAULT_PARSER_ID,
    TWENTY_NEWSGROUPS_DEFAULT_PARSER_VERSION,
    parse_twenty_newsgroups_snapshot,
    twenty_newsgroups_file_specs,
    twenty_newsgroups_source_metadata,
)


@dataclass(frozen=True)
class DatasetAcquireConfig:
    slug: str
    file_specs: Callable[[], List[FileFetchSpec]]
    source_metadata: Callable[[], Dict[str, Any]]
    default_parser: ParserCallable | None = None
    default_parser_id: str | None = None
    default_parser_version: str | None = None

    def __post_init__(self) -> None:
        has_parser = self.default_parser is not None
        has_id = bool(str(self.default_parser_id or "").strip())
        has_version = bool(str(self.default_parser_version or "").strip())
        if has_parser and (not has_id or not has_version):
            raise ValueError(
                "DatasetAcquireConfig with default_parser requires both "
                "default_parser_id and default_parser_version."
            )


ACQUIRE_REGISTRY: Dict[str, DatasetAcquireConfig] = {
    BANKING77_DATASET_SLUG: DatasetAcquireConfig(
        slug=BANKING77_DATASET_SLUG,
        file_specs=banking77_file_specs,
        source_metadata=banking77_source_metadata,
        default_parser=parse_banking77_snapshot,
        default_parser_id=BANKING77_DEFAULT_PARSER_ID,
        default_parser_version=BANKING77_DEFAULT_PARSER_VERSION,
    ),
    ESTELA_DATASET_SLUG: DatasetAcquireConfig(
        slug=ESTELA_DATASET_SLUG,
        file_specs=estela_file_specs,
        source_metadata=estela_source_metadata,
        default_parser=parse_estela_snapshot,
        default_parser_id=ESTELA_DEFAULT_PARSER_ID,
        default_parser_version=ESTELA_DEFAULT_PARSER_VERSION,
    ),
    AUSEM_DATASET_SLUG: DatasetAcquireConfig(
        slug=AUSEM_DATASET_SLUG,
        file_specs=ausem_file_specs,
        source_metadata=ausem_source_metadata,
        default_parser=parse_ausem_snapshot,
        default_parser_id=AUSEM_DEFAULT_PARSER_ID,
        default_parser_version=AUSEM_DEFAULT_PARSER_VERSION,
    ),
    SOURCES_UNCERTAINTY_QC_SLUG: DatasetAcquireConfig(
        slug=SOURCES_UNCERTAINTY_QC_SLUG,
        file_specs=sources_uncertainty_file_specs,
        source_metadata=sources_uncertainty_source_metadata,
        default_parser=parse_sources_uncertainty_snapshot,
        default_parser_id=SOURCES_UNCERTAINTY_DEFAULT_PARSER_ID,
        default_parser_version=SOURCES_UNCERTAINTY_DEFAULT_PARSER_VERSION,
    ),
    SEMEVAL2013_SRA_5WAY_SLUG: DatasetAcquireConfig(
        slug=SEMEVAL2013_SRA_5WAY_SLUG,
        file_specs=semeval2013_sra_5way_file_specs,
        source_metadata=semeval2013_sra_5way_source_metadata,
        default_parser=parse_semeval2013_sra_5way_snapshot,
        default_parser_id=SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_ID,
        default_parser_version=SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION,
    ),
    TWENTY_NEWSGROUPS_DATASET_SLUG: DatasetAcquireConfig(
        slug=TWENTY_NEWSGROUPS_DATASET_SLUG,
        file_specs=twenty_newsgroups_file_specs,
        source_metadata=twenty_newsgroups_source_metadata,
        default_parser=parse_twenty_newsgroups_snapshot,
        default_parser_id=TWENTY_NEWSGROUPS_DEFAULT_PARSER_ID,
        default_parser_version=TWENTY_NEWSGROUPS_DEFAULT_PARSER_VERSION,
    ),
}
