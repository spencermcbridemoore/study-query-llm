"""Pinned download specs per public dataset."""

from study_query_llm.datasets.source_specs.ausem import (
    AUSEM_DATASET_SLUG,
    ausem_file_specs,
    parse_ausem_problem1_snapshot,
    parse_ausem_problem2_snapshot,
    parse_ausem_problem3_snapshot,
    parse_ausem_problem4_snapshot,
    parse_ausem_snapshot,
    ausem_source_metadata,
)
from study_query_llm.datasets.source_specs.banking77 import (
    BANKING77_DATASET_SLUG,
    BANKING77_HF_DATASET,
    BANKING77_HF_REVISION,
    banking77_file_specs,
    banking77_resolve_url,
    banking77_source_metadata,
    parse_banking77_snapshot,
)
from study_query_llm.datasets.source_specs.estela import (
    ESTELA_DATASET_SLUG,
    estela_file_specs,
    estela_source_metadata,
    parse_estela_snapshot,
)
from study_query_llm.datasets.source_specs.registry import (
    ACQUIRE_REGISTRY,
    DatasetAcquireConfig,
)
from study_query_llm.datasets.source_specs.parser_protocol import (
    ParserCallable,
    ParserContext,
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
    sources_uncertainty_file_specs,
    sources_uncertainty_source_metadata,
)
from study_query_llm.datasets.source_specs.twenty_newsgroups import (
    TWENTY_NEWSGROUPS_DATASET_SLUG,
    TWENTY_NEWSGROUPS_DEFAULT_PARSER_ID,
    TWENTY_NEWSGROUPS_DEFAULT_PARSER_VERSION,
    TWENTY_NEWSGROUPS_6CAT,
    TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE,
    parse_twenty_newsgroups_snapshot,
    twenty_newsgroups_6cat_subquery_spec,
    twenty_newsgroups_file_specs,
    twenty_newsgroups_source_metadata,
)

__all__ = [
    "ACQUIRE_REGISTRY",
    "AUSEM_DATASET_SLUG",
    "BANKING77_DATASET_SLUG",
    "BANKING77_HF_DATASET",
    "BANKING77_HF_REVISION",
    "DatasetAcquireConfig",
    "ESTELA_DATASET_SLUG",
    "ParserCallable",
    "ParserContext",
    "SEMEVAL2013_SRA_5WAY_SLUG",
    "SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_ID",
    "SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION",
    "SOURCES_UNCERTAINTY_QC_SLUG",
    "TWENTY_NEWSGROUPS_DATASET_SLUG",
    "TWENTY_NEWSGROUPS_DEFAULT_PARSER_ID",
    "TWENTY_NEWSGROUPS_DEFAULT_PARSER_VERSION",
    "TWENTY_NEWSGROUPS_6CAT",
    "TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE",
    "ausem_file_specs",
    "parse_ausem_problem1_snapshot",
    "parse_ausem_problem2_snapshot",
    "parse_ausem_problem3_snapshot",
    "parse_ausem_problem4_snapshot",
    "parse_ausem_snapshot",
    "ausem_source_metadata",
    "banking77_file_specs",
    "banking77_resolve_url",
    "banking77_source_metadata",
    "estela_file_specs",
    "estela_source_metadata",
    "parse_banking77_snapshot",
    "parse_estela_snapshot",
    "parse_semeval2013_sra_5way_snapshot",
    "semeval2013_sra_5way_file_specs",
    "semeval2013_sra_5way_source_metadata",
    "sources_uncertainty_file_specs",
    "sources_uncertainty_source_metadata",
    "parse_twenty_newsgroups_snapshot",
    "twenty_newsgroups_6cat_subquery_spec",
    "twenty_newsgroups_file_specs",
    "twenty_newsgroups_source_metadata",
]
