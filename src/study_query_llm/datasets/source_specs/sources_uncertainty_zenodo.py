"""
Sources of Uncertainty in Quantum and Classical Measurement — Zenodo deposit.

See: https://zenodo.org/records/16912394 (DOI 10.5281/zenodo.16912394).
"""

from __future__ import annotations

from typing import Any, Dict, List

from study_query_llm.datasets.acquisition import FileFetchSpec, zenodo_file_download_url

SOURCES_UNCERTAINTY_QC_SLUG = "sources_uncertainty_qc"
ZENODO_RECORD_ID = 16912394
ZENODO_DOI = "10.5281/zenodo.16912394"
_DATA_FILE = "sources_v2.xlsx"


def sources_uncertainty_file_specs() -> List[FileFetchSpec]:
    return [
        FileFetchSpec(
            relative_path=_DATA_FILE,
            url=zenodo_file_download_url(ZENODO_RECORD_ID, _DATA_FILE),
        )
    ]


def sources_uncertainty_source_metadata() -> Dict[str, Any]:
    return {
        "kind": "zenodo",
        "record_id": ZENODO_RECORD_ID,
        "doi": ZENODO_DOI,
        "title": "Sources of Uncertainty in Quantum and Classical Measurement Dataset",
        "description": "Student natural-language responses (long format) with research team codes",
    }
