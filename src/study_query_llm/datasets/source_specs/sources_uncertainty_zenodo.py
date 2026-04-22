"""
Sources of Uncertainty in Quantum and Classical Measurement — Zenodo deposit.

See: https://zenodo.org/records/16912394 (DOI 10.5281/zenodo.16912394).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import pandas as pd

from study_query_llm.datasets.acquisition import FileFetchSpec, zenodo_file_download_url
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow

SOURCES_UNCERTAINTY_QC_SLUG = "sources_uncertainty_qc"
ZENODO_RECORD_ID = 16912394
ZENODO_DOI = "10.5281/zenodo.16912394"
SOURCES_UNCERTAINTY_DEFAULT_PARSER_ID = "sources_uncertainty_qc.default"
SOURCES_UNCERTAINTY_DEFAULT_PARSER_VERSION = "v1"
_DATA_FILE = "sources_v2.xlsx"
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ResponseId",
    "response",
    "Experiment",
    "code",
    "updated_code",
)
_LABEL_CODE_ORDER: tuple[str, ...] = ("L", "O", "P", "S")
_LABEL_TO_INT: dict[str, int] = {code: idx for idx, code in enumerate(_LABEL_CODE_ORDER)}


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


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).replace("\x00", "").strip()


def _load_sources_rows(ctx: ParserContext) -> list[dict[str, Any]]:
    xlsx_path = ctx.artifact_dir_local / _DATA_FILE
    if not xlsx_path.is_file():
        raise ValueError(f"sources_uncertainty_qc parser expected file missing: {xlsx_path}")
    try:
        table = pd.read_excel(xlsx_path, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            "sources_uncertainty_qc parser requires openpyxl to read sources_v2.xlsx"
        ) from exc
    missing = [name for name in _REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(
            f"sources_uncertainty_qc parser missing required columns: {missing}; "
            f"got={list(table.columns)}"
        )
    return list(table.to_dict(orient="records"))


def _normalize_label(
    raw_row: dict[str, Any],
    *,
    row_index: int,
) -> tuple[int, str, str]:
    preferred_code = _clean_cell(raw_row.get("updated_code")).upper()
    fallback_code = _clean_cell(raw_row.get("code")).upper()
    if preferred_code:
        label_code = preferred_code
        label_source = "updated_code"
    else:
        label_code = fallback_code
        label_source = "code"
    if not label_code:
        raise ValueError(f"sources_uncertainty_qc row={row_index} missing code/updated_code")
    if label_code not in _LABEL_TO_INT:
        raise ValueError(
            f"sources_uncertainty_qc row={row_index} has unknown label code {label_code!r}; "
            f"expected one of {sorted(_LABEL_TO_INT.keys())}"
        )
    return _LABEL_TO_INT[label_code], label_code, label_source


def _build_snapshot_rows(
    raw_rows: list[dict[str, Any]],
    *,
    experiment_filter: str | None,
    subset_profile: str,
) -> list["SnapshotRow"]:
    from study_query_llm.pipeline.types import SnapshotRow

    rows: list[SnapshotRow] = []
    for raw_index, raw_row in enumerate(raw_rows):
        experiment = _clean_cell(raw_row.get("Experiment")).upper()
        if experiment_filter and experiment != experiment_filter.upper():
            continue
        text = _clean_cell(raw_row.get("response"))
        if not text:
            raise ValueError(f"sources_uncertainty_qc row={raw_index} has empty response text")
        response_id = _clean_cell(raw_row.get("ResponseId"))
        if not response_id:
            raise ValueError(f"sources_uncertainty_qc row={raw_index} has empty ResponseId")
        label, label_name, label_source = _normalize_label(raw_row, row_index=raw_index)
        canonical_code = _clean_cell(raw_row.get("code")).upper()
        canonical_updated_code = _clean_cell(raw_row.get("updated_code")).upper()
        rows.append(
            SnapshotRow(
                position=len(rows),
                source_id=f"{experiment}:{response_id}:{raw_index}",
                text=text,
                label=label,
                label_name=label_name,
                extra={
                    "response_id": response_id,
                    "experiment": experiment,
                    "code": canonical_code,
                    "updated_code": canonical_updated_code,
                    "label_source": label_source,
                    "subset_profile": subset_profile,
                },
            )
        )
    if not rows:
        scope = experiment_filter.upper() if experiment_filter else "all"
        raise ValueError(
            f"sources_uncertainty_qc parser produced no rows for scope={scope}; "
            "check filter settings and source file contents"
        )
    return rows


def parse_sources_uncertainty_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Default parser for all experiments in sources_v2.xlsx."""
    raw_rows = _load_sources_rows(ctx)
    return _build_snapshot_rows(
        raw_rows,
        experiment_filter=None,
        subset_profile="all",
    )


def parse_sources_uncertainty_pm_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """PM-only parser variant for focused analyses."""
    raw_rows = _load_sources_rows(ctx)
    return _build_snapshot_rows(
        raw_rows,
        experiment_filter="PM",
        subset_profile="experiment=PM",
    )
