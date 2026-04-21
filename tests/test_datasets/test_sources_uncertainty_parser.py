"""Unit tests for sources_uncertainty_qc snapshot parser behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from study_query_llm.datasets.source_specs.parser_protocol import ParserContext
from study_query_llm.datasets.source_specs.sources_uncertainty_zenodo import (
    parse_sources_uncertainty_pm_snapshot,
    parse_sources_uncertainty_snapshot,
)


def _parser_ctx(tmp_path: Path) -> ParserContext:
    xlsx_path = tmp_path / "sources_v2.xlsx"
    xlsx_path.write_bytes(b"fixture")
    return ParserContext(
        dataset_group_id=12,
        artifact_uris={"sources_v2.xlsx": "file://sources_v2.xlsx"},
        artifact_dir_local=tmp_path,
        source_metadata={"kind": "zenodo"},
    )


def _patch_table(monkeypatch: pytest.MonkeyPatch, records: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(records)
    monkeypatch.setattr(
        "study_query_llm.datasets.source_specs.sources_uncertainty_zenodo.pd.read_excel",
        lambda *_args, **_kwargs: frame,
    )


def test_parse_sources_uncertainty_snapshot_deterministic_label_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_table(
        monkeypatch,
        [
            {
                "ResponseId": "R-L",
                "response": "local source of variation",
                "Experiment": "PM",
                "code": "L",
                "updated_code": "L",
            },
            {
                "ResponseId": "R-O",
                "response": "observer effect",
                "Experiment": "BM",
                "code": "O",
                "updated_code": "O",
            },
            {
                "ResponseId": "R-P",
                "response": "procedural variation",
                "Experiment": "SG",
                "code": "P",
                "updated_code": "P",
            },
            {
                "ResponseId": "R-S",
                "response": "systematic offset",
                "Experiment": "SS",
                "code": "S",
                "updated_code": "S",
            },
            {
                "ResponseId": "R-FALLBACK",
                "response": "fallback to code column",
                "Experiment": "PM",
                "code": "O",
                "updated_code": "",
            },
        ],
    )
    rows = list(parse_sources_uncertainty_snapshot(_parser_ctx(tmp_path)))
    assert [row.label for row in rows] == [0, 1, 2, 3, 1]
    assert [row.label_name for row in rows] == ["L", "O", "P", "S", "O"]
    assert rows[-1].extra["label_source"] == "code"
    assert rows[0].extra["subset_profile"] == "all"


def test_parse_sources_uncertainty_snapshot_missing_required_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_table(
        monkeypatch,
        [
            {
                "ResponseId": "R-1",
                "response": "text",
                "Experiment": "PM",
                "code": "L",
            }
        ],
    )
    with pytest.raises(ValueError, match="missing required columns"):
        list(parse_sources_uncertainty_snapshot(_parser_ctx(tmp_path)))


def test_parse_sources_uncertainty_snapshot_unknown_label_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_table(
        monkeypatch,
        [
            {
                "ResponseId": "R-1",
                "response": "text",
                "Experiment": "PM",
                "code": "Z",
                "updated_code": "Z",
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown label code"):
        list(parse_sources_uncertainty_snapshot(_parser_ctx(tmp_path)))


def test_parse_sources_uncertainty_pm_snapshot_filters_experiment_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_table(
        monkeypatch,
        [
            {
                "ResponseId": "R-1",
                "response": "pm row one",
                "Experiment": "PM",
                "code": "L",
                "updated_code": "L",
            },
            {
                "ResponseId": "R-2",
                "response": "non-pm row",
                "Experiment": "BM",
                "code": "O",
                "updated_code": "O",
            },
            {
                "ResponseId": "R-3",
                "response": "pm row two",
                "Experiment": "PM",
                "code": "S",
                "updated_code": "S",
            },
        ],
    )
    pm_rows = list(parse_sources_uncertainty_pm_snapshot(_parser_ctx(tmp_path)))
    assert len(pm_rows) == 2
    assert [row.position for row in pm_rows] == [0, 1]
    assert {row.extra["experiment"] for row in pm_rows} == {"PM"}
    assert {row.extra["subset_profile"] for row in pm_rows} == {"experiment=PM"}
