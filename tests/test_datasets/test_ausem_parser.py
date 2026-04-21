"""Unit tests for AuSeM snapshot parser profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from study_query_llm.datasets.source_specs.ausem import (
    parse_ausem_problem1_snapshot,
    parse_ausem_problem3_snapshot,
    parse_ausem_snapshot,
)
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext

_PROBLEM_KEYS: tuple[str, ...] = ("problem1", "problem2", "problem3", "problem4")


def _write_problem_csv(
    tmp_path: Path,
    *,
    problem_key: str,
    records: list[dict[str, object]],
) -> None:
    target = tmp_path / "Student_Explanations" / f"{problem_key}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(target, index=False, encoding="utf-8")


def _parser_ctx(tmp_path: Path) -> ParserContext:
    artifact_uris: dict[str, str] = {}
    for problem_key in _PROBLEM_KEYS:
        rel_path = f"Student_Explanations/{problem_key}.csv"
        csv_path = tmp_path / rel_path
        if csv_path.is_file():
            artifact_uris[rel_path] = f"file://{rel_path}"
    return ParserContext(
        dataset_group_id=77,
        artifact_uris=artifact_uris,
        artifact_dir_local=tmp_path,
        source_metadata={"kind": "github_raw"},
    )


def test_parse_ausem_snapshot_all_problem_profile(tmp_path: Path) -> None:
    _write_problem_csv(
        tmp_path,
        problem_key="problem1",
        records=[{"correct": "TRUE", "student_id": "Student 1", "Text": "p1 text"}],
    )
    _write_problem_csv(
        tmp_path,
        problem_key="problem2",
        records=[{"correct": "FALSE", "student_id": "Student 2", "Text": "p2 text"}],
    )
    _write_problem_csv(
        tmp_path,
        problem_key="problem3",
        records=[{"correct": "1", "student_id": "Student 3", "Text": "p3 text"}],
    )
    _write_problem_csv(
        tmp_path,
        problem_key="problem4",
        records=[{"correct": "0", "student_id": "Student 4", "Text": "p4 text"}],
    )

    rows = list(parse_ausem_snapshot(_parser_ctx(tmp_path)))
    assert len(rows) == 4
    assert [row.position for row in rows] == [0, 1, 2, 3]
    assert [row.extra["problem"] for row in rows] == [
        "problem1",
        "problem2",
        "problem3",
        "problem4",
    ]
    assert [row.label for row in rows] == [1, 0, 1, 0]
    assert [row.label_name for row in rows] == [
        "correct",
        "incorrect",
        "correct",
        "incorrect",
    ]
    assert {row.extra["subset_profile"] for row in rows} == {"all"}
    assert rows[0].source_id.startswith("problem1:Student 1:")


def test_parse_ausem_problem3_profile_filters_to_single_problem(tmp_path: Path) -> None:
    for index, problem_key in enumerate(_PROBLEM_KEYS, start=1):
        _write_problem_csv(
            tmp_path,
            problem_key=problem_key,
            records=[
                {
                    "correct": True,
                    "student_id": f"Student {index}",
                    "Text": f"{problem_key} explanation",
                }
            ],
        )

    rows = list(parse_ausem_problem3_snapshot(_parser_ctx(tmp_path)))
    assert len(rows) == 1
    assert rows[0].position == 0
    assert rows[0].extra["problem"] == "problem3"
    assert rows[0].extra["subset_profile"] == "problem=3"
    assert rows[0].source_id.startswith("problem3:")


def test_parse_ausem_problem1_snapshot_missing_required_column(tmp_path: Path) -> None:
    _write_problem_csv(
        tmp_path,
        problem_key="problem1",
        records=[{"correct": True, "student_id": "Student 1"}],
    )
    with pytest.raises(ValueError, match="missing required columns"):
        list(parse_ausem_problem1_snapshot(_parser_ctx(tmp_path)))


def test_parse_ausem_problem1_snapshot_rejects_unknown_correct_value(tmp_path: Path) -> None:
    _write_problem_csv(
        tmp_path,
        problem_key="problem1",
        records=[{"correct": "maybe", "student_id": "Student 1", "Text": "text"}],
    )
    with pytest.raises(ValueError, match="unknown 'correct' value"):
        list(parse_ausem_problem1_snapshot(_parser_ctx(tmp_path)))
