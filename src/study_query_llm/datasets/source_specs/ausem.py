"""
AuSeM (Automating Sensemaking Measurements) — public CSVs from tufts-ml/AuSeM.

Pinned ref: commit on main at documentation time (reproducible raw.githubusercontent.com URLs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import pandas as pd

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow

AUSEM_DATASET_SLUG = "ausem"
AUSEM_GITHUB_ORG = "tufts-ml"
AUSEM_GITHUB_REPO = "AuSeM"
# Pinned commit (repo main as of integration); update intentionally when bumping data version.
AUSEM_PINNED_GIT_REF = "271b93baa0ab9aae70806c7364a4c4304f927143"

_STUDENT_EXPLANATION_CSV: tuple[str, ...] = (
    "problem1.csv",
    "problem2.csv",
    "problem3.csv",
    "problem4.csv",
)
_PROBLEM_KEY_ORDER: tuple[str, ...] = (
    "problem1",
    "problem2",
    "problem3",
    "problem4",
)
_PROBLEM_FILE_BY_KEY: dict[str, str] = {
    problem_key: f"Student_Explanations/{problem_key}.csv"
    for problem_key in _PROBLEM_KEY_ORDER
}
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "correct",
    "student_id",
    "Text",
)


def ausem_raw_url(relative_repo_path: str) -> str:
    rel = relative_repo_path.lstrip("/")
    return (
        f"https://raw.githubusercontent.com/{AUSEM_GITHUB_ORG}/"
        f"{AUSEM_GITHUB_REPO}/{AUSEM_PINNED_GIT_REF}/{rel}"
    )


def ausem_file_specs() -> List[FileFetchSpec]:
    """Four Student_Explanations problem CSVs."""
    specs: List[FileFetchSpec] = []
    for name in _STUDENT_EXPLANATION_CSV:
        rel = f"Student_Explanations/{name}"
        specs.append(FileFetchSpec(relative_path=rel, url=ausem_raw_url(rel)))
    return specs


def ausem_source_metadata() -> Dict[str, Any]:
    """`source` block for acquisition.json."""
    return {
        "kind": "github_raw",
        "organization": AUSEM_GITHUB_ORG,
        "repository": AUSEM_GITHUB_REPO,
        "git_ref": AUSEM_PINNED_GIT_REF,
        "description": "AuSeM Student_Explanations problem CSVs",
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


def _normalize_correct_label(
    value: Any,
    *,
    problem_key: str,
    row_index: int,
) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False
    normalized = _clean_cell(value).lower()
    truthy = {"true", "t", "1", "yes"}
    falsy = {"false", "f", "0", "no"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError(
        f"ausem parser row={row_index} in {problem_key} has unknown 'correct' value {value!r}"
    )


def _load_problem_rows(
    ctx: ParserContext,
    *,
    selected_problem_keys: tuple[str, ...],
) -> list[tuple[str, int, dict[str, Any]]]:
    loaded: list[tuple[str, int, dict[str, Any]]] = []
    for problem_key in selected_problem_keys:
        relative_path = _PROBLEM_FILE_BY_KEY[problem_key]
        csv_path = ctx.artifact_dir_local / relative_path
        if not csv_path.is_file():
            raise ValueError(f"ausem parser expected file missing: {csv_path}")
        table = pd.read_csv(csv_path, encoding="utf-8")
        missing = [column for column in _REQUIRED_COLUMNS if column not in table.columns]
        if missing:
            raise ValueError(
                f"ausem parser missing required columns in {relative_path}: {missing}; "
                f"got={list(table.columns)}"
            )
        for raw_index, raw_row in enumerate(table.to_dict(orient="records")):
            loaded.append((problem_key, raw_index, raw_row))
    return loaded


def _build_snapshot_rows(
    raw_rows: list[tuple[str, int, dict[str, Any]]],
    *,
    subset_profile: str,
) -> list["SnapshotRow"]:
    from study_query_llm.pipeline.types import SnapshotRow

    rows: list[SnapshotRow] = []
    for problem_key, raw_index, raw_row in raw_rows:
        text = _clean_cell(raw_row.get("Text"))
        if not text:
            raise ValueError(f"ausem parser row={raw_index} in {problem_key} has empty Text")
        student_id = _clean_cell(raw_row.get("student_id"))
        if not student_id:
            raise ValueError(
                f"ausem parser row={raw_index} in {problem_key} has empty student_id"
            )
        is_correct = _normalize_correct_label(
            raw_row.get("correct"),
            problem_key=problem_key,
            row_index=raw_index,
        )
        rows.append(
            SnapshotRow(
                position=len(rows),
                source_id=f"{problem_key}:{student_id}:{raw_index}",
                text=text,
                label=1 if is_correct else 0,
                label_name="correct" if is_correct else "incorrect",
                extra={
                    "problem": problem_key,
                    "student_id": student_id,
                    "correct": bool(is_correct),
                    "feature_vectors": _clean_cell(raw_row.get("Feature Vectors")),
                    "other_strategy": _clean_cell(raw_row.get("Other Strategy")),
                    "time_spent": _clean_cell(raw_row.get("Unnamed: 6")),
                    "instructor_note": _clean_cell(raw_row.get("Unnamed: 7")),
                    "subset_profile": subset_profile,
                },
            )
        )
    if not rows:
        raise ValueError("ausem parser produced no rows; check source file contents")
    return rows


def _parse_ausem_snapshot(
    ctx: ParserContext,
    *,
    selected_problem_keys: tuple[str, ...],
    subset_profile: str,
) -> Iterable["SnapshotRow"]:
    raw_rows = _load_problem_rows(ctx, selected_problem_keys=selected_problem_keys)
    return _build_snapshot_rows(raw_rows, subset_profile=subset_profile)


def parse_ausem_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Default parser profile spanning all four AuSeM problem files."""
    return _parse_ausem_snapshot(
        ctx,
        selected_problem_keys=_PROBLEM_KEY_ORDER,
        subset_profile="all",
    )


def parse_ausem_problem1_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Parser profile for AuSeM problem1 only."""
    return _parse_ausem_snapshot(
        ctx,
        selected_problem_keys=("problem1",),
        subset_profile="problem=1",
    )


def parse_ausem_problem2_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Parser profile for AuSeM problem2 only."""
    return _parse_ausem_snapshot(
        ctx,
        selected_problem_keys=("problem2",),
        subset_profile="problem=2",
    )


def parse_ausem_problem3_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Parser profile for AuSeM problem3 only."""
    return _parse_ausem_snapshot(
        ctx,
        selected_problem_keys=("problem3",),
        subset_profile="problem=3",
    )


def parse_ausem_problem4_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Parser profile for AuSeM problem4 only."""
    return _parse_ausem_snapshot(
        ctx,
        selected_problem_keys=("problem4",),
        subset_profile="problem=4",
    )
