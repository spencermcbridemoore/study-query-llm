"""
SemEval-2013 Task 7 — Student Response Analysis, five-way labels (mirror).

Uses public GitHub mirror ashudeep/Student-Response-Analysis (not the official
LDC distribution). Pinned commit for reproducible raw.githubusercontent.com URLs.

Gold files and ``answers.csv`` are tab-delimited and include duplicated answer
rows per `id`; parse-time logic selects deterministic text + label per id.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow

SEMEVAL2013_SRA_5WAY_SLUG = "semeval2013_sra_5way"
GITHUB_ORG = "ashudeep"
GITHUB_REPO = "Student-Response-Analysis"
# Tip of master at integration time (reproducible mirror).
PINNED_GIT_REF = "1d6d30b265e6038fd6f6395d4cfd6686aef4b97f"
SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_ID = "semeval2013_sra_5way.default"
SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION = "v1"

_PREFIX = "semevalFormatProcessing-5way"
_GOLD_FILES: tuple[str, ...] = (
    "trainingGold.txt",
    "trainingGold-partial.txt",
    "testGold-UA.txt",
    "testGold-UQ.txt",
    "partialEntailmentGold.txt",
)
# Same directory as gold files in the mirror; keeps acquisition self-contained.
_5WAY_EXTRA_FILES: tuple[str, ...] = ("answers.csv",)
_LABEL_ORDER: tuple[str, ...] = (
    "correct",
    "contradictory",
    "partially_correct_incomplete",
    "irrelevant",
    "non_domain",
)
_LABEL_TO_INT: dict[str, int] = {
    label: idx for idx, label in enumerate(_LABEL_ORDER)
}


def _raw_url(repo_relative_path: str) -> str:
    rel = repo_relative_path.lstrip("/")
    return (
        f"https://raw.githubusercontent.com/{GITHUB_ORG}/"
        f"{GITHUB_REPO}/{PINNED_GIT_REF}/{rel}"
    )


def semeval2013_sra_5way_file_specs() -> List[FileFetchSpec]:
    specs: List[FileFetchSpec] = []
    for name in _GOLD_FILES + _5WAY_EXTRA_FILES:
        rel = f"{_PREFIX}/{name}"
        specs.append(FileFetchSpec(relative_path=rel, url=_raw_url(rel)))
    specs.append(FileFetchSpec(relative_path="README.md", url=_raw_url("README.md")))
    return specs


def semeval2013_sra_5way_source_metadata() -> Dict[str, Any]:
    return {
        "kind": "github_raw",
        "organization": GITHUB_ORG,
        "repository": GITHUB_REPO,
        "git_ref": PINNED_GIT_REF,
        "description": "SemEval-2013 Task 7 SRA five-way gold files + answers.csv (community mirror)",
        "note": "Official corpus may require LDC; this pin is the ashudeep/Student-Response-Analysis mirror.",
    }


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def _load_tsv_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(text.splitlines(), delimiter="\t")
    return [list(row) for row in reader if any(str(cell).strip() for cell in row)]


def _load_answers_by_id(base_dir: Path) -> dict[str, dict[str, Any]]:
    answers_path = base_dir / _PREFIX / "answers.csv"
    if not answers_path.is_file():
        raise ValueError(f"semeval parser expected file missing: {answers_path}")
    rows = _load_tsv_rows(answers_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if len(row) < 3:
            continue
        answer_id = _clean_cell(row[0])
        if not answer_id:
            continue
        question = _clean_cell(row[1]) if len(row) >= 2 else ""
        reference_answer = _clean_cell(row[2]) if len(row) >= 3 else ""
        student_answer = _clean_cell(row[3]) if len(row) >= 4 else ""
        entry = by_id.setdefault(
            answer_id,
            {
                "question": question,
                "reference_answer": reference_answer,
                "student_answers": [],
                "row_count": 0,
            },
        )
        if not entry["question"] and question:
            entry["question"] = question
        if not entry["reference_answer"] and reference_answer:
            entry["reference_answer"] = reference_answer
        if student_answer:
            entry["student_answers"].append(student_answer)
        entry["row_count"] += 1
    return by_id


def _resolve_answer_text(answer_entry: dict[str, Any], *, answer_id: str) -> str:
    student_answers = list(answer_entry.get("student_answers") or [])
    if student_answers:
        return _clean_cell(student_answers[0])
    reference_answer = _clean_cell(answer_entry.get("reference_answer"))
    if reference_answer:
        return reference_answer
    raise ValueError(
        f"semeval parser id={answer_id!r} has no non-empty student/reference answer text"
    )


def _iter_gold_rows(base_dir: Path) -> Iterable[dict[str, Any]]:
    for gold_file in _GOLD_FILES:
        path = base_dir / _PREFIX / gold_file
        if not path.is_file():
            raise ValueError(f"semeval parser expected file missing: {path}")
        rows = _load_tsv_rows(path)
        for idx, row in enumerate(rows):
            if idx == 0 and row and _clean_cell(row[0]).lower() == "id":
                continue
            if len(row) < 6:
                continue
            yield {
                "id": _clean_cell(row[0]),
                "qid": _clean_cell(row[1]),
                "test_set": _clean_cell(row[2]),
                "module": _clean_cell(row[3]),
                "count": _clean_cell(row[4]),
                "accuracy": _clean_cell(row[5]).lower(),
                "gold_file": gold_file,
            }


def parse_semeval2013_sra_5way_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    from study_query_llm.pipeline.types import SnapshotRow

    answer_lookup = _load_answers_by_id(ctx.artifact_dir_local)
    rows: list[SnapshotRow] = []
    seen_gold: dict[str, tuple[str, str, str, str]] = {}
    for gold in _iter_gold_rows(ctx.artifact_dir_local):
        answer_id = str(gold["id"])
        if not answer_id:
            continue
        label_name = str(gold["accuracy"])
        qid = str(gold["qid"])
        module = str(gold["module"])
        test_set = str(gold["test_set"])
        prior = seen_gold.get(answer_id)
        if prior is not None:
            if prior != (label_name, qid, module, test_set):
                raise ValueError(
                    f"semeval parser found conflicting gold rows for id={answer_id!r}"
                )
            continue
        if answer_id not in answer_lookup:
            raise ValueError(
                f"semeval parser id={answer_id!r} present in gold files but missing from answers.csv"
            )
        if label_name not in _LABEL_TO_INT:
            raise ValueError(
                f"semeval parser id={answer_id!r} has unknown label {label_name!r}; "
                f"expected one of {sorted(_LABEL_TO_INT.keys())}"
            )
        answer_entry = answer_lookup[answer_id]
        text = _resolve_answer_text(answer_entry, answer_id=answer_id)
        count_raw = str(gold["count"])
        rows.append(
            SnapshotRow(
                position=len(rows),
                source_id=answer_id,
                text=text,
                label=_LABEL_TO_INT[label_name],
                label_name=label_name,
                extra={
                    "qid": qid,
                    "module": module,
                    "test_set": test_set or None,
                    "gold_file": str(gold["gold_file"]),
                    "gold_count": int(count_raw) if count_raw.isdigit() else count_raw,
                    "question": str(answer_entry.get("question") or ""),
                    "reference_answer": str(answer_entry.get("reference_answer") or ""),
                    "answer_row_count": int(answer_entry.get("row_count") or 0),
                    "student_answer_count": len(
                        list(answer_entry.get("student_answers") or [])
                    ),
                },
            )
        )
        seen_gold[answer_id] = (label_name, qid, module, test_set)
    if not rows:
        raise ValueError("semeval parser produced no rows; check source file contents")
    return rows
