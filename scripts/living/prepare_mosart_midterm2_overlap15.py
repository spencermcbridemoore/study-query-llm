#!/usr/bin/env python3
"""Reshape MOSART overlap-15 questions into midterm2 MCQ logprob schema.

Reads MOSART ``questions.csv``, filters to the 15 same-test misconception-overlap
targets (excluding ``9_12_astronomy_q16``), maps columns to midterm2 shape, emits
an ID sidecar joined to ``question_misconception_links.csv``, and writes cl100k
prompt-token estimates for v1 / v2_chat_system formats.

Example::

    python scripts/living/prepare_mosart_midterm2_overlap15.py

Default MOSART root:
``c:/Users/spenc/OneDrive/Documents/Claude/Projects/MOSART/mosart_tables/``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_MOSART_TABLES = Path(
    r"c:/Users/spenc/OneDrive/Documents/Claude/Projects/MOSART/mosart_tables"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "mosart"

EXCLUDED_QUESTION_ID = "9_12_astronomy_q16"

TARGET_QUESTION_IDS: tuple[str, ...] = (
    "5_8_astronomy_q02",
    "5_8_astronomy_q04",
    "5_8_astronomy_q06",
    "5_8_astronomy_q07",
    "5_8_astronomy_q10",
    "9_12_astronomy_q08",
    "9_12_astronomy_q11",
    "9_12_astronomy_q12",
    "9_12_astronomy_q15",
    "k_4_astronomy_q01",
    "k_4_astronomy_q04",
    "k_4_astronomy_q08",
    "k_4_astronomy_q13",
    "ls_mastery_q05",
    "ls_mastery_q06",
)

MIDTERM2_COLUMNS: tuple[str, ...] = (
    "ItemID",
    "ItemType",
    "ProblemBody",
    "OptionA",
    "OptionB",
    "OptionC",
    "OptionD",
    "OptionE",
    "CorrectLetter",
    "temp_bank_id",
)

SIDECAR_COLUMNS: tuple[str, ...] = (
    "ItemID",
    "temp_bank_id",
    "question_id",
    "document_id",
    "misconception_id",
)

DOCUMENT_ID_TO_TEMP_BANK: dict[str, int] = {
    "5_8_astronomy": 1,
    "9_12_astronomy": 2,
    "k_4_astronomy": 3,
    "ls_mastery": 4,
}

OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E")

PROMPT_V2_SYSTEM_MESSAGE = (
    "Reply with exactly one letter (A, B, C, D, or E). "
    "Do not explain. Do not output any other text."
)

_FORMAT_SUFFIX_V1 = "Answer:"

_MATCH_CONFIDENCE_RANK: dict[str, int] = {
    "almost_certain": 4,
    "high": 3,
    "likely": 2,
    "medium": 1,
    "possible": 0,
    "low": -1,
}


def _normalize_letter(value: Any) -> str:
    letter = str(value or "").strip().upper()
    if letter not in OPTION_LABELS:
        raise ValueError(f"invalid_correct_letter:{letter!r}")
    return letter


def _validate_required_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    context: str,
) -> None:
    observed = {str(col) for col in frame.columns}
    expected = set(required)
    if observed >= expected:
        return
    missing = sorted(expected - observed)
    raise ValueError(
        f"{context}_missing_columns:expected={sorted(expected)},observed={sorted(observed)},missing={missing}"
    )


def _match_confidence_sort_key(value: Any) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return float("-inf")
    if text in _MATCH_CONFIDENCE_RANK:
        return float(_MATCH_CONFIDENCE_RANK[text])
    try:
        return float(text)
    except ValueError:
        return float("-inf")


def load_misconception_links(links_path: Path) -> pd.DataFrame:
    if not links_path.is_file():
        raise FileNotFoundError(str(links_path.resolve()))
    frame = pd.read_csv(links_path, encoding="utf-8")
    _validate_required_columns(
        frame,
        ("question_id", "misconception_id", "match_confidence"),
        context="misconception_links",
    )
    return frame


def collapse_misconception_links(
    links: pd.DataFrame,
    *,
    question_ids: Sequence[str],
) -> pd.DataFrame:
    subset = links[links["question_id"].astype(str).isin(set(question_ids))].copy()
    if subset.empty and question_ids:
        missing = sorted(set(question_ids))
        raise ValueError(f"sidecar_join_empty:question_ids={missing}")

    collapsed_rows: list[dict[str, Any]] = []
    for qid in question_ids:
        rows = subset[subset["question_id"].astype(str) == str(qid)]
        count = int(len(rows))
        if count == 0:
            raise ValueError(f"sidecar_join_cardinality:{qid}:0")
        if count == 1:
            collapsed_rows.append(dict(rows.iloc[0]))
            continue

        ranked = rows.copy()
        ranked["_confidence_rank"] = ranked["match_confidence"].map(_match_confidence_sort_key)
        ranked = ranked.sort_values(
            by=["_confidence_rank", "misconception_id", "option_letter"],
            ascending=[False, True, True],
            kind="stable",
        )
        top = ranked.iloc[0]
        top_rank = float(top["_confidence_rank"])
        tied = ranked[ranked["_confidence_rank"] == top_rank]
        if len(tied) > 1:
            winner = tied.sort_values(
                by=["misconception_id", "option_letter"],
                ascending=[True, True],
                kind="stable",
            ).iloc[0]
            collapsed_rows.append(dict(winner))
        else:
            collapsed_rows.append(dict(top))

    out = pd.DataFrame(collapsed_rows)
    counts = out.groupby("question_id").size()
    bad = {str(qid): int(count) for qid, count in counts.items() if int(count) != 1}
    if bad:
        first_qid, first_count = next(iter(bad.items()))
        raise ValueError(f"sidecar_join_cardinality:{first_qid}:{first_count}")
    return out


def map_questions_to_midterm2(questions: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(
        questions,
        (
            "question_id",
            "document_id",
            "stem",
            "opt_a_text",
            "opt_b_text",
            "opt_c_text",
            "opt_d_text",
            "opt_e_text",
            "correct_answer",
        ),
        context="questions",
    )

    filtered = questions[
        questions["question_id"].astype(str).isin(set(TARGET_QUESTION_IDS))
    ].copy()
    if EXCLUDED_QUESTION_ID in set(filtered["question_id"].astype(str)):
        raise ValueError(f"excluded_question_present:{EXCLUDED_QUESTION_ID}")

    present = set(filtered["question_id"].astype(str))
    missing = sorted(set(TARGET_QUESTION_IDS) - present)
    if missing:
        raise ValueError(f"target_questions_missing:{missing}")

    ordered = filtered.assign(
        _sort_key=filtered["question_id"].astype(str).map(
            {qid: idx for idx, qid in enumerate(TARGET_QUESTION_IDS)}
        )
    ).sort_values("_sort_key", kind="stable")

    rows: list[dict[str, Any]] = []
    for item_idx, (_, row) in enumerate(ordered.iterrows(), start=1):
        document_id = str(row["document_id"]).strip()
        if document_id not in DOCUMENT_ID_TO_TEMP_BANK:
            raise ValueError(f"unknown_document_id:{document_id}")
        rows.append(
            {
                "ItemID": int(item_idx),
                "ItemType": "choice",
                "ProblemBody": str(row["stem"]),
                "OptionA": str(row["opt_a_text"]),
                "OptionB": str(row["opt_b_text"]),
                "OptionC": str(row["opt_c_text"]),
                "OptionD": str(row["opt_d_text"]),
                "OptionE": str(row["opt_e_text"]),
                "CorrectLetter": _normalize_letter(row["correct_answer"]),
                "temp_bank_id": int(DOCUMENT_ID_TO_TEMP_BANK[document_id]),
                "question_id": str(row["question_id"]),
                "document_id": document_id,
            }
        )
    return pd.DataFrame(rows)


def build_id_sidecar(
    mapped: pd.DataFrame,
    collapsed_links: pd.DataFrame,
) -> pd.DataFrame:
    link_by_qid = {
        str(row["question_id"]): row for _, row in collapsed_links.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in mapped.iterrows():
        qid = str(row["question_id"])
        link = link_by_qid.get(qid)
        if link is None:
            raise ValueError(f"sidecar_join_cardinality:{qid}:0")
        rows.append(
            {
                "ItemID": int(row["ItemID"]),
                "temp_bank_id": int(row["temp_bank_id"]),
                "question_id": qid,
                "document_id": str(row["document_id"]),
                "misconception_id": str(link["misconception_id"]),
            }
        )
    out = pd.DataFrame(rows, columns=list(SIDECAR_COLUMNS))
    if len(out) != len(TARGET_QUESTION_IDS):
        raise ValueError(f"sidecar_row_count_mismatch:{len(out)}")
    return out


def _format_prompt_v1(problem_body: str, options: Mapping[str, str]) -> str:
    options_block = "\n".join(
        f"{label}. {options[label]}" for label in OPTION_LABELS
    )
    return f"{problem_body}\n\n{options_block}\n\n{_FORMAT_SUFFIX_V1}"


def _format_prompt_v2_chat_system(problem_body: str, options: Mapping[str, str]) -> str:
    # Token estimate uses user prompt only; system message counted separately.
    return _format_prompt_v1(problem_body, options)


def estimate_prompt_tokens_cl100k(mapped: pd.DataFrame) -> dict[str, Any]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("tiktoken_required_for_token_estimates") from exc

    encoding = tiktoken.get_encoding("cl100k_base")
    v1_counts: list[int] = []
    v2_user_counts: list[int] = []
    v2_total_counts: list[int] = []
    per_question: list[dict[str, Any]] = []

    system_tokens = len(encoding.encode(PROMPT_V2_SYSTEM_MESSAGE))

    for _, row in mapped.iterrows():
        options = {
            label: str(row[f"Option{label}"]) for label in OPTION_LABELS
        }
        body = str(row["ProblemBody"])
        v1_prompt = _format_prompt_v1(body, options)
        v2_user_prompt = _format_prompt_v2_chat_system(body, options)
        v1_n = len(encoding.encode(v1_prompt))
        v2_user_n = len(encoding.encode(v2_user_prompt))
        v2_total_n = system_tokens + v2_user_n
        v1_counts.append(v1_n)
        v2_user_counts.append(v2_user_n)
        v2_total_counts.append(v2_total_n)
        per_question.append(
            {
                "ItemID": int(row["ItemID"]),
                "question_id": str(row["question_id"]),
                "tokens_v1": v1_n,
                "tokens_v2_user": v2_user_n,
                "tokens_v2_with_system": v2_total_n,
            }
        )

    def _stats(values: Sequence[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": int(min(values)),
            "max": int(max(values)),
            "mean": float(sum(values)) / float(len(values)),
        }

    return {
        "per_question": per_question,
        "v1": _stats(v1_counts),
        "v2_user": _stats(v2_user_counts),
        "v2_with_system": _stats(v2_total_counts),
        "system_message_tokens": int(system_tokens),
    }


def format_token_estimate_markdown(estimates: Mapping[str, Any]) -> str:
    v1 = dict(estimates.get("v1") or {})
    v2 = dict(estimates.get("v2_with_system") or {})
    lines = [
        "# MOSART overlap15 prompt token estimates (cl100k_base)",
        "",
        f"- question_count: `{len(estimates.get('per_question') or [])}`",
        f"- system_message_tokens_v2: `{int(estimates.get('system_message_tokens') or 0)}`",
        "",
        "## Per-format means",
        "",
        f"- v1_mean: `{float(v1.get('mean', 0.0)):.2f}`",
        f"- v2_chat_system_mean: `{float(v2.get('mean', 0.0)):.2f}`",
        "",
        "## Ranges",
        "",
        f"- v1_min: `{int(v1.get('min', 0))}`",
        f"- v1_max: `{int(v1.get('max', 0))}`",
        f"- v2_min: `{int(v2.get('min', 0))}`",
        f"- v2_max: `{int(v2.get('max', 0))}`",
        "",
        "## Per-question",
        "",
        "| ItemID | question_id | tokens_v1 | tokens_v2_with_system |",
        "|---:|---|---:|---:|",
    ]
    for row in estimates.get("per_question") or []:
        lines.append(
            f"| {int(row['ItemID'])} | `{row['question_id']}` | "
            f"{int(row['tokens_v1'])} | {int(row['tokens_v2_with_system'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def prepare_overlap15(
    *,
    mosart_tables_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    questions_path = mosart_tables_dir / "questions.csv"
    links_path = mosart_tables_dir / "question_misconception_links.csv"
    if not questions_path.is_file():
        raise FileNotFoundError(str(questions_path.resolve()))

    questions = pd.read_csv(questions_path, encoding="utf-8")
    mapped = map_questions_to_midterm2(questions)
    midterm2_frame = mapped[list(MIDTERM2_COLUMNS)].copy()

    links = load_misconception_links(links_path)
    collapsed = collapse_misconception_links(
        links,
        question_ids=[str(row["question_id"]) for _, row in mapped.iterrows()],
    )
    sidecar = build_id_sidecar(mapped, collapsed)
    estimates = estimate_prompt_tokens_cl100k(mapped)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mosart_midterm2_overlap15.csv"
    sidecar_path = output_dir / "mosart_midterm2_overlap15_id_map.csv"
    token_md_path = output_dir / "mosart_midterm2_overlap15_token_estimate.md"
    token_json_path = output_dir / "mosart_midterm2_overlap15_token_estimate.json"

    midterm2_frame.to_csv(csv_path, index=False, encoding="utf-8")
    sidecar.to_csv(sidecar_path, index=False, encoding="utf-8")
    token_md_path.write_text(format_token_estimate_markdown(estimates), encoding="utf-8")
    token_json_path.write_text(
        json.dumps(
            {
                "v1_mean": float(estimates["v1"]["mean"]),
                "v2_chat_system_mean": float(estimates["v2_with_system"]["mean"]),
                "system_message_tokens": int(estimates["system_message_tokens"]),
                "per_question": estimates["per_question"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "csv": csv_path,
        "sidecar": sidecar_path,
        "token_estimate_md": token_md_path,
        "token_estimate_json": token_json_path,
    }


def parse_token_estimate_json(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("token_estimate_json_not_object")
    return {
        "v1": float(payload.get("v1_mean") or 0.0),
        "v2_chat_system": float(payload.get("v2_chat_system_mean") or 0.0),
    }


def parse_token_estimate_markdown(text: str) -> dict[str, float]:
    v1_match = re.search(r"^- v1_mean: `([0-9.]+)`", text, flags=re.MULTILINE)
    v2_match = re.search(
        r"^- v2_chat_system_mean: `([0-9.]+)`",
        text,
        flags=re.MULTILINE,
    )
    if v1_match is None or v2_match is None:
        raise ValueError("token_estimate_markdown_missing_means")
    return {
        "v1": float(v1_match.group(1)),
        "v2_chat_system": float(v2_match.group(1)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosart-tables-dir",
        type=Path,
        default=DEFAULT_MOSART_TABLES,
        help="Directory containing MOSART questions.csv and question_misconception_links.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for reshaped CSV, sidecar, and token estimates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = prepare_overlap15(
        mosart_tables_dir=Path(args.mosart_tables_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    print(f"wrote_csv={outputs['csv']}")
    print(f"wrote_sidecar={outputs['sidecar']}")
    print(f"wrote_token_estimate_md={outputs['token_estimate_md']}")
    print(f"wrote_token_estimate_json={outputs['token_estimate_json']}")
    print(f"row_count={len(TARGET_QUESTION_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
