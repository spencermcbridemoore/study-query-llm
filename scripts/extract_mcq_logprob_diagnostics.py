#!/usr/bin/env python3
"""Extract per-model MCQ logprob diagnostics for one request group."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import ProvenancedRun
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.services.artifact_service import ArtifactService

TARGET_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "qwen/qwen3.6-max-preview",
    "mancer/weaver",
)

CSV_COLUMNS: tuple[str, ...] = (
    "question_idx",
    "item_id",
    "temp_bank_id",
    "permutation_idx",
    "permutation_sigma",
    "format_idx",
    "prompt_template_version",
    "predicted_letter",
    "correct_letter",
    "is_correct",
    "error",
    "rate_limit_retry_count",
    "rate_limit_total_wait_ms",
    "first_token_top_logprobs_json",
    "candidate_logprobs_json",
)


@dataclass(frozen=True)
class ModelExtraction:
    model: str
    slug: str
    run_id: int
    strategy: str
    frame: pd.DataFrame
    csv_path: Path
    stats_path: Path
    samples_path: Path


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def _slug(model: str) -> str:
    return re.sub(r"[\/\.\-]+", "_", str(model)).strip("_")


def _display_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float) and pd.isna(value):
        return "null"
    return str(value)


def _display_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(float(numerator) / float(denominator) * 100.0):.2f}%"


def _parse_verifier_accuracy(
    *,
    request_group_id: int,
) -> dict[str, str]:
    report_path = REPO_ROOT / f"scratch/mcq_logprob_run_{request_group_id}_verification.md"
    if not report_path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in report_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 6:
            continue
        model_cell = cells[0]
        if not (model_cell.startswith("`") and model_cell.endswith("`")):
            continue
        model = model_cell[1:-1]
        accuracy = cells[5]
        out[model] = accuracy
    return out


def _ensure_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out


def _format_sigma(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float) and pd.isna(value):
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return "[" + ", ".join(str(item) for item in converted) + "]"
    if isinstance(value, str):
        return value
    return str(value)


def _letter_distribution(series: pd.Series) -> pd.DataFrame:
    total = int(len(series))
    letters = ["A", "B", "C", "D", "E"]
    include_null = bool(series.isna().any() or (series.astype(str).str.strip() == "").any())
    labels = letters + (["null"] if include_null else [])
    rows: list[dict[str, Any]] = []
    cleaned = series.astype(object)
    for label in labels:
        if label == "null":
            count = int(cleaned.isna().sum() + (cleaned.astype(str).str.strip() == "").sum())
        else:
            count = int((cleaned == label).sum())
        rows.append({"value": label, "count": count, "percent": _display_pct(count, total)})
    return pd.DataFrame(rows)


def _error_distribution(series: pd.Series) -> pd.DataFrame:
    total = int(len(series))
    counts = series.value_counts(dropna=False)
    rows: list[dict[str, Any]] = []
    for key, count in counts.items():
        label = _display_value(key)
        rows.append(
            {
                "error": label,
                "count": int(count),
                "percent": _display_pct(int(count), total),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table_from_frame(frame: pd.DataFrame) -> list[str]:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        cells = [_display_value(row[col]) for col in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _cross_tab(frame: pd.DataFrame) -> pd.DataFrame:
    pred = frame["predicted_letter"].astype(object).where(frame["predicted_letter"].notna(), "null")
    corr = frame["correct_letter"].astype(object).where(frame["correct_letter"].notna(), "null")
    letters = ["A", "B", "C", "D", "E"]
    include_null = bool((pred == "null").any() or (corr == "null").any())
    categories = letters + (["null"] if include_null else [])
    pred_cat = pd.Categorical(pred, categories=categories, ordered=True)
    corr_cat = pd.Categorical(corr, categories=categories, ordered=True)
    tab = pd.crosstab(pred_cat, corr_cat, dropna=False, margins=True, margins_name="All")
    tab.index = [str(item) for item in tab.index]
    return tab.reset_index(names=["predicted_letter"])


def _parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_first_token_lines(parsed: Any) -> list[str]:
    lines: list[str] = []
    if parsed is None:
        return [
            "- first_token_top_logprobs_json (parsed): empty/null payload",
        ]
    if parsed == []:
        return [
            "- first_token_top_logprobs_json (parsed): empty list []",
        ]
    if not isinstance(parsed, list):
        return [
            f"- first_token_top_logprobs_json (parsed): unsupported type {type(parsed).__name__}"
        ]
    lines.append("- first_token_top_logprobs_json (parsed):")
    for entry in parsed:
        if isinstance(entry, dict):
            lines.append(
                f"  - token={_display_value(entry.get('token'))}, "
                f"logprob={_display_value(entry.get('logprob'))}, "
                f"token_id={_display_value(entry.get('token_id'))}"
            )
        else:
            lines.append(f"  - entry={_display_value(entry)}")
    return lines


def _parse_candidate_lines(parsed: Any) -> list[str]:
    lines = ["- candidate_logprobs_json (parsed):"]
    if parsed is None:
        lines.append("  - explicit_empty: null payload")
        for letter in ["A", "B", "C", "D", "E"]:
            lines.append(f"  - {letter}=null")
        return lines
    if parsed == {}:
        lines.append("  - explicit_empty: empty object {}")
        for letter in ["A", "B", "C", "D", "E"]:
            lines.append(f"  - {letter}=null")
        return lines
    if not isinstance(parsed, dict):
        lines.append(f"  - explicit_empty: unsupported type {type(parsed).__name__}")
        for letter in ["A", "B", "C", "D", "E"]:
            lines.append(f"  - {letter}=null")
        return lines
    for letter in ["A", "B", "C", "D", "E"]:
        value = parsed.get(letter)
        lines.append(f"  - {letter}={_display_value(value)}")
    return lines


def _select_samples(model: str, frame: pd.DataFrame) -> pd.DataFrame:
    sorted_frame = frame.sort_values(by=["question_idx", "permutation_idx"], kind="stable")
    if model == "openai/gpt-4o-mini":
        cat1 = sorted_frame[
            (sorted_frame["predicted_letter"] == "A")
            & (sorted_frame["correct_letter"] != "A")
        ].head(5)
        cat2 = sorted_frame[sorted_frame["predicted_letter"] != "A"].head(5)
        return pd.concat([cat1, cat2], axis=0, ignore_index=True)
    if model == "qwen/qwen3.6-max-preview":
        cat1 = sorted_frame[sorted_frame["error"] == "http_400"].head(5)
        cat2 = sorted_frame[sorted_frame["error"] == "typeerror"].head(5)
        return pd.concat([cat1, cat2], axis=0, ignore_index=True)
    if model == "mancer/weaver":
        null_pred = sorted_frame[sorted_frame["predicted_letter"].isna()]
        selected_rows = []
        for _, group in null_pred.groupby("question_idx", sort=True):
            selected_rows.append(group.iloc[0])
            if len(selected_rows) >= 5:
                break
        if not selected_rows:
            return null_pred.head(0)
        return pd.DataFrame(selected_rows).reset_index(drop=True)
    return sorted_frame.head(10)


def _write_samples_markdown(
    *,
    model: str,
    frame: pd.DataFrame,
    path: Path,
) -> None:
    sampled = _select_samples(model, frame)
    lines: list[str] = [
        f"# Raw payload samples — {model}",
        "",
    ]
    if sampled.empty:
        lines.append("No rows matched sampling criteria.")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        lines.append(f"### Row {idx}")
        lines.append(f"- question_idx: {_display_value(row.get('question_idx'))}")
        lines.append(f"- permutation_idx: {_display_value(row.get('permutation_idx'))}")
        lines.append(f"- correct_letter: {_display_value(row.get('correct_letter'))}")
        lines.append(f"- predicted_letter: {_display_value(row.get('predicted_letter'))}")
        lines.append(f"- is_correct: {_display_value(row.get('is_correct'))}")
        lines.append(f"- error: {_display_value(row.get('error'))}")
        lines.append(f"- rate_limit_retry_count: {_display_value(row.get('rate_limit_retry_count'))}")
        first_token_parsed = _parse_json_field(row.get("first_token_top_logprobs_json"))
        lines.extend(_parse_first_token_lines(first_token_parsed))
        candidate_parsed = _parse_json_field(row.get("candidate_logprobs_json"))
        lines.extend(_parse_candidate_lines(candidate_parsed))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_stats_markdown(
    *,
    model: str,
    run_id: int,
    strategy: str,
    frame: pd.DataFrame,
    verifier_accuracy: Optional[str],
    path: Path,
) -> None:
    total_rows = int(len(frame))
    error_non_null = int(frame["error"].notna().sum())
    error_rate = float(error_non_null) / float(total_rows) if total_rows else 0.0

    pred_notna = frame["predicted_letter"].notna()
    denom = int(pred_notna.sum())
    numer = int((frame.loc[pred_notna, "predicted_letter"] == frame.loc[pred_notna, "correct_letter"]).sum())
    manual_accuracy = _display_pct(numer, denom)

    lines: list[str] = []
    lines.append(f"# Aggregate stats — {model}")
    lines.append("")
    lines.append("## Header")
    lines.append("")
    lines.append(f"- model: `{model}`")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- strategy: `{strategy}`")
    lines.append(f"- parquet_row_count: `{total_rows}`")
    lines.append(f"- error_rate: `{error_rate * 100.0:.2f}%`")
    lines.append("")

    lines.append("## Distribution of correct_letter")
    lines.append("")
    lines.extend(_markdown_table_from_frame(_letter_distribution(frame["correct_letter"])))
    lines.append("")

    lines.append("## Distribution of predicted_letter")
    lines.append("")
    lines.extend(_markdown_table_from_frame(_letter_distribution(frame["predicted_letter"])))
    lines.append("")

    lines.append("## Cross-tab predicted_letter × correct_letter")
    lines.append("")
    lines.extend(_markdown_table_from_frame(_cross_tab(frame)))
    lines.append("")

    lines.append("## Manual accuracy")
    lines.append("")
    lines.append(f"- numerator `(predicted_letter == correct_letter).sum()`: `{numer}`")
    lines.append(f"- denominator `predicted_letter.notna().sum()`: `{denom}`")
    lines.append(f"- manual_accuracy: `{manual_accuracy}`")
    lines.append("")

    lines.append("## Verifier-reported accuracy comparison")
    lines.append("")
    comparison = pd.DataFrame(
        [
            {
                "manual_accuracy": manual_accuracy,
                "verifier_reported_accuracy": verifier_accuracy or "n/a",
            }
        ]
    )
    lines.extend(_markdown_table_from_frame(comparison))
    lines.append("")

    lines.append("## Distribution of error values")
    lines.append("")
    lines.extend(_markdown_table_from_frame(_error_distribution(frame["error"])))
    lines.append("")

    lines.append("## Per-permutation correct_letter check (question_idx=0)")
    lines.append("")
    q0 = frame[frame["question_idx"] == 0].sort_values(by=["permutation_idx"], kind="stable")
    if q0.empty:
        lines.append("No rows for `question_idx=0`.")
    else:
        q0_rows = q0[["permutation_idx", "permutation_sigma", "correct_letter"]].copy()
        q0_rows["permutation_sigma"] = q0_rows["permutation_sigma"].apply(_format_sigma)
        lines.extend(_markdown_table_from_frame(q0_rows))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_model_run(
    *,
    runs: list[ProvenancedRun],
    model: str,
) -> ProvenancedRun:
    matches = [run for run in runs if str((run.metadata_json or {}).get("model") or "") == model]
    if not matches:
        raise ValueError(f"model_not_found_in_request_group:{model}")
    if len(matches) != 1:
        ids = [int(run.id) for run in matches]
        raise ValueError(f"model_run_count_not_one:{model}:run_ids={ids}")
    return matches[0]


def main(argv: Optional[list[str]] = None) -> int:
    start = time.perf_counter()
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract MCQ logprob diagnostic slices for selected models."
    )
    parser.add_argument(
        "--request-group-id",
        required=True,
        type=int,
        help="Request group id with MCQ logprob execution rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/diagnostics/",
        help="Output directory for diagnostics files.",
    )
    args = parser.parse_args(argv)

    request_group_id = int(args.request_group_id)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_url = _resolve_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(db_url),
    )

    verifier_accuracy = _parse_verifier_accuracy(request_group_id=request_group_id)

    outputs: list[Path] = []
    extractions: list[ModelExtraction] = []

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifact_service = ArtifactService(repository=repo)

        runs = (
            session.query(ProvenancedRun)
            .filter(ProvenancedRun.request_group_id == request_group_id)
            .order_by(ProvenancedRun.id.asc())
            .all()
        )

        for model in TARGET_MODELS:
            run = _extract_model_run(runs=runs, model=model)
            metadata = dict(run.metadata_json or {})
            strategy = str(metadata.get("permutation_strategy") or "")
            slug = _slug(model)
            result_ref = str(run.result_ref or "")
            if not result_ref:
                raise ValueError(f"missing_result_ref_for_model:{model}:run_id={run.id}")

            parquet_bytes = artifact_service.storage.read_from_uri(result_ref)
            frame = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
            frame = _ensure_columns(frame, CSV_COLUMNS)

            csv_frame = frame.loc[:, list(CSV_COLUMNS)].copy()
            csv_frame["permutation_sigma"] = csv_frame["permutation_sigma"].apply(_format_sigma)
            csv_path = output_dir / f"{request_group_id}_{slug}_rows.csv"
            csv_frame.to_csv(csv_path, index=False, encoding="utf-8")

            stats_path = output_dir / f"{request_group_id}_{slug}_stats.md"
            _write_stats_markdown(
                model=model,
                run_id=int(run.id),
                strategy=strategy,
                frame=frame,
                verifier_accuracy=verifier_accuracy.get(model),
                path=stats_path,
            )

            samples_path = output_dir / f"{request_group_id}_{slug}_samples.md"
            _write_samples_markdown(
                model=model,
                frame=frame,
                path=samples_path,
            )

            outputs.extend([csv_path, stats_path, samples_path])
            extractions.append(
                ModelExtraction(
                    model=model,
                    slug=slug,
                    run_id=int(run.id),
                    strategy=strategy,
                    frame=frame,
                    csv_path=csv_path,
                    stats_path=stats_path,
                    samples_path=samples_path,
                )
            )

    elapsed = time.perf_counter() - start
    print("row_counts:")
    for item in extractions:
        print(f"  {item.model}: {len(item.frame)}")
    print("output_files:")
    for path in outputs:
        print(f"  {path}")
    print(f"total_wall_clock_seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
