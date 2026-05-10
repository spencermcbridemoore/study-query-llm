#!/usr/bin/env python3
"""Extract full MCQ logprob diagnostics for all model runs in one request group."""

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

TARGET_SAMPLE_QUESTION_IDXS: tuple[int, ...] = (0, 15, 30)
ASCII_WS = " \t\r\n\f\v"
LETTER_SET = {"A", "B", "C", "D", "E"}


@dataclass(frozen=True)
class ModelRun:
    model: str
    run_id: int
    strategy: str
    frame: pd.DataFrame


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def _display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float) and pd.isna(value):
        return "null"
    return str(value)


def _slug(value: str) -> str:
    return re.sub(r"[\/\.\-]+", "_", str(value)).strip("_")


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
    return str(value)


def _letter_token_category(raw_token: Any) -> set[str]:
    categories: set[str] = set()
    if raw_token is None:
        return categories
    token = str(raw_token)
    stripped = token.strip(ASCII_WS)
    if not stripped:
        return categories

    if len(stripped) == 1 and stripped.upper() in LETTER_SET:
        categories.add("no_prefix")

    if len(stripped) == 2 and stripped[0] == "▁" and stripped[1].upper() in LETTER_SET:
        categories.add("sentencepiece_prefix")

    if len(stripped) == 2 and stripped[0] == "Ġ" and stripped[1].upper() in LETTER_SET:
        categories.add("bpe_prefix")

    if stripped and stripped[0] not in ("▁", "Ġ") and not stripped[0].isalpha():
        remainder = re.sub(r"^[^A-Za-z]+", "", stripped)
        if len(remainder) == 1 and remainder.upper() in LETTER_SET:
            categories.add("other_prefix")
    return categories


def _load_model_runs(
    *,
    request_group_id: int,
    repo: RawCallRepository,
    artifact_service: ArtifactService,
) -> list[ModelRun]:
    session = repo.session
    runs = (
        session.query(ProvenancedRun)
        .filter(ProvenancedRun.request_group_id == int(request_group_id))
        .order_by(ProvenancedRun.id.asc())
        .all()
    )
    out: list[ModelRun] = []
    for run in runs:
        meta = dict(run.metadata_json or {})
        model = str(meta.get("model") or "").strip()
        result_ref = str(run.result_ref or "").strip()
        strategy = str(meta.get("permutation_strategy") or "").strip()
        if not model or not result_ref:
            continue
        parquet_bytes = artifact_service.storage.read_from_uri(result_ref)
        frame = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
        out.append(
            ModelRun(
                model=model,
                run_id=int(run.id),
                strategy=strategy,
                frame=frame,
            )
        )
    return out


def _write_summary_csv(*, model_runs: list[ModelRun], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in sorted(model_runs, key=lambda x: x.model):
        frame = item.frame
        total_rows = int(len(frame))
        predicted = frame["predicted_letter"] if "predicted_letter" in frame.columns else pd.Series([None] * total_rows)
        correct = frame["correct_letter"] if "correct_letter" in frame.columns else pd.Series([None] * total_rows)
        error = frame["error"] if "error" in frame.columns else pd.Series([None] * total_rows)

        error_count = int(error.notna().sum())
        silent_null_count = int((error.isna() & predicted.isna()).sum())
        non_null_prediction_count = int(predicted.notna().sum())
        non_null_prediction_rate = (
            float(non_null_prediction_count) / float(total_rows) if total_rows > 0 else 0.0
        )

        if non_null_prediction_count > 0:
            numer = int((predicted[predicted.notna()] == correct[predicted.notna()]).sum())
            manual_accuracy: Any = float(numer) / float(non_null_prediction_count)
            pred_counts = predicted[predicted.notna()].value_counts(dropna=True)
            top_predicted_letter = str(pred_counts.index[0])
            top_predicted_letter_pct: Any = float(pred_counts.iloc[0]) / float(non_null_prediction_count)
            unique_predicted_letters_count = int(predicted[predicted.notna()].nunique(dropna=True))
        else:
            manual_accuracy = "n/a"
            top_predicted_letter = "n/a"
            top_predicted_letter_pct = "n/a"
            unique_predicted_letters_count = 0

        rows.append(
            {
                "model": item.model,
                "run_id": item.run_id,
                "strategy": item.strategy,
                "parquet_row_count": total_rows,
                "error_count": error_count,
                "silent_null_count": silent_null_count,
                "non_null_prediction_count": non_null_prediction_count,
                "non_null_prediction_rate": non_null_prediction_rate,
                "manual_accuracy": manual_accuracy,
                "top_predicted_letter": top_predicted_letter,
                "top_predicted_letter_pct": top_predicted_letter_pct,
                "unique_predicted_letters_count": unique_predicted_letters_count,
            }
        )

    out_df = pd.DataFrame(rows)
    cols = [
        "model",
        "run_id",
        "strategy",
        "parquet_row_count",
        "error_count",
        "silent_null_count",
        "non_null_prediction_count",
        "non_null_prediction_rate",
        "manual_accuracy",
        "top_predicted_letter",
        "top_predicted_letter_pct",
        "unique_predicted_letters_count",
    ]
    out_df = out_df[cols]
    output_path.write_text(out_df.to_csv(index=False), encoding="utf-8")


def _write_top_tokens_markdown(*, model_runs: list[ModelRun], output_path: Path) -> None:
    lines: list[str] = [f"# Top-token distribution — {len(model_runs)} models", ""]
    for item in sorted(model_runs, key=lambda x: x.model):
        frame = item.frame
        total_rows = int(len(frame))
        lines.append(f"## {item.model}")
        lines.append("")
        lines.append(
            f"- model: `{item.model}`, run_id: `{item.run_id}`, parquet_row_count: `{total_rows}`"
        )

        rank1_counts: dict[str, int] = {}
        captured_rows = 0
        row_presence = {
            "no_prefix": 0,
            "sentencepiece_prefix": 0,
            "bpe_prefix": 0,
            "other_prefix": 0,
        }

        payload_col = (
            frame["first_token_top_logprobs_json"]
            if "first_token_top_logprobs_json" in frame.columns
            else pd.Series([None] * total_rows)
        )

        for _, raw_payload in payload_col.items():
            parsed = _parse_json_field(raw_payload)
            if not isinstance(parsed, list) or len(parsed) == 0:
                continue
            captured_rows += 1

            first = parsed[0]
            rank1_token = ""
            if isinstance(first, dict):
                rank1_token = str(first.get("token"))
            else:
                rank1_token = str(first)
            rank1_counts[rank1_token] = rank1_counts.get(rank1_token, 0) + 1

            row_categories: set[str] = set()
            for entry in parsed:
                token = entry.get("token") if isinstance(entry, dict) else entry
                row_categories.update(_letter_token_category(token))
            for key in row_presence.keys():
                if key in row_categories:
                    row_presence[key] += 1

        empty_rows = total_rows - captured_rows
        lines.append(
            f"- top_logprobs payload capture: captured_rows={captured_rows}, empty_rows={empty_rows}"
        )

        if captured_rows == 0 or empty_rows > captured_rows:
            lines.append(
                "- most rows have empty payloads; skipping rank-1 token frequency and letter-token presence stats."
            )
            lines.append("")
            continue

        lines.append("")
        lines.append("### Most common rank-1 token (top 20)")
        lines.append("")
        lines.append("| token (verbatim) | count | percent_of_captured_rows |")
        lines.append("|---|---:|---:|")
        for token, count in sorted(rank1_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
            pct = (float(count) / float(captured_rows)) * 100.0 if captured_rows else 0.0
            lines.append(f"| `{token}` | {count} | {pct:.2f}% |")

        lines.append("")
        lines.append("### Letter-token presence stats (top-20 per row)")
        lines.append("")
        lines.append("| category | row_count | percent_of_all_rows |")
        lines.append("|---|---:|---:|")
        labels = [
            ("no_prefix", "no_prefix (`A`, ` A`, etc.)"),
            ("sentencepiece_prefix", "sentencepiece_prefix (`▁A`)"),
            ("bpe_prefix", "bpe_prefix (`ĠA`)"),
            ("other_prefix", "other_prefix (non-alpha leading char, e.g. `(A`, `[A`, `**A`)"),
        ]
        for key, label in labels:
            count = int(row_presence[key])
            pct = (float(count) / float(total_rows)) * 100.0 if total_rows else 0.0
            lines.append(f"| {label} | {count} | {pct:.2f}% |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _sample_rows_for_model(frame: pd.DataFrame) -> list[pd.Series]:
    samples: list[pd.Series] = []
    if "question_idx" not in frame.columns:
        return samples
    sorted_frame = frame.sort_values(by=["question_idx", "permutation_idx"], kind="stable")
    for q_idx in TARGET_SAMPLE_QUESTION_IDXS:
        match = sorted_frame[sorted_frame["question_idx"] == q_idx]
        if match.empty:
            continue
        samples.append(match.iloc[0])
    return samples


def _write_samples_markdown(*, model_runs: list[ModelRun], output_path: Path) -> None:
    lines: list[str] = [f"# MCQ logprob samples — {len(model_runs)} models", ""]
    for item in sorted(model_runs, key=lambda x: x.model):
        lines.append(f"## {item.model}")
        lines.append("")
        samples = _sample_rows_for_model(item.frame)
        if not samples:
            lines.append("No sample rows found for target question indices.")
            lines.append("")
            continue

        for idx, row in enumerate(samples, start=1):
            q_idx = row.get("question_idx")
            p_idx = row.get("permutation_idx")
            sigma = _format_sigma(row.get("permutation_sigma"))
            lines.append(
                f"### {item.model} — Row {idx}: question_idx={_display(q_idx)}, "
                f"permutation_idx={_display(p_idx)}, permutation_sigma={sigma}"
            )
            lines.append(f"correct_letter: {_display(row.get('correct_letter'))}")
            lines.append(f"predicted_letter: {_display(row.get('predicted_letter'))}")
            lines.append(f"is_correct: {_display(row.get('is_correct'))}")
            lines.append(f"error: {_display(row.get('error'))}")

            first_parsed = _parse_json_field(row.get("first_token_top_logprobs_json"))
            if not isinstance(first_parsed, list) or len(first_parsed) == 0:
                lines.append("first_token_top_logprobs_json (parsed): empty list []")
            else:
                lines.append("first_token_top_logprobs_json (parsed):")
                for entry in first_parsed:
                    if isinstance(entry, dict):
                        token = entry.get("token")
                        logprob = entry.get("logprob")
                        token_id = entry.get("token_id")
                        lines.append(
                            f"  - token={_display(token)}, logprob={_display(logprob)}, token_id={_display(token_id)}"
                        )
                    else:
                        lines.append(
                            f"  - token={_display(entry)}, logprob=null, token_id=null"
                        )

            candidate_parsed = _parse_json_field(row.get("candidate_logprobs_json"))
            lines.append("candidate_logprobs_json (parsed):")
            if not isinstance(candidate_parsed, dict):
                for letter in ["A", "B", "C", "D", "E"]:
                    lines.append(f"  - {letter}=null")
            else:
                for letter in ["A", "B", "C", "D", "E"]:
                    lines.append(f"  - {letter}={_display(candidate_parsed.get(letter))}")
            lines.append("")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    started_at = time.perf_counter()
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract null-prediction and top-token diagnostics for all model runs in one request group."
    )
    parser.add_argument(
        "--request-group-id",
        required=True,
        type=int,
        help="Request group id to extract diagnostics from.",
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/diagnostics/",
        help="Output directory for diagnostic files.",
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

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifact_service = ArtifactService(repository=repo)
        model_runs = _load_model_runs(
            request_group_id=request_group_id,
            repo=repo,
            artifact_service=artifact_service,
        )

    summary_path = output_dir / f"{request_group_id}_all_models_summary.csv"
    top_tokens_path = output_dir / f"{request_group_id}_all_models_top_tokens.md"
    samples_path = output_dir / f"{request_group_id}_all_models_samples.md"

    _write_summary_csv(model_runs=model_runs, output_path=summary_path)
    _write_top_tokens_markdown(model_runs=model_runs, output_path=top_tokens_path)
    _write_samples_markdown(model_runs=model_runs, output_path=samples_path)

    elapsed = time.perf_counter() - started_at
    print("row_counts:")
    for item in sorted(model_runs, key=lambda x: x.model):
        print(f"  {item.model}: {len(item.frame)}")
    print("output_files:")
    print(f"  {summary_path}")
    print(f"  {top_tokens_path}")
    print(f"  {samples_path}")
    print(f"total_wall_clock_seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
