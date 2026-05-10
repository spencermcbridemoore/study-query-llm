#!/usr/bin/env python3
"""Verify MCQ logprob execution artifacts for one request group."""

from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class RunMetrics:
    created_at: Optional[datetime]
    run_id: int
    run_key: str
    run_status: str
    model: str
    strategy: str
    experiment_label: str
    imported_run_id: Optional[int]
    dataset_name: str
    dataset_version: str
    result_ref: str
    parquet_rows: int
    parquet_cols: int
    error_rate: float
    accuracy: Optional[float]
    retry_count_sum: int
    retry_count_max: int
    wait_ms_sum: int
    error_counts: dict[str, int]
    is_correct_counts: dict[str, int]
    predicted_counts: dict[str, int]
    correct_letter_counts: dict[str, int]
    error_examples: dict[str, tuple[Any, Any]]


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _display_key(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and pd.isna(value):
        return "None"
    text = str(value)
    return text if text else "''"


def _series_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    out: dict[str, int] = {}
    for key, count in counts.items():
        out[_display_key(key)] = int(count)
    return out


def _normalize_error_counts(raw: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, count in raw.items():
        if key == "None":
            continue
        out[str(key)] = int(count)
    return out


def _expected_rows_for_strategy(strategy: str) -> Optional[int]:
    if strategy == "full_120":
        return 5400
    if strategy == "latin_squares_25":
        return 1125
    return None


def _apply_artifact_suffix(uri: str, artifact_suffix: str) -> str:
    if not artifact_suffix:
        return uri
    if not uri.endswith(".parquet"):
        raise ValueError(f"result_ref_not_parquet:{uri}")
    return f"{uri[:-8]}{artifact_suffix}.parquet"


def _load_metrics_for_run(
    *,
    run: ProvenancedRun,
    artifact_service: ArtifactService,
    artifact_suffix: str,
) -> RunMetrics:
    metadata = dict(run.metadata_json or {})
    model = str(metadata.get("model") or "")
    strategy = str(metadata.get("permutation_strategy") or "")
    experiment_label = str(metadata.get("experiment_label") or "")
    imported_run_id_raw = metadata.get("imported_run_id")
    imported_run_id = (
        int(imported_run_id_raw)
        if imported_run_id_raw is not None and str(imported_run_id_raw).strip()
        else None
    )
    dataset_name = str(metadata.get("dataset_name") or "")
    dataset_version = str(metadata.get("dataset_version") or "")
    result_ref = str(run.result_ref or "")
    resolved_result_ref = (
        _apply_artifact_suffix(result_ref, artifact_suffix) if result_ref else result_ref
    )

    if not result_ref:
        return RunMetrics(
            created_at=run.created_at,
            run_id=int(run.id),
            run_key=str(run.run_key or ""),
            run_status=str(run.run_status or ""),
            model=model,
            strategy=strategy,
            experiment_label=experiment_label,
            imported_run_id=imported_run_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            result_ref=resolved_result_ref,
            parquet_rows=0,
            parquet_cols=0,
            error_rate=0.0,
            accuracy=None,
            retry_count_sum=0,
            retry_count_max=0,
            wait_ms_sum=0,
            error_counts={},
            is_correct_counts={},
            predicted_counts={},
            correct_letter_counts={},
            error_examples={},
        )

    parquet_bytes = artifact_service.storage.read_from_uri(resolved_result_ref)
    frame = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
    row_count = int(len(frame))
    col_count = int(len(frame.columns))

    error_series = (
        frame["error"] if "error" in frame.columns else pd.Series([None] * row_count, dtype=object)
    )
    is_correct_series = (
        frame["is_correct"]
        if "is_correct" in frame.columns
        else pd.Series([None] * row_count, dtype=object)
    )
    predicted_series = (
        frame["predicted_letter"]
        if "predicted_letter" in frame.columns
        else pd.Series([None] * row_count, dtype=object)
    )
    correct_letter_series = (
        frame["correct_letter"]
        if "correct_letter" in frame.columns
        else pd.Series([None] * row_count, dtype=object)
    )
    retry_series = (
        pd.to_numeric(frame["rate_limit_retry_count"], errors="coerce")
        if "rate_limit_retry_count" in frame.columns
        else pd.Series([0] * row_count, dtype="int64")
    )
    wait_series = (
        pd.to_numeric(frame["rate_limit_total_wait_ms"], errors="coerce")
        if "rate_limit_total_wait_ms" in frame.columns
        else pd.Series([0] * row_count, dtype="int64")
    )

    error_non_null = int(error_series.notna().sum())
    error_rate = float(error_non_null) / float(row_count) if row_count else 0.0

    non_error_mask = error_series.isna()
    is_correct_non_error = is_correct_series[non_error_mask]
    denom = int(is_correct_non_error.notna().sum())
    correct = int((is_correct_non_error == True).sum())  # noqa: E712
    accuracy = (float(correct) / float(denom)) if denom > 0 else None

    error_counts_display = _series_counts(error_series)
    error_counts = _normalize_error_counts(error_counts_display)

    error_examples: dict[str, tuple[Any, Any]] = {}
    if error_counts:
        for error_class in error_counts.keys():
            class_mask = error_series.astype(str) == str(error_class)
            if not class_mask.any():
                continue
            first_row = frame.loc[class_mask].iloc[0]
            error_examples[error_class] = (
                first_row.get("question_idx"),
                first_row.get("permutation_idx"),
            )

    retry_sum = int(retry_series.fillna(0).sum())
    retry_max = int(retry_series.fillna(0).max()) if row_count else 0
    wait_sum = int(wait_series.fillna(0).sum())

    return RunMetrics(
        created_at=run.created_at,
        run_id=int(run.id),
        run_key=str(run.run_key or ""),
        run_status=str(run.run_status or ""),
        model=model,
        strategy=strategy,
        experiment_label=experiment_label,
        imported_run_id=imported_run_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        result_ref=resolved_result_ref,
        parquet_rows=row_count,
        parquet_cols=col_count,
        error_rate=error_rate,
        accuracy=accuracy,
        retry_count_sum=retry_sum,
        retry_count_max=retry_max,
        wait_ms_sum=wait_sum,
        error_counts=error_counts,
        is_correct_counts=_series_counts(is_correct_series),
        predicted_counts=_series_counts(predicted_series),
        correct_letter_counts=_series_counts(correct_letter_series),
        error_examples=error_examples,
    )


def _sanity_flags(metrics: list[RunMetrics]) -> list[str]:
    flags: list[str] = []
    for m in metrics:
        expected = _expected_rows_for_strategy(m.strategy)
        if expected is not None and m.parquet_rows != expected:
            flags.append(
                f"`{m.model}` row_count_mismatch: got {m.parquet_rows}, expected {expected} for `{m.strategy}`"
            )
        if m.error_rate > 0.05:
            flags.append(f"`{m.model}` high_error_rate: {_format_pct(m.error_rate)}")
        predicted_total = int(sum(count for key, count in m.predicted_counts.items() if key != "None"))
        if predicted_total > 0:
            top_letter, top_count = max(
                ((k, v) for k, v in m.predicted_counts.items() if k != "None"),
                key=lambda item: item[1],
            )
            top_ratio = float(top_count) / float(predicted_total)
            if top_ratio > 0.5:
                flags.append(
                    f"`{m.model}` predicted_letter_bias: `{top_letter}`={_format_pct(top_ratio)} of non-null predictions"
                )
        if m.run_status != "completed":
            flags.append(f"`{m.model}` non_completed_run_status: `{m.run_status}`")
    return flags


def _error_breakdown_rows(metrics: list[RunMetrics]) -> list[tuple[str, str, int, Any, Any]]:
    rows: list[tuple[str, str, int, Any, Any]] = []
    for m in metrics:
        for error_class, count in sorted(m.error_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            q_idx, p_idx = m.error_examples.get(error_class, (None, None))
            rows.append((m.model, error_class, int(count), q_idx, p_idx))
    return rows


def _resolve_output_path(request_group_id: int, raw_output: Optional[str]) -> Path:
    if raw_output:
        output = Path(raw_output).expanduser()
    else:
        output = REPO_ROOT / f"scratch/mcq_logprob_run_{request_group_id}_verification.md"
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    return output


def _iso(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _build_markdown(
    *,
    request_group_id: int,
    metrics: list[RunMetrics],
    generated_at: datetime,
) -> str:
    created_times = [m.created_at for m in metrics if m.created_at is not None]
    run_start = min(created_times) if created_times else None
    run_end = max(created_times) if created_times else None

    experiment_labels = sorted({m.experiment_label for m in metrics if m.experiment_label})
    imported_run_ids = sorted({str(m.imported_run_id) for m in metrics if m.imported_run_id is not None})
    dataset_pairs = sorted(
        {
            f"{m.dataset_name}@{m.dataset_version or 'n/a'}"
            for m in metrics
            if m.dataset_name
        }
    )
    dataset_display = ", ".join(dataset_pairs) if dataset_pairs else "n/a"

    total_rows = int(sum(m.parquet_rows for m in metrics))
    flags = _sanity_flags(metrics)
    error_rows = _error_breakdown_rows(metrics)

    out: list[str] = []
    out.append(f"# MCQ logprob run verification — request_group_id={request_group_id}")
    out.append("")
    out.append("## Run metadata")
    out.append("")
    out.append(f"- request_group_id: `{request_group_id}`")
    out.append(
        f"- experiment_label: `{', '.join(experiment_labels) if experiment_labels else 'n/a'}`"
    )
    out.append(
        f"- dataset: `{dataset_display}` (imported_run_id: `{', '.join(imported_run_ids) if imported_run_ids else 'n/a'}`)"
    )
    out.append(f"- run_timestamp_range_utc: `{_iso(run_start)}` -> `{_iso(run_end)}`")
    out.append(f"- total_provenanced_runs_rows: `{len(metrics)}`")
    out.append(f"- total_artifact_rows_across_parquets: `{total_rows}`")
    out.append("")
    out.append("## Execution row inventory")
    out.append("")
    out.append("| id | run_key | run_status | model | permutation_strategy | experiment_label | result_ref |")
    out.append("|---:|---|---|---|---|---|---|")
    for m in sorted(metrics, key=lambda item: item.run_id):
        out.append(
            f"| {m.run_id} | `{m.run_key}` | `{m.run_status}` | `{m.model}` | "
            f"`{m.strategy}` | `{m.experiment_label}` | `{m.result_ref}` |"
        )
    out.append("")
    out.append("## Per-model summary")
    out.append("")
    out.append(
        "| model | strategy | run_id | parquet rows | error_rate | accuracy | retry_count_sum | wait_ms_sum |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in sorted(metrics, key=lambda item: (item.model, item.run_id)):
        out.append(
            f"| `{m.model}` | `{m.strategy}` | {m.run_id} | {m.parquet_rows} | "
            f"{_format_pct(m.error_rate)} | {_format_pct(m.accuracy)} | "
            f"{m.retry_count_sum} | {m.wait_ms_sum} |"
        )
    out.append("")
    out.append("## Error breakdown")
    out.append("")
    if not error_rows:
        out.append("No non-null `error` values observed in artifact rows.")
    else:
        out.append("| model | error_class | count | example_row_context |")
        out.append("|---|---|---:|---|")
        for model, error_class, count, q_idx, p_idx in error_rows:
            out.append(
                f"| `{model}` | `{error_class}` | {count} | "
                f"`question_idx={q_idx}, permutation_idx={p_idx}` |"
            )
    out.append("")
    out.append("## Sanity flags")
    out.append("")
    if flags:
        for flag in flags:
            out.append(f"- {flag}")
    else:
        out.append("No sanity flags triggered.")
    out.append("")
    if flags:
        out.append("## TODOs")
        out.append("")
        for flag in flags:
            out.append(f"- [ ] investigate: {flag}")
        out.append("")
    out.append(f"Report generated at `{generated_at.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')}`.")
    out.append("")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Verify MCQ logprob run artifacts for one request_group_id."
    )
    parser.add_argument(
        "--request-group-id",
        required=True,
        type=int,
        help="Request group id whose provenanced_runs execution rows should be verified.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output markdown path. Default: scratch/mcq_logprob_run_<request_group_id>_verification.md",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix inserted before .parquet when reading artifact URIs.",
    )
    args = parser.parse_args(argv)

    request_group_id = int(args.request_group_id)
    output_path = _resolve_output_path(request_group_id, args.output_path)

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
        runs = (
            session.query(ProvenancedRun)
            .filter(ProvenancedRun.request_group_id == request_group_id)
            .order_by(ProvenancedRun.id.asc())
            .all()
        )
        if not runs:
            print(f"ERROR: no provenanced_runs rows for request_group_id={request_group_id}", file=sys.stderr)
            return 1

        artifact_service = ArtifactService(repository=repo)
        metrics: list[RunMetrics] = []
        for run in runs:
            try:
                metrics.append(
                    _load_metrics_for_run(
                        run=run,
                        artifact_service=artifact_service,
                        artifact_suffix=str(args.artifact_suffix or ""),
                    )
                )
            except Exception as exc:
                print(
                    f"ERROR: failed to load artifact for run_id={run.id} result_ref={run.result_ref!r}: {exc}",
                    file=sys.stderr,
                )
                return 1

    markdown = _build_markdown(
        request_group_id=request_group_id,
        metrics=metrics,
        generated_at=datetime.now(timezone.utc),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    flagged = _sanity_flags(metrics)
    print(f"request_group_id={request_group_id}")
    print(f"provenanced_runs_rows={len(runs)}")
    print(f"report_path={output_path}")
    print(f"sanity_flags={len(flagged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
