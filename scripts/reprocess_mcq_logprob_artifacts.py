#!/usr/bin/env python3
"""Reprocess MCQ logprob parquet artifacts with updated token matcher semantics."""

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
from study_query_llm.services.method_runners.mcq_logprob_basic import (
    _candidate_logprobs_from_top,
    _predict_letter,
)

LOGICAL_PATH_RE = re.compile(r"(?P<group_id>\d+)/(?P<step_name>[^/]+)/(?P<filename>[^/]+)$")


@dataclass(frozen=True)
class ReprocessSummary:
    run_id: int
    model: str
    strategy: str
    source_uri: str
    target_uri: str
    row_count: int
    original_non_null: int
    reprocessed_non_null: int

    @property
    def delta_non_null(self) -> int:
        return int(self.reprocessed_non_null) - int(self.original_non_null)


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _parse_json_payload(raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return None
    if isinstance(raw_value, (list, dict)):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_top_logprobs(raw_value: Any) -> list[dict[str, Any]]:
    parsed = _parse_json_payload(raw_value)
    if not isinstance(parsed, list):
        return []

    out: list[dict[str, Any]] = []
    for entry in parsed:
        if isinstance(entry, dict):
            out.append(dict(entry))
            continue
        out.append(
            {
                "token": getattr(entry, "token", None),
                "logprob": getattr(entry, "logprob", None),
                "token_id": getattr(entry, "token_id", None),
            }
        )
    return out


def _non_null_prediction_count(series: pd.Series) -> int:
    normalized = series.astype(object)
    return int(
        (normalized.notna() & (normalized.astype(str).str.strip() != "")).sum()
    )


def _extract_logical_path_from_uri(uri: str) -> str:
    normalized = str(uri).replace("\\", "/").strip()
    match = LOGICAL_PATH_RE.search(normalized)
    if match is None:
        raise ValueError(f"unable_to_extract_logical_path_from_uri:{uri}")
    return (
        f"{match.group('group_id')}/"
        f"{match.group('step_name')}/"
        f"{match.group('filename')}"
    )


def _apply_suffix_to_logical_path(logical_path: str, output_suffix: str) -> str:
    path = str(logical_path)
    if not path.endswith(".parquet"):
        raise ValueError(f"logical_path_not_parquet:{logical_path}")
    return f"{path[:-8]}{output_suffix}.parquet"


def _recompute_columns(frame: pd.DataFrame) -> tuple[list[str], list[str | None], list[bool | None]]:
    row_count = int(len(frame))
    top_series = (
        frame["first_token_top_logprobs_json"]
        if "first_token_top_logprobs_json" in frame.columns
        else pd.Series([None] * row_count, dtype=object)
    )
    correct_series = (
        frame["correct_letter"]
        if "correct_letter" in frame.columns
        else pd.Series([None] * row_count, dtype=object)
    )

    candidate_json: list[str] = []
    predicted_letters: list[str | None] = []
    is_correct_values: list[bool | None] = []

    for raw_top, raw_correct_letter in zip(top_series, correct_series):
        top_logprobs = _normalize_top_logprobs(raw_top)
        candidate = _candidate_logprobs_from_top(top_logprobs)
        predicted_letter = _predict_letter(candidate)
        correct_letter = _to_text(raw_correct_letter).strip().upper()
        is_correct = bool(predicted_letter == correct_letter) if predicted_letter else None

        candidate_json.append(json.dumps(candidate, sort_keys=True))
        predicted_letters.append(predicted_letter)
        is_correct_values.append(is_correct)

    return candidate_json, predicted_letters, is_correct_values


def main(argv: Optional[list[str]] = None) -> int:
    started_at = time.perf_counter()
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Reprocess MCQ logprob parquet artifacts using fixed token matcher semantics."
    )
    parser.add_argument(
        "--request-group-id",
        required=True,
        type=int,
        help="Request group id whose provenanced_runs rows should be reprocessed.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_v2_matcher_fix",
        help="Suffix inserted before .parquet for rewritten artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print reprocessing plan and deltas without writing artifacts.",
    )
    args = parser.parse_args(argv)

    request_group_id = int(args.request_group_id)
    output_suffix = str(args.output_suffix)
    dry_run = bool(args.dry_run)

    if not output_suffix:
        print("ERROR: --output-suffix must be non-empty.", file=sys.stderr)
        return 1

    db_url = _resolve_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(db_url),
    )

    summaries: list[ReprocessSummary] = []

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifact_service = ArtifactService(repository=repo)

        runs = (
            session.query(ProvenancedRun)
            .filter(ProvenancedRun.request_group_id == request_group_id)
            .order_by(ProvenancedRun.id.asc())
            .all()
        )
        if not runs:
            print(
                f"ERROR: no provenanced_runs rows for request_group_id={request_group_id}",
                file=sys.stderr,
            )
            return 1

        for run in runs:
            source_uri = str(run.result_ref or "").strip()
            if not source_uri:
                print(
                    f"ERROR: run_id={run.id} has empty result_ref; cannot reprocess.",
                    file=sys.stderr,
                )
                return 1

            source_bytes = artifact_service.storage.read_from_uri(source_uri)
            source_frame = pd.read_parquet(io.BytesIO(source_bytes), engine="pyarrow")
            output_frame = source_frame.copy()

            candidate_json, predicted_letters, is_correct_values = _recompute_columns(output_frame)
            output_frame["candidate_logprobs_json"] = candidate_json
            output_frame["predicted_letter"] = predicted_letters
            output_frame["is_correct"] = is_correct_values

            original_non_null = (
                _non_null_prediction_count(source_frame["predicted_letter"])
                if "predicted_letter" in source_frame.columns
                else 0
            )
            reprocessed_non_null = _non_null_prediction_count(output_frame["predicted_letter"])

            source_logical_path = _extract_logical_path_from_uri(source_uri)
            target_logical_path = _apply_suffix_to_logical_path(
                source_logical_path,
                output_suffix,
            )
            planned_target_uri = artifact_service.storage.get_uri(target_logical_path)

            if dry_run:
                target_uri = planned_target_uri
            else:
                out_buffer = io.BytesIO()
                output_frame.to_parquet(out_buffer, engine="pyarrow", index=False)
                target_uri = artifact_service.storage.write(
                    target_logical_path,
                    out_buffer.getvalue(),
                    content_type="application/octet-stream",
                )

            metadata = dict(run.metadata_json or {})
            summaries.append(
                ReprocessSummary(
                    run_id=int(run.id),
                    model=str(metadata.get("model") or ""),
                    strategy=str(metadata.get("permutation_strategy") or ""),
                    source_uri=source_uri,
                    target_uri=str(target_uri),
                    row_count=int(len(output_frame)),
                    original_non_null=int(original_non_null),
                    reprocessed_non_null=int(reprocessed_non_null),
                )
            )

    mode = "dry_run" if dry_run else "write"
    total_original_non_null = int(sum(item.original_non_null for item in summaries))
    total_reprocessed_non_null = int(sum(item.reprocessed_non_null for item in summaries))
    total_delta = total_reprocessed_non_null - total_original_non_null

    print(f"mode={mode}")
    print(f"request_group_id={request_group_id}")
    print(f"runs={len(summaries)}")
    for item in summaries:
        print(
            f"model={item.model} run_id={item.run_id} strategy={item.strategy} rows={item.row_count} "
            f"non_null={item.original_non_null}->{item.reprocessed_non_null} "
            f"delta={item.delta_non_null:+d} target_uri={item.target_uri}"
        )
    print(
        f"total_non_null={total_original_non_null}->{total_reprocessed_non_null} delta={total_delta:+d}"
    )
    print(f"wall_clock_seconds={time.perf_counter() - started_at:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
