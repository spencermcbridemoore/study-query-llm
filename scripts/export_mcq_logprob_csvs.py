#!/usr/bin/env python3
"""Export canonical MCQ logprob experiment artifacts as CSV files."""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.services.artifact_service import ArtifactService

_JSON_STRING_COLUMNS: tuple[str, ...] = (
    "first_token_top_logprobs_json",
    "candidate_logprobs_json",
)


@dataclass(frozen=True)
class CompletedRunRow:
    run_id: int
    request_group_id: Optional[int]
    run_key: str
    model: str
    prompt_template_version: str
    created_at: datetime
    result_ref: str


@dataclass(frozen=True)
class CanonicalSelection:
    model: str
    prompt_template_version: str
    selected: CompletedRunRow
    ignored: tuple[CompletedRunRow, ...]


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def _resolve_output_dir(
    *,
    experiment_label: str,
    raw_output_dir: Optional[str],
) -> Path:
    if raw_output_dir:
        output_dir = Path(raw_output_dir).expanduser()
    else:
        output_dir = REPO_ROOT / "scratch" / "exports" / str(experiment_label)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    return output_dir


def _slug_model_id(model_id: str) -> str:
    token = re.sub(r"[\/.\-]+", "_", str(model_id or ""))
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "model"


def _slug_format_name(prompt_template_version: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(prompt_template_version or ""))
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "format"


def _to_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text_value = str(value)
    if text_value.endswith("Z"):
        text_value = f"{text_value[:-1]}+00:00"
    parsed = datetime.fromisoformat(text_value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_null_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        probe = pd.isna(value)
    except Exception:
        return False
    if isinstance(probe, bool):
        return probe
    return False


def _serialize_permutation_sigma(value: Any) -> str:
    if _is_null_scalar(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        materialized = tolist()
        if isinstance(materialized, list):
            return "[" + ", ".join(str(item) for item in materialized) + "]"
    return str(value)


def _preserve_json_text(value: Any) -> str:
    if _is_null_scalar(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _prepare_frame_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "permutation_sigma" in prepared.columns:
        prepared["permutation_sigma"] = prepared["permutation_sigma"].map(
            _serialize_permutation_sigma
        )
    for column in _JSON_STRING_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].map(_preserve_json_text)
    return prepared


def _query_completed_rows(
    *,
    db: DatabaseConnectionV2,
    experiment_label: str,
) -> list[CompletedRunRow]:
    postgres_query = text(
        """
        SELECT
            id AS run_id,
            request_group_id,
            run_key,
            result_ref,
            created_at,
            metadata_json->>'model' AS model,
            metadata_json->>'prompt_template_version' AS prompt_template_version
        FROM provenanced_runs
        WHERE metadata_json->>'experiment_label' = :experiment_label
          AND run_kind = 'execution'
          AND run_status = 'completed'
          AND metadata_json->>'model' IS NOT NULL
          AND metadata_json->>'prompt_template_version' IS NOT NULL
        ORDER BY created_at DESC, id DESC
        """
    )
    sqlite_query = text(
        """
        SELECT
            id AS run_id,
            request_group_id,
            run_key,
            result_ref,
            created_at,
            json_extract(metadata_json, '$.model') AS model,
            json_extract(metadata_json, '$.prompt_template_version') AS prompt_template_version
        FROM provenanced_runs
        WHERE json_extract(metadata_json, '$.experiment_label') = :experiment_label
          AND run_kind = 'execution'
          AND run_status = 'completed'
          AND json_extract(metadata_json, '$.model') IS NOT NULL
          AND json_extract(metadata_json, '$.prompt_template_version') IS NOT NULL
        ORDER BY created_at DESC, id DESC
        """
    )

    with db.session_scope() as session:
        bind = session.get_bind()
        dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "")).lower()
        query = sqlite_query if dialect_name == "sqlite" else postgres_query
        rows = session.execute(
            query,
            {"experiment_label": str(experiment_label)},
        ).mappings().all()

    out: list[CompletedRunRow] = []
    for row in rows:
        model = str(row.get("model") or "").strip()
        prompt_template_version = str(row.get("prompt_template_version") or "").strip()
        if not model or not prompt_template_version:
            continue
        result_ref = str(row.get("result_ref") or "").strip()
        if not result_ref:
            continue
        request_group_id_raw = row.get("request_group_id")
        request_group_id = (
            int(request_group_id_raw)
            if request_group_id_raw is not None and str(request_group_id_raw).strip()
            else None
        )
        out.append(
            CompletedRunRow(
                run_id=int(row.get("run_id")),
                request_group_id=request_group_id,
                run_key=str(row.get("run_key") or ""),
                model=model,
                prompt_template_version=prompt_template_version,
                created_at=_coerce_datetime(row.get("created_at")),
                result_ref=result_ref,
            )
        )
    return out


def _select_canonical_rows(rows: list[CompletedRunRow]) -> list[CanonicalSelection]:
    grouped: dict[tuple[str, str], list[CompletedRunRow]] = {}
    for row in rows:
        key = (row.model, row.prompt_template_version)
        grouped.setdefault(key, []).append(row)

    selections: list[CanonicalSelection] = []
    for key in sorted(grouped.keys()):
        candidates = grouped[key]
        candidates_sorted = sorted(
            candidates,
            key=lambda item: (item.created_at, item.run_id),
            reverse=True,
        )
        selected = candidates_sorted[0]
        ignored = tuple(candidates_sorted[1:])
        selections.append(
            CanonicalSelection(
                model=key[0],
                prompt_template_version=key[1],
                selected=selected,
                ignored=ignored,
            )
        )
    return selections


def _write_manifest(
    *,
    manifest_path: Path,
    selections: list[CanonicalSelection],
) -> None:
    lines: list[str] = []
    lines.append("# Canonical row selection manifest")
    lines.append("")
    lines.append(
        "| model | format | run_id | request_group_id | created_at | result_ref | duplicate_run_ids_ignored |"
    )
    lines.append("|---|---|---:|---:|---|---|---|")
    for selection in selections:
        duplicate_ids = ", ".join(str(item.run_id) for item in selection.ignored)
        request_group_id = (
            str(selection.selected.request_group_id)
            if selection.selected.request_group_id is not None
            else ""
        )
        lines.append(
            f"| {selection.model} | {selection.prompt_template_version} | "
            f"{selection.selected.run_id} | {request_group_id} | "
            f"{_to_utc_iso(selection.selected.created_at)} | "
            f"{selection.selected.result_ref} | {duplicate_ids} |"
        )
    lines.append("")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Export canonical MCQ logprob parquet artifacts to per-cell CSVs and one "
            "concatenated CSV for offline analysis."
        )
    )
    parser.add_argument(
        "--experiment-label",
        required=True,
        help="Experiment label stored at metadata_json.experiment_label.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: scratch/exports/<experiment_label>/).",
    )
    args = parser.parse_args(argv)

    db_url = _resolve_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    output_dir = _resolve_output_dir(
        experiment_label=str(args.experiment_label),
        raw_output_dir=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(db_url),
    )
    completed_rows = _query_completed_rows(
        db=db,
        experiment_label=str(args.experiment_label),
    )
    if not completed_rows:
        print(
            f"ERROR: no completed execution rows for experiment_label={args.experiment_label}",
            file=sys.stderr,
        )
        return 1

    selections = _select_canonical_rows(completed_rows)
    if not selections:
        print("ERROR: no canonical rows selected.", file=sys.stderr)
        return 1

    print(
        f"selection_scan: completed_rows={len(completed_rows)} "
        f"canonical_rows={len(selections)}"
    )
    print("selection_details:")
    for selection in selections:
        ignored_ids = [item.run_id for item in selection.ignored]
        print(
            "  selected "
            f"model={selection.model} format={selection.prompt_template_version} "
            f"run_id={selection.selected.run_id} "
            f"request_group_id={selection.selected.request_group_id} "
            f"created_at={_to_utc_iso(selection.selected.created_at)} "
            f"result_ref={selection.selected.result_ref} "
            f"duplicate_run_ids_ignored={ignored_ids}"
        )
        for ignored in selection.ignored:
            print(
                "    ignored_duplicate "
                f"run_id={ignored.run_id} "
                f"request_group_id={ignored.request_group_id} "
                f"created_at={_to_utc_iso(ignored.created_at)} "
                f"result_ref={ignored.result_ref}"
            )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifact_service = ArtifactService(repository=repo)

        master_frames: list[pd.DataFrame] = []
        per_cell_row_counts: list[tuple[str, str, int, Path]] = []
        for selection in selections:
            parquet_bytes = artifact_service.storage.read_from_uri(selection.selected.result_ref)
            parquet_frame = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
            csv_frame = _prepare_frame_for_csv(parquet_frame)

            model_slug = _slug_model_id(selection.model)
            format_slug = _slug_format_name(selection.prompt_template_version)
            csv_path = output_dir / f"{model_slug}_{format_slug}.csv"
            csv_frame.to_csv(csv_path, index=False, encoding="utf-8")
            per_cell_row_counts.append(
                (
                    selection.model,
                    selection.prompt_template_version,
                    int(len(csv_frame)),
                    csv_path,
                )
            )

            master_prefix = pd.DataFrame(
                {
                    "model": [selection.model] * len(csv_frame),
                    "prompt_template_version": [selection.prompt_template_version]
                    * len(csv_frame),
                }
            )
            master_frames.append(
                pd.concat(
                    [master_prefix.reset_index(drop=True), csv_frame.reset_index(drop=True)],
                    axis=1,
                )
            )

    all_rows_path = output_dir / "all_rows.csv"
    if master_frames:
        master_frame = pd.concat(master_frames, ignore_index=True)
    else:
        master_frame = pd.DataFrame(columns=["model", "prompt_template_version"])
    master_frame.to_csv(all_rows_path, index=False, encoding="utf-8")

    manifest_path = output_dir / "canonical_rows_manifest.md"
    _write_manifest(manifest_path=manifest_path, selections=selections)

    print("")
    print("=== MCQ EXPORT SUMMARY ===")
    for model, prompt_template_version, row_count, csv_path in sorted(
        per_cell_row_counts,
        key=lambda item: (item[0], item[1]),
    ):
        print(
            f"{model} | {prompt_template_version} | rows={row_count} | csv={csv_path}"
        )
    print(f"canonical_rows={len(selections)}")
    print(f"master_rows={len(master_frame)}")
    print(f"per_cell_csv_count={len(per_cell_row_counts)}")
    print(f"output_dir={output_dir}")
    print(f"master_csv={all_rows_path}")
    print(f"manifest_path={manifest_path}")
    print("=== END MCQ EXPORT SUMMARY ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
