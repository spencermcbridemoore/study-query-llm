#!/usr/bin/env python3
"""Materialize BANKING77 min/max contrast snapshots for 6/12/20 label subsets.

Contrast is defined over per-label median utterance character length:

- max contrast: combine shortest-median and longest-median labels
- min contrast: choose the tightest contiguous median-length window

This script creates six dataset_snapshot groups total:
    min_6, max_6, min_12, max_12, min_20, max_20
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.banking77 import BANKING77_DATASET_SLUG
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri, parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SubquerySpec
from study_query_llm.services.artifact_service import ArtifactService

LABEL_COUNTS: tuple[int, ...] = (6, 12, 20)
EXPECTED_BANK77_LABELS = 77
SNAPSHOT_GROUP_TYPE = "dataset_snapshot"
SNAPSHOT_NAME_PREFIX = f"snap:{BANKING77_DATASET_SLUG}:contrast"


@dataclass(frozen=True)
class ContrastSelection:
    mode: str
    label_count: int
    labels: tuple[str, ...]
    median_range: float


def _resolve_database_url(database_url: str | None, artifact_dir: str) -> tuple[str, bool]:
    explicit = str(database_url or "").strip()
    if explicit:
        return explicit, False
    env_value = str(os.environ.get("DATABASE_URL") or "").strip()
    if env_value:
        return env_value, False
    fallback = Path(artifact_dir) / "bank77_contrast_snapshots.sqlite3"
    return f"sqlite:///{fallback.resolve().as_posix()}", True


def _redact_database_url(raw_url: str) -> str:
    """Mask URL password for manifests/logs while preserving target identity."""
    try:
        parsed = urlsplit(str(raw_url))
    except Exception:
        return str(raw_url)
    netloc = parsed.netloc
    if "@" not in netloc:
        return str(raw_url)
    auth_part, host_part = netloc.rsplit("@", 1)
    if ":" in auth_part:
        username, _password = auth_part.split(":", 1)
        auth_part = f"{username}:***"
    netloc = f"{auth_part}@{host_part}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _profile_id(mode: str, label_count: int) -> str:
    return f"{str(mode)}_{int(label_count)}"


def _canonical_snapshot_group_name(*, mode: str, label_count: int, spec_hash: str) -> str:
    """Stable descriptive name for BANK77 contrast snapshot groups."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"min", "max"}:
        raise ValueError(f"unsupported contrast mode for naming: {mode!r}")
    label_count_int = int(label_count)
    spec_hash_text = str(spec_hash or "").strip().lower()
    if len(spec_hash_text) < 8:
        raise ValueError(f"spec_hash too short for naming: {spec_hash!r}")
    return (
        f"{SNAPSHOT_NAME_PREFIX}:{normalized_mode}:"
        f"l{label_count_int}:{spec_hash_text[:8]}"
    )


def _extract_group_metadata(row: Group) -> dict[str, Any]:
    metadata = row.metadata_json
    return dict(metadata) if isinstance(metadata, dict) else {}


def _preflight_snapshot_group(
    *,
    db: DatabaseConnectionV2,
    group_id: int,
    source_dataframe_group_id: int,
    expected_spec_hash: str,
    expected_resolved_index_hash: str,
) -> tuple[str, str, dict[str, Any]]:
    """Validate that target snapshot group matches expected provenance keys."""
    target_id = int(group_id)
    with db.session_scope() as session:
        row = session.query(Group).filter(Group.id == target_id).first()
        if row is None:
            raise ValueError(f"snapshot group id={target_id} not found")
        if str(row.group_type) != SNAPSHOT_GROUP_TYPE:
            raise ValueError(
                f"group id={target_id} type={row.group_type!r}; expected {SNAPSHOT_GROUP_TYPE!r}"
            )
        metadata = _extract_group_metadata(row)
        current_name = str(row.name or "")

    dataset_slug = str(metadata.get("dataset_slug") or "").strip().lower()
    if dataset_slug != BANKING77_DATASET_SLUG:
        raise ValueError(
            f"group id={target_id} dataset_slug={dataset_slug!r}; "
            f"expected {BANKING77_DATASET_SLUG!r}"
        )
    metadata_df_id = int(metadata.get("source_dataframe_group_id") or -1)
    if metadata_df_id != int(source_dataframe_group_id):
        raise ValueError(
            f"group id={target_id} source_dataframe_group_id={metadata_df_id}; "
            f"expected {int(source_dataframe_group_id)}"
        )
    metadata_spec_hash = str(metadata.get("spec_hash") or "")
    if metadata_spec_hash != str(expected_spec_hash):
        raise ValueError(
            f"group id={target_id} spec_hash mismatch: "
            f"{metadata_spec_hash!r} != {expected_spec_hash!r}"
        )
    metadata_resolved_hash = str(metadata.get("resolved_index_hash") or "")
    if metadata_resolved_hash != str(expected_resolved_index_hash):
        raise ValueError(
            f"group id={target_id} resolved_index_hash mismatch: "
            f"{metadata_resolved_hash!r} != {expected_resolved_index_hash!r}"
        )
    return current_name, str(dataset_slug), metadata


def _apply_snapshot_group_name(
    *,
    db: DatabaseConnectionV2,
    group_id: int,
    target_name: str,
) -> tuple[str, bool]:
    """Rename a snapshot group if needed; return final name and whether changed."""
    with db.session_scope() as session:
        row = session.query(Group).filter(Group.id == int(group_id)).first()
        if row is None:
            raise ValueError(f"snapshot group id={group_id} vanished before rename")
        current_name = str(row.name or "")
        if current_name == target_name:
            return current_name, False
        row.name = str(target_name)
        session.flush()
        return str(row.name or ""), True


def _load_bank77_dataframe(*, db: DatabaseConnectionV2, dataframe_group_id: int, artifact_dir: str):
    import pandas as pd

    with db.session_scope() as session:
        dataframe_parquet_uri = find_dataframe_parquet_uri(session, int(dataframe_group_id))
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(dataframe_parquet_uri)
    table = pq.read_table(
        source=io.BytesIO(payload),
        columns=["position", "text", "label", "label_name"],
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError("BANKING77 dataframe parquet is empty")
    frame["text"] = frame["text"].astype(str)
    frame["label_name"] = frame["label_name"].astype(str)
    frame["char_length"] = frame["text"].str.len().astype(float)
    return frame


def _label_filter_expr(labels: Sequence[str]) -> str:
    quoted = ", ".join(json.dumps(label, ensure_ascii=True) for label in labels)
    return f"label_name in [{quoted}]"


def _build_summary(frame):
    summary = (
        frame.groupby("label_name")["char_length"]
        .agg(count="size", median_len="median")
        .sort_values("median_len")
    )
    unique_labels = int(summary.shape[0])
    if unique_labels != EXPECTED_BANK77_LABELS:
        raise ValueError(
            f"Expected {EXPECTED_BANK77_LABELS} BANKING77 labels, found {unique_labels}"
        )
    return summary


def _select_max_contrast(summary, label_count: int) -> ContrastSelection:
    if label_count % 2 != 0:
        raise ValueError("label_count must be even for symmetric max-contrast selection")
    labels = summary.index.tolist()
    low_n = label_count // 2
    high_n = label_count - low_n
    selected = labels[:low_n] + labels[-high_n:]
    medians = summary.loc[selected, "median_len"].astype(float).tolist()
    return ContrastSelection(
        mode="max",
        label_count=label_count,
        labels=tuple(selected),
        median_range=float(max(medians) - min(medians)),
    )


def _select_min_contrast(summary, label_count: int) -> ContrastSelection:
    labels = summary.index.tolist()
    medians = summary["median_len"].astype(float).tolist()
    if label_count > len(labels):
        raise ValueError(
            f"label_count {label_count} exceeds label cardinality {len(labels)}"
        )
    best_start = 0
    best_range = float("inf")
    for start in range(0, len(labels) - label_count + 1):
        end = start + label_count - 1
        window_range = float(medians[end] - medians[start])
        if window_range < best_range:
            best_range = window_range
            best_start = start
    selected = labels[best_start : best_start + label_count]
    return ContrastSelection(
        mode="min",
        label_count=label_count,
        labels=tuple(selected),
        median_range=best_range,
    )


def _render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Profile | Labels | Rows | Snapshot group_id | Reused | Median range | Snapshot name |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {label_count} | {row_count} | {snapshot_group_id} | {reused} | {median_range:.1f} | {snapshot_group_name} |".format(
                profile=row["profile"],
                label_count=row["label_count"],
                row_count=row["row_count"],
                snapshot_group_id=row["snapshot_group_id"],
                reused=str(bool(row["reused"])).lower(),
                median_range=float(row["median_range"]),
                snapshot_group_name=row.get("snapshot_group_name") or "",
            )
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 6 BANKING77 snapshots for min/max contrast at "
            "label counts 6/12/20."
        )
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL override (defaults to DATABASE_URL env; else local sqlite fallback).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Artifact base directory (default: artifacts).",
    )
    parser.add_argument(
        "--force-snapshot",
        action="store_true",
        help="Bypass snapshot idempotent reuse and force fresh snapshot runs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON manifest path (default: <artifact-dir>/bank77_contrast_snapshots.json).",
    )
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        help=(
            "Skip applying group-name updates (still validates and reports canonical "
            "target names)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    artifact_dir = str(Path(args.artifact_dir).resolve())
    os.makedirs(artifact_dir, exist_ok=True)

    # Make local artifact writes deterministic and independent of external blob config.
    os.environ["ARTIFACT_STORAGE_BACKEND"] = "local"
    os.environ["ARTIFACT_STORAGE_STRICT_MODE"] = "0"
    os.environ["ARTIFACT_RUNTIME_ENV"] = "dev"

    database_url, used_fallback_db = _resolve_database_url(args.database_url, artifact_dir)
    write_intent = WriteIntent.SANDBOX if used_fallback_db else WriteIntent.CANONICAL
    db = DatabaseConnectionV2(
        database_url,
        enable_pgvector=False,
        write_intent=write_intent,
    )
    db.init_db()

    acquire_spec = ACQUIRE_REGISTRY[BANKING77_DATASET_SLUG]
    acquired = acquire(acquire_spec, db=db, artifact_dir=artifact_dir)
    parsed = parse(acquired.group_id, db=db, artifact_dir=artifact_dir)
    frame = _load_bank77_dataframe(
        db=db,
        dataframe_group_id=int(parsed.group_id),
        artifact_dir=artifact_dir,
    )
    summary = _build_summary(frame)

    selections: list[ContrastSelection] = []
    for label_count in LABEL_COUNTS:
        selections.append(_select_min_contrast(summary, label_count))
        selections.append(_select_max_contrast(summary, label_count))

    results: list[dict[str, Any]] = []
    for selection in selections:
        spec = SubquerySpec(
            label_mode="labeled",
            filter_expr=_label_filter_expr(selection.labels),
        )
        profile = _profile_id(selection.mode, selection.label_count)
        snapped = snapshot(
            parsed.group_id,
            subquery_spec=spec,
            force=bool(args.force_snapshot),
            db=db,
            artifact_dir=artifact_dir,
        )
        results.append(
            {
                "profile": profile,
                "mode": selection.mode,
                "label_count": int(selection.label_count),
                "labels": list(selection.labels),
                "median_range": float(selection.median_range),
                "snapshot_group_id": int(snapped.group_id),
                "snapshot_run_id": (
                    int(snapped.run_id) if snapped.run_id is not None else None
                ),
                "row_count": int(snapped.metadata.get("row_count", 0)),
                "reused": bool(snapped.metadata.get("reused", False)),
                "spec_hash": str(snapped.metadata.get("spec_hash") or ""),
                "resolved_index_hash": str(
                    snapped.metadata.get("resolved_index_hash") or ""
                ),
                "artifact_uri": str(snapped.artifact_uris.get("subquery_spec.json") or ""),
            }
        )

    # Preflight each target group before optional rename.
    for row in results:
        expected_name = _canonical_snapshot_group_name(
            mode=str(row["mode"]),
            label_count=int(row["label_count"]),
            spec_hash=str(row["spec_hash"]),
        )
        current_name, _dataset_slug, _metadata = _preflight_snapshot_group(
            db=db,
            group_id=int(row["snapshot_group_id"]),
            source_dataframe_group_id=int(parsed.group_id),
            expected_spec_hash=str(row["spec_hash"]),
            expected_resolved_index_hash=str(row["resolved_index_hash"]),
        )
        row["snapshot_group_name_before"] = current_name
        row["snapshot_group_name_target"] = expected_name
        row["rename_required"] = bool(current_name != expected_name)

        if bool(args.skip_rename):
            row["rename_applied"] = False
            row["snapshot_group_name"] = current_name
            continue

        final_name, changed = _apply_snapshot_group_name(
            db=db,
            group_id=int(row["snapshot_group_id"]),
            target_name=expected_name,
        )
        row["rename_applied"] = bool(changed)
        row["snapshot_group_name"] = final_name

    manifest_path = Path(
        args.output_json
        if args.output_json
        else Path(artifact_dir) / "bank77_contrast_snapshots.json"
    ).resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_slug": BANKING77_DATASET_SLUG,
                "source_dataframe_group_id": int(parsed.group_id),
                "acquire_group_id": int(acquired.group_id),
                "database_url_redacted": _redact_database_url(database_url),
                "used_fallback_sqlite_database": bool(used_fallback_db),
                "artifact_dir": artifact_dir,
                "rename_applied": not bool(args.skip_rename),
                "profiles": results,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"[setup] database_url={_redact_database_url(database_url)}")
    print(f"[setup] artifact_dir={artifact_dir}")
    print(f"[ok] wrote manifest: {manifest_path.as_posix()}")
    print()
    print(_render_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
