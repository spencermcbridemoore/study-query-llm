"""Read-only export of snapshot input rows (text + optional labels) to CSV.

Default scope matches the recent clustering coverage export snapshots:
6, 9, 10, 17, 18.
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from dotenv import dotenv_values
from sqlalchemy import create_engine, text

from study_query_llm.services.artifact_service import ArtifactService

SNAPSHOT_SCOPE: dict[int, str] = {
    6: "sources_uncertainty_qc",
    9: "estela",
    10: "estela research",
    17: "banking77 contrast min l6",
    18: "banking77 contrast max l6",
}
SNAPSHOT_IDS = sorted(SNAPSHOT_SCOPE.keys())
CSV_COLUMNS = [
    "snapshot_id",
    "snapshot_name",
    "dataset_slug",
    "snapshot_scope",
    "snapshot_position",
    "resolved_position",
    "resolved_source_id",
    "source_id",
    "text",
    "label",
    "label_name",
    "has_label",
]


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_db_url(root: Path) -> str:
    env = dotenv_values(root / ".env")
    url = (
        os.environ.get("DATABASE_URL")
        or env.get("DATABASE_URL")
        or os.environ.get("CANONICAL_DATABASE_URL")
        or env.get("CANONICAL_DATABASE_URL")
        or ""
    ).strip()
    if not url:
        raise RuntimeError("DATABASE_URL / CANONICAL_DATABASE_URL is not set.")
    return url


def _fetch_snapshots(conn) -> list[dict[str, Any]]:
    rows = (
        conn.execute(
            text(
                """
                SELECT
                    g.id AS snapshot_id,
                    g.name AS snapshot_name,
                    g.metadata_json AS metadata_json
                FROM groups g
                WHERE g.group_type = 'dataset_snapshot'
                  AND g.id = ANY(:snapshot_ids)
                ORDER BY g.id ASC
                """
            ),
            {"snapshot_ids": SNAPSHOT_IDS},
        )
        .mappings()
        .all()
    )
    snapshots: list[dict[str, Any]] = []
    for row in rows:
        sid = int(row["snapshot_id"])
        md = dict(row["metadata_json"] or {})
        snapshots.append(
            {
                "snapshot_id": sid,
                "snapshot_name": str(row.get("snapshot_name") or f"snapshot_{sid}"),
                "dataset_slug": str(md.get("dataset_slug") or ""),
                "source_dataframe_group_id": _safe_int(md.get("source_dataframe_group_id")),
                "snapshot_row_count": _safe_int(md.get("row_count")) or 0,
                "snapshot_scope": SNAPSHOT_SCOPE.get(sid, ""),
            }
        )
    found = {int(row["snapshot_id"]) for row in snapshots}
    missing = sorted(set(SNAPSHOT_IDS) - found)
    if missing:
        raise RuntimeError(f"Missing snapshots in DB: {missing}")
    return snapshots


def _latest_group_artifact_uri(conn, *, group_id: int, artifact_type: str) -> str:
    row = (
        conn.execute(
            text(
                """
                SELECT ca.uri
                FROM call_artifacts ca
                WHERE ca.artifact_type = :artifact_type
                  AND (ca.metadata_json->>'group_id')::int = :group_id
                ORDER BY ca.id DESC
                LIMIT 1
                """
            ),
            {"artifact_type": artifact_type, "group_id": int(group_id)},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise RuntimeError(
            f"No artifact uri found for group_id={group_id}, type={artifact_type!r}"
        )
    return str(row["uri"])


def _read_snapshot_payload(artifact_service: ArtifactService, *, uri: str) -> dict[str, Any]:
    raw = artifact_service.storage.read_from_uri(uri)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Snapshot payload is not JSON object: {uri}")
    return payload


def _read_dataframe_by_position(
    artifact_service: ArtifactService,
    *,
    uri: str,
) -> dict[int, dict[str, Any]]:
    raw = artifact_service.storage.read_from_uri(uri)
    table = pq.read_table(
        io.BytesIO(raw),
        columns=["position", "source_id", "text", "label", "label_name"],
    )
    frame = table.to_pandas()
    out: dict[int, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        pos = int(row.position)
        out[pos] = {
            "source_id": str(row.source_id),
            "text": str(row.text),
            "label": _normalize_optional_label(row.label),
            "label_name": None if row.label_name is None else str(row.label_name),
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
            delimiter=",",
            quotechar='"',
            doublequote=True,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    export_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_dir = root / "scratch" / "exports" / f"snapshot_inputs_{export_stamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    db_url = _load_db_url(root)
    engine = create_engine(db_url, pool_pre_ping=True)
    artifact_service = ArtifactService(artifact_dir="artifacts")

    all_rows: list[dict[str, Any]] = []
    per_snapshot_paths: list[Path] = []
    per_snapshot_counts: dict[int, int] = {}

    with engine.connect() as conn:
        snapshots = _fetch_snapshots(conn)
        for snap in snapshots:
            snapshot_id = int(snap["snapshot_id"])
            source_dataframe_group_id = _safe_int(snap["source_dataframe_group_id"])
            if source_dataframe_group_id is None or source_dataframe_group_id <= 0:
                raise RuntimeError(
                    f"snapshot {snapshot_id} missing source_dataframe_group_id"
                )
            subquery_uri = _latest_group_artifact_uri(
                conn,
                group_id=snapshot_id,
                artifact_type="dataset_subquery_spec",
            )
            parquet_uri = _latest_group_artifact_uri(
                conn,
                group_id=source_dataframe_group_id,
                artifact_type="dataset_canonical_parquet",
            )

            snapshot_payload = _read_snapshot_payload(artifact_service, uri=subquery_uri)
            resolved_index_raw = list(snapshot_payload.get("resolved_index") or [])
            dataframe_map = _read_dataframe_by_position(artifact_service, uri=parquet_uri)

            rows: list[dict[str, Any]] = []
            for idx, item in enumerate(resolved_index_raw):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise RuntimeError(
                        f"Invalid resolved_index row in snapshot {snapshot_id}: {item!r}"
                    )
                resolved_position = int(item[0])
                resolved_source_id = str(item[1])
                dataframe_row = dataframe_map.get(resolved_position)
                if dataframe_row is None:
                    raise RuntimeError(
                        f"Snapshot {snapshot_id} position {resolved_position} "
                        "not found in source dataframe artifact."
                    )
                label = dataframe_row.get("label")
                label_name = dataframe_row.get("label_name")
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "snapshot_name": str(snap["snapshot_name"]),
                        "dataset_slug": str(snap["dataset_slug"]),
                        "snapshot_scope": str(snap["snapshot_scope"]),
                        "snapshot_position": int(idx),
                        "resolved_position": resolved_position,
                        "resolved_source_id": resolved_source_id,
                        "source_id": str(dataframe_row["source_id"]),
                        "text": str(dataframe_row["text"]),
                        "label": label,
                        "label_name": label_name,
                        "has_label": 1 if label is not None else 0,
                    }
                )

            out_path = export_dir / f"snapshot_inputs__snap{snapshot_id}.csv"
            _write_csv(out_path, rows)
            per_snapshot_paths.append(out_path)
            per_snapshot_counts[snapshot_id] = len(rows)
            all_rows.extend(rows)

    combined_path = export_dir / "snapshot_inputs_all.csv"
    _write_csv(combined_path, all_rows)

    readme_lines = [
        "# Snapshot Input CSV Export",
        "",
        "- One CSV per snapshot with resolved input rows (strings) from snapshot artifacts.",
        "- Includes labels when present (`label`, `label_name`, `has_label`).",
        "- `snapshot_position` preserves snapshot order used by downstream analysis labels.",
        "",
        "## Files",
        "",
    ]
    for path in per_snapshot_paths:
        readme_lines.append(f"- `{path.name}`")
    readme_lines.append(f"- `{combined_path.name}`")
    (export_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    print(f"EXPORT_DIR={export_dir}")
    for sid in SNAPSHOT_IDS:
        print(f"SNAPSHOT_{sid}_ROWS={per_snapshot_counts.get(sid, 0)}")
    print(f"ALL_ROWS={len(all_rows)}")
    print(f"CSV={combined_path.name} SIZE={combined_path.stat().st_size}")
    for path in per_snapshot_paths:
        print(f"CSV={path.name} SIZE={path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
