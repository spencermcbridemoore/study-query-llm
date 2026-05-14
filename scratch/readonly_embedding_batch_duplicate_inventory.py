"""Read-only inventory of same-key duplicate embedding_batch groups.

Phase 2 deliverable script for per-pair collision cleanup:
- exports duplicate sets grouped by canonical embedding-batch key
- includes per-candidate reference counts from provenanced_runs/group_links
- writes JSON + CSV under campaign pre_cleanup/inventory artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_STAMP = "20260514T085641Z"

GROUP_KEY_FIELDS = (
    "source_dataframe_group_id",
    "entry_max",
    "embedding_engine",
    "provider",
    "representation",
    "dataset_key",
    "key_version",
)


def _load_database_url() -> str:
    env = dotenv_values(ROOT / ".env")
    url = (
        os.environ.get("DATABASE_URL")
        or env.get("DATABASE_URL")
        or os.environ.get("CANONICAL_DATABASE_URL")
        or env.get("CANONICAL_DATABASE_URL")
        or ""
    ).strip()
    if not url:
        raise RuntimeError("DATABASE_URL or CANONICAL_DATABASE_URL is required")
    return url


def _query_duplicate_sets(conn) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT
            (g.metadata_json->>'source_dataframe_group_id')::int AS source_dataframe_group_id,
            (g.metadata_json->>'entry_max')::int AS entry_max,
            COALESCE(g.metadata_json->>'embedding_engine', g.metadata_json->>'deployment', '') AS embedding_engine,
            COALESCE(g.metadata_json->>'provider', '') AS provider,
            COALESCE(g.metadata_json->>'representation', '') AS representation,
            COALESCE(g.metadata_json->>'dataset_key', '') AS dataset_key,
            COALESCE(g.metadata_json->>'key_version', '') AS key_version,
            ARRAY_AGG(g.id ORDER BY g.id ASC) AS batch_ids,
            COUNT(*)::int AS candidate_count
        FROM groups g
        WHERE g.group_type = 'embedding_batch'
          AND COALESCE(g.metadata_json->>'source_dataframe_group_id', '') ~ '^[0-9]+$'
          AND COALESCE(g.metadata_json->>'entry_max', '') ~ '^[0-9]+$'
        GROUP BY
            (g.metadata_json->>'source_dataframe_group_id')::int,
            (g.metadata_json->>'entry_max')::int,
            COALESCE(g.metadata_json->>'embedding_engine', g.metadata_json->>'deployment', ''),
            COALESCE(g.metadata_json->>'provider', ''),
            COALESCE(g.metadata_json->>'representation', ''),
            COALESCE(g.metadata_json->>'dataset_key', ''),
            COALESCE(g.metadata_json->>'key_version', '')
        HAVING COUNT(*) > 1
        ORDER BY
            COUNT(*) DESC,
            (g.metadata_json->>'source_dataframe_group_id')::int ASC,
            (g.metadata_json->>'entry_max')::int ASC,
            COALESCE(g.metadata_json->>'embedding_engine', g.metadata_json->>'deployment', '') ASC
        """
    )
    rows = conn.execute(sql).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source_dataframe_group_id": int(row["source_dataframe_group_id"]),
                "entry_max": int(row["entry_max"]),
                "embedding_engine": str(row.get("embedding_engine") or ""),
                "provider": str(row.get("provider") or ""),
                "representation": str(row.get("representation") or ""),
                "dataset_key": str(row.get("dataset_key") or ""),
                "key_version": str(row.get("key_version") or ""),
                "batch_ids": [int(x) for x in list(row.get("batch_ids") or [])],
                "candidate_count": int(row["candidate_count"]),
            }
        )
    return out


def _query_candidate_metadata(conn, batch_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not batch_ids:
        return {}
    sql = text(
        """
        SELECT
            g.id AS batch_id,
            g.created_at AS created_at,
            g.name AS name,
            (g.metadata_json->>'source_dataframe_group_id')::int AS source_dataframe_group_id,
            (g.metadata_json->>'entry_max')::int AS entry_max,
            COALESCE(g.metadata_json->>'embedding_engine', g.metadata_json->>'deployment', '') AS embedding_engine,
            COALESCE(g.metadata_json->>'provider', '') AS provider,
            COALESCE(g.metadata_json->>'representation', '') AS representation,
            COALESCE(g.metadata_json->>'dataset_key', '') AS dataset_key,
            COALESCE(g.metadata_json->>'key_version', '') AS key_version
        FROM groups g
        WHERE g.id = ANY(:batch_ids)
        ORDER BY g.id ASC
        """
    )
    rows = conn.execute(sql, {"batch_ids": batch_ids}).mappings().all()
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        batch_id = int(row["batch_id"])
        out[batch_id] = {
            "batch_id": batch_id,
            "created_at": (
                row["created_at"].isoformat()
                if hasattr(row.get("created_at"), "isoformat")
                else str(row.get("created_at") or "")
            ),
            "name": str(row.get("name") or ""),
            "source_dataframe_group_id": int(row["source_dataframe_group_id"]),
            "entry_max": int(row["entry_max"]),
            "embedding_engine": str(row.get("embedding_engine") or ""),
            "provider": str(row.get("provider") or ""),
            "representation": str(row.get("representation") or ""),
            "dataset_key": str(row.get("dataset_key") or ""),
            "key_version": str(row.get("key_version") or ""),
        }
    return out


def _query_ref_counts(conn, batch_ids: list[int]) -> dict[int, dict[str, int]]:
    if not batch_ids:
        return {}
    baseline = {
        int(batch_id): {
            "provenanced_runs_metadata_ref_count": 0,
            "provenanced_runs_config_ref_count": 0,
            "provenanced_runs_source_group_ref_count": 0,
            "group_links_parent_ref_count": 0,
            "group_links_child_ref_count": 0,
        }
        for batch_id in batch_ids
    }

    metadata_sql = text(
        """
        SELECT
            (pr.metadata_json->>'embedding_batch_group_id')::int AS batch_id,
            COUNT(*)::int AS ref_count
        FROM provenanced_runs pr
        WHERE COALESCE(pr.metadata_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
          AND (pr.metadata_json->>'embedding_batch_group_id')::int = ANY(:batch_ids)
        GROUP BY (pr.metadata_json->>'embedding_batch_group_id')::int
        """
    )
    for row in conn.execute(metadata_sql, {"batch_ids": batch_ids}).mappings().all():
        baseline[int(row["batch_id"])]["provenanced_runs_metadata_ref_count"] = int(
            row["ref_count"]
        )

    config_sql = text(
        """
        SELECT
            (pr.config_json->>'embedding_batch_group_id')::int AS batch_id,
            COUNT(*)::int AS ref_count
        FROM provenanced_runs pr
        WHERE COALESCE(pr.config_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
          AND (pr.config_json->>'embedding_batch_group_id')::int = ANY(:batch_ids)
        GROUP BY (pr.config_json->>'embedding_batch_group_id')::int
        """
    )
    for row in conn.execute(config_sql, {"batch_ids": batch_ids}).mappings().all():
        baseline[int(row["batch_id"])]["provenanced_runs_config_ref_count"] = int(
            row["ref_count"]
        )

    source_sql = text(
        """
        SELECT
            pr.source_group_id AS batch_id,
            COUNT(*)::int AS ref_count
        FROM provenanced_runs pr
        JOIN groups g ON g.id = pr.source_group_id
        WHERE pr.source_group_id = ANY(:batch_ids)
          AND g.group_type = 'embedding_batch'
        GROUP BY pr.source_group_id
        """
    )
    for row in conn.execute(source_sql, {"batch_ids": batch_ids}).mappings().all():
        baseline[int(row["batch_id"])]["provenanced_runs_source_group_ref_count"] = int(
            row["ref_count"]
        )

    parent_sql = text(
        """
        SELECT
            gl.parent_group_id AS batch_id,
            COUNT(*)::int AS ref_count
        FROM group_links gl
        WHERE gl.parent_group_id = ANY(:batch_ids)
        GROUP BY gl.parent_group_id
        """
    )
    for row in conn.execute(parent_sql, {"batch_ids": batch_ids}).mappings().all():
        baseline[int(row["batch_id"])]["group_links_parent_ref_count"] = int(
            row["ref_count"]
        )

    child_sql = text(
        """
        SELECT
            gl.child_group_id AS batch_id,
            COUNT(*)::int AS ref_count
        FROM group_links gl
        WHERE gl.child_group_id = ANY(:batch_ids)
        GROUP BY gl.child_group_id
        """
    )
    for row in conn.execute(child_sql, {"batch_ids": batch_ids}).mappings().all():
        baseline[int(row["batch_id"])]["group_links_child_ref_count"] = int(
            row["ref_count"]
        )

    for batch_id, refs in baseline.items():
        refs["provenanced_runs_total_ref_count"] = (
            int(refs["provenanced_runs_metadata_ref_count"])
            + int(refs["provenanced_runs_config_ref_count"])
            + int(refs["provenanced_runs_source_group_ref_count"])
        )
        refs["group_links_total_ref_count"] = int(refs["group_links_parent_ref_count"]) + int(
            refs["group_links_child_ref_count"]
        )
        refs["total_ref_count"] = int(refs["provenanced_runs_total_ref_count"]) + int(
            refs["group_links_total_ref_count"]
        )

    return baseline


def _build_rows(duplicate_sets: list[dict[str, Any]], candidate_map: dict[int, dict[str, Any]], ref_counts: dict[int, dict[str, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duplicate_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for idx, dset in enumerate(duplicate_sets, start=1):
        set_id = f"dup_set_{idx:04d}"
        key_payload = {field: dset[field] for field in GROUP_KEY_FIELDS}
        candidates: list[dict[str, Any]] = []
        for batch_id in dset["batch_ids"]:
            candidate = dict(candidate_map.get(batch_id) or {"batch_id": int(batch_id)})
            refs = ref_counts.get(int(batch_id), {})
            candidate.update(refs)
            candidates.append(candidate)

            csv_rows.append(
                {
                    "duplicate_set_id": set_id,
                    **key_payload,
                    **candidate,
                }
            )
        duplicate_rows.append(
            {
                "duplicate_set_id": set_id,
                "group_key": key_payload,
                "candidate_count": int(dset["candidate_count"]),
                "batch_ids": [int(x) for x in dset["batch_ids"]],
                "candidates": candidates,
            }
        )
    return duplicate_rows, csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export duplicate embedding_batch inventory (read-only).")
    parser.add_argument(
        "--run-stamp",
        default=DEFAULT_RUN_STAMP,
        help="Campaign run stamp (default: 20260514T085641Z).",
    )
    args = parser.parse_args()

    output_dir = (
        ROOT
        / "experimental_results"
        / "backfill_per_pair"
        / str(args.run_stamp)
        / "pre_cleanup"
        / "inventory"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "duplicate_sets.json"
    csv_path = output_dir / "duplicate_sets.csv"

    db_url = _load_database_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    with engine.connect() as conn:
        duplicate_sets = _query_duplicate_sets(conn)
        batch_ids = sorted({int(batch_id) for row in duplicate_sets for batch_id in row["batch_ids"]})
        candidate_map = _query_candidate_metadata(conn, batch_ids)
        ref_counts = _query_ref_counts(conn, batch_ids)
        duplicate_rows, csv_rows = _build_rows(duplicate_sets, candidate_map, ref_counts)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_stamp": str(args.run_stamp),
        "group_key_fields": list(GROUP_KEY_FIELDS),
        "duplicate_set_count": int(len(duplicate_rows)),
        "candidate_row_count": int(len(csv_rows)),
        "duplicate_sets": duplicate_rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_fieldnames = [
        "duplicate_set_id",
        *GROUP_KEY_FIELDS,
        "batch_id",
        "created_at",
        "name",
        "provenanced_runs_metadata_ref_count",
        "provenanced_runs_config_ref_count",
        "provenanced_runs_source_group_ref_count",
        "provenanced_runs_total_ref_count",
        "group_links_parent_ref_count",
        "group_links_child_ref_count",
        "group_links_total_ref_count",
        "total_ref_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: row.get(key, "") for key in csv_fieldnames})

    print(f"OUTPUT_JSON={json_path}")
    print(f"OUTPUT_CSV={csv_path}")
    print(f"DUPLICATE_SET_COUNT={len(duplicate_rows)}")
    print(f"CANDIDATE_ROW_COUNT={len(csv_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

