"""One-off cleanup for same-key duplicate embedding_batch groups.

Phase 3 tool contract:
- supports --dry-run and --apply modes
- --apply requires --receipt-path and explicit --confirm-token
- selects canonical survivor per duplicate set with deterministic policy
- migrates references to survivor before deleting duplicate group rows
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_STAMP = "20260514T085641Z"
DEFAULT_TARGET_LINEAGES = ("5:2086", "8:324")
REQUIRED_CONFIRMATION_TOKEN = "APPLY_EMBEDDING_BATCH_SAME_KEY_DUPLICATES"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _parse_lineages(raw_values: list[str]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for raw in raw_values:
        token = str(raw).strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid --target-lineage value {token!r}, expected <sdf_id>:<entry_max>")
        try:
            sdf_id = int(parts[0])
            entry_max = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid --target-lineage value {token!r}, expected integers") from exc
        if sdf_id <= 0 or entry_max <= 0:
            raise ValueError(f"invalid --target-lineage value {token!r}, expected positive integers")
        out.add((sdf_id, entry_max))
    if not out:
        raise ValueError("at least one target lineage is required")
    return out


def _inventory_path_for(run_stamp: str) -> Path:
    return (
        ROOT
        / "experimental_results"
        / "backfill_per_pair"
        / run_stamp
        / "pre_cleanup"
        / "inventory"
        / "duplicate_sets.json"
    )


def _default_report_path(*, run_stamp: str, mode: str) -> Path:
    base = ROOT / "experimental_results" / "backfill_per_pair" / run_stamp
    if mode == "dry_run":
        return (
            base
            / "pre_cleanup"
            / "inventory"
            / "dry_run"
            / f"cleanup_dry_run_report_{_utc_stamp()}.json"
        )
    return base / "post_cleanup" / "apply_report.json"


def _load_duplicate_inventory(inventory_path: Path) -> dict[str, Any]:
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"duplicate inventory not found at {inventory_path}. Run Phase 2 inventory first."
        )
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("inventory payload must be a JSON object")
    return payload


def _receipt_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "exists": False,
            "valid": False,
            "status": "not_provided",
        }
    if not path.exists():
        return {
            "provided": True,
            "exists": False,
            "valid": False,
            "status": "missing",
            "path": str(path),
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    status = str((raw or {}).get("status") or "")
    valid = status == "ok"
    return {
        "provided": True,
        "exists": True,
        "valid": valid,
        "status": status,
        "path": str(path),
    }


def _progress(message: str) -> None:
    print(message, flush=True)


def _zero_required_counts() -> dict[str, int]:
    return {
        "provenanced_runs_metadata_ref_count": 0,
        "provenanced_runs_config_ref_count": 0,
        "provenanced_runs_source_group_ref_count": 0,
        "group_links_parent_ref_count": 0,
        "group_links_child_ref_count": 0,
        "call_artifacts_group_ref_count": 0,
        "provenanced_runs_total_ref_count": 0,
        "group_links_total_ref_count": 0,
        "required_ref_total": 0,
    }


def _build_bulk_reference_index(conn, candidate_ids: list[int]) -> dict[str, Any]:
    """Load all reference paths once and index in memory by embedding_batch id."""
    candidate_ids_sorted = sorted({int(x) for x in candidate_ids})
    candidate_id_set = set(candidate_ids_sorted)
    analysis_rows_by_batch: dict[int, list[dict[str, Any]]] = {
        batch_id: [] for batch_id in candidate_ids_sorted
    }
    required_counts_by_batch: dict[int, dict[str, int]] = {
        batch_id: _zero_required_counts() for batch_id in candidate_ids_sorted
    }
    stats: dict[str, int] = {
        "candidate_batch_count": len(candidate_ids_sorted),
        "provenanced_runs_rows_scanned": 0,
        "analysis_reference_rows_indexed": 0,
    }
    if not candidate_ids_sorted:
        return {
            "analysis_rows_by_batch": analysis_rows_by_batch,
            "required_counts_by_batch": required_counts_by_batch,
            "stats": stats,
        }

    _progress(
        "loading provenanced_runs reference index in bulk "
        f"(candidate batches={len(candidate_ids_sorted)})"
    )
    t0 = time.perf_counter()
    provenanced_rows = conn.execute(
        text(
            """
            SELECT
                pr.id AS provenanced_run_id,
                pr.run_key AS run_key,
                pr.created_at AS created_at,
                pr.run_status AS run_status,
                pr.run_kind AS run_kind,
                COALESCE(pr.metadata_json->>'execution_role', '') AS execution_role,
                CASE
                  WHEN COALESCE(pr.metadata_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
                  THEN (pr.metadata_json->>'embedding_batch_group_id')::int
                  ELSE NULL
                END AS metadata_batch_id,
                CASE
                  WHEN COALESCE(pr.config_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
                  THEN (pr.config_json->>'embedding_batch_group_id')::int
                  ELSE NULL
                END AS config_batch_id,
                pr.source_group_id AS source_group_id,
                COALESCE(sg.group_type, '') AS source_group_type
            FROM provenanced_runs pr
            LEFT JOIN groups sg ON sg.id = pr.source_group_id
            """
        )
    ).mappings().all()
    elapsed = time.perf_counter() - t0
    stats["provenanced_runs_rows_scanned"] = int(len(provenanced_rows))
    _progress(
        f"loaded {len(provenanced_rows)} provenanced_runs rows in {elapsed:.2f}s"
    )

    def _is_analysis_ref_row(row: dict[str, Any]) -> bool:
        run_status = str(row.get("run_status") or "")
        run_kind = str(row.get("run_kind") or "")
        execution_role = str(row.get("execution_role") or "")
        run_key = str(row.get("run_key") or "")
        return (
            run_status == "completed"
            and (
                run_kind == "analysis_execution"
                or (run_kind == "execution" and execution_role == "analysis_execution")
            )
            and not run_key.startswith("backfill_exec__")
        )

    for row in provenanced_rows:
        row_base = {
            "provenanced_run_id": int(row["provenanced_run_id"]),
            "run_key": str(row.get("run_key") or ""),
            "created_at": (
                row["created_at"].isoformat()
                if hasattr(row.get("created_at"), "isoformat")
                else str(row.get("created_at") or "")
            ),
        }
        metadata_batch_id = row.get("metadata_batch_id")
        config_batch_id = row.get("config_batch_id")
        source_group_id = row.get("source_group_id")
        source_group_type = str(row.get("source_group_type") or "")
        is_analysis_ref = _is_analysis_ref_row(row)

        if metadata_batch_id is not None:
            metadata_batch_id = int(metadata_batch_id)
            if metadata_batch_id in candidate_id_set:
                required_counts_by_batch[metadata_batch_id][
                    "provenanced_runs_metadata_ref_count"
                ] += 1
                if is_analysis_ref:
                    analysis_rows_by_batch[metadata_batch_id].append(
                        {
                            **row_base,
                            "batch_id": metadata_batch_id,
                            "ref_field": "provenanced_runs.metadata_json.embedding_batch_group_id",
                        }
                    )
                    stats["analysis_reference_rows_indexed"] += 1

        if config_batch_id is not None:
            config_batch_id = int(config_batch_id)
            if config_batch_id in candidate_id_set:
                required_counts_by_batch[config_batch_id][
                    "provenanced_runs_config_ref_count"
                ] += 1
                if is_analysis_ref:
                    analysis_rows_by_batch[config_batch_id].append(
                        {
                            **row_base,
                            "batch_id": config_batch_id,
                            "ref_field": "provenanced_runs.config_json.embedding_batch_group_id",
                        }
                    )
                    stats["analysis_reference_rows_indexed"] += 1

        if (
            source_group_id is not None
            and source_group_type == "embedding_batch"
            and int(source_group_id) in candidate_id_set
        ):
            source_group_id_int = int(source_group_id)
            required_counts_by_batch[source_group_id_int][
                "provenanced_runs_source_group_ref_count"
            ] += 1
            if is_analysis_ref:
                analysis_rows_by_batch[source_group_id_int].append(
                    {
                        **row_base,
                        "batch_id": source_group_id_int,
                        "ref_field": "provenanced_runs.source_group_id",
                    }
                )
                stats["analysis_reference_rows_indexed"] += 1

    _progress(
        "loading group_links parent reference counts in bulk "
        f"(candidate batches={len(candidate_ids_sorted)})"
    )
    parent_rows = conn.execute(
        text(
            """
            SELECT
                gl.parent_group_id AS batch_id,
                COUNT(*)::int AS ref_count
            FROM group_links gl
            WHERE gl.parent_group_id = ANY(:candidate_ids)
            GROUP BY gl.parent_group_id
            """
        ),
        {"candidate_ids": candidate_ids_sorted},
    ).mappings().all()
    for row in parent_rows:
        batch_id = int(row["batch_id"])
        required_counts_by_batch[batch_id]["group_links_parent_ref_count"] = int(
            row["ref_count"]
        )

    _progress(
        "loading group_links child reference counts in bulk "
        f"(candidate batches={len(candidate_ids_sorted)})"
    )
    child_rows = conn.execute(
        text(
            """
            SELECT
                gl.child_group_id AS batch_id,
                COUNT(*)::int AS ref_count
            FROM group_links gl
            WHERE gl.child_group_id = ANY(:candidate_ids)
            GROUP BY gl.child_group_id
            """
        ),
        {"candidate_ids": candidate_ids_sorted},
    ).mappings().all()
    for row in child_rows:
        batch_id = int(row["batch_id"])
        required_counts_by_batch[batch_id]["group_links_child_ref_count"] = int(
            row["ref_count"]
        )

    _progress(
        "loading call_artifacts group_id reference counts in bulk "
        f"(candidate batches={len(candidate_ids_sorted)})"
    )
    artifact_rows = conn.execute(
        text(
            """
            SELECT
                (ca.metadata_json->>'group_id')::int AS batch_id,
                COUNT(*)::int AS ref_count
            FROM call_artifacts ca
            WHERE COALESCE(ca.metadata_json->>'group_id', '') ~ '^[0-9]+$'
              AND (ca.metadata_json->>'group_id')::int = ANY(:candidate_ids)
            GROUP BY (ca.metadata_json->>'group_id')::int
            """
        ),
        {"candidate_ids": candidate_ids_sorted},
    ).mappings().all()
    for row in artifact_rows:
        batch_id = int(row["batch_id"])
        required_counts_by_batch[batch_id]["call_artifacts_group_ref_count"] = int(
            row["ref_count"]
        )

    for batch_id in candidate_ids_sorted:
        counts = required_counts_by_batch[batch_id]
        counts["provenanced_runs_total_ref_count"] = (
            int(counts["provenanced_runs_metadata_ref_count"])
            + int(counts["provenanced_runs_config_ref_count"])
            + int(counts["provenanced_runs_source_group_ref_count"])
        )
        counts["group_links_total_ref_count"] = int(
            counts["group_links_parent_ref_count"]
        ) + int(counts["group_links_child_ref_count"])
        counts["required_ref_total"] = int(counts["provenanced_runs_total_ref_count"]) + int(
            counts["group_links_total_ref_count"]
        )
        analysis_rows_by_batch[batch_id].sort(
            key=lambda row: (
                str(row.get("created_at") or ""),
                int(row.get("provenanced_run_id") or 0),
                str(row.get("ref_field") or ""),
            )
        )

    return {
        "analysis_rows_by_batch": analysis_rows_by_batch,
        "required_counts_by_batch": required_counts_by_batch,
        "stats": stats,
    }


def _analysis_reference_rows(conn, candidate_ids: list[int]) -> list[dict[str, Any]]:
    if not candidate_ids:
        return []
    sql = text(
        """
        WITH refs AS (
            SELECT
                pr.id AS provenanced_run_id,
                pr.run_key AS run_key,
                pr.created_at AS created_at,
                (pr.metadata_json->>'embedding_batch_group_id')::int AS batch_id,
                'provenanced_runs.metadata_json.embedding_batch_group_id' AS ref_field
            FROM provenanced_runs pr
            WHERE COALESCE(pr.metadata_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
              AND pr.run_status = 'completed'
              AND (
                pr.run_kind = 'analysis_execution'
                OR (
                  pr.run_kind = 'execution'
                  AND COALESCE(pr.metadata_json->>'execution_role', '') = 'analysis_execution'
                )
              )
              AND COALESCE(pr.run_key, '') NOT LIKE 'backfill_exec__%'
            UNION ALL
            SELECT
                pr.id AS provenanced_run_id,
                pr.run_key AS run_key,
                pr.created_at AS created_at,
                (pr.config_json->>'embedding_batch_group_id')::int AS batch_id,
                'provenanced_runs.config_json.embedding_batch_group_id' AS ref_field
            FROM provenanced_runs pr
            WHERE COALESCE(pr.config_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
              AND pr.run_status = 'completed'
              AND (
                pr.run_kind = 'analysis_execution'
                OR (
                  pr.run_kind = 'execution'
                  AND COALESCE(pr.metadata_json->>'execution_role', '') = 'analysis_execution'
                )
              )
              AND COALESCE(pr.run_key, '') NOT LIKE 'backfill_exec__%'
            UNION ALL
            SELECT
                pr.id AS provenanced_run_id,
                pr.run_key AS run_key,
                pr.created_at AS created_at,
                pr.source_group_id AS batch_id,
                'provenanced_runs.source_group_id' AS ref_field
            FROM provenanced_runs pr
            JOIN groups g ON g.id = pr.source_group_id
            WHERE pr.source_group_id IS NOT NULL
              AND g.group_type = 'embedding_batch'
              AND pr.run_status = 'completed'
              AND (
                pr.run_kind = 'analysis_execution'
                OR (
                  pr.run_kind = 'execution'
                  AND COALESCE(pr.metadata_json->>'execution_role', '') = 'analysis_execution'
                )
              )
              AND COALESCE(pr.run_key, '') NOT LIKE 'backfill_exec__%'
        )
        SELECT
            provenanced_run_id,
            run_key,
            created_at,
            batch_id,
            ref_field
        FROM refs
        WHERE batch_id = ANY(:candidate_ids)
        ORDER BY created_at ASC, provenanced_run_id ASC
        """
    )
    rows = conn.execute(sql, {"candidate_ids": candidate_ids}).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "provenanced_run_id": int(row["provenanced_run_id"]),
                "run_key": str(row.get("run_key") or ""),
                "created_at": (
                    row["created_at"].isoformat()
                    if hasattr(row.get("created_at"), "isoformat")
                    else str(row.get("created_at") or "")
                ),
                "batch_id": int(row["batch_id"]),
                "ref_field": str(row.get("ref_field") or ""),
            }
        )
    return out


def _required_ref_counts(conn, batch_id: int) -> dict[str, int]:
    params = {"batch_id": int(batch_id)}
    metadata_count = int(
        conn.execute(
            text(
                """
                SELECT COUNT(*)::int
                FROM provenanced_runs pr
                WHERE COALESCE(pr.metadata_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
                  AND (pr.metadata_json->>'embedding_batch_group_id')::int = :batch_id
                """
            ),
            params,
        ).scalar_one()
    )
    config_count = int(
        conn.execute(
            text(
                """
                SELECT COUNT(*)::int
                FROM provenanced_runs pr
                WHERE COALESCE(pr.config_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
                  AND (pr.config_json->>'embedding_batch_group_id')::int = :batch_id
                """
            ),
            params,
        ).scalar_one()
    )
    source_count = int(
        conn.execute(
            text(
                """
                SELECT COUNT(*)::int
                FROM provenanced_runs pr
                JOIN groups g ON g.id = pr.source_group_id
                WHERE pr.source_group_id = :batch_id
                  AND g.group_type = 'embedding_batch'
                """
            ),
            params,
        ).scalar_one()
    )
    parent_link_count = int(
        conn.execute(
            text(
                """
                SELECT COUNT(*)::int
                FROM group_links gl
                WHERE gl.parent_group_id = :batch_id
                """
            ),
            params,
        ).scalar_one()
    )
    child_link_count = int(
        conn.execute(
            text(
                """
                SELECT COUNT(*)::int
                FROM group_links gl
                WHERE gl.child_group_id = :batch_id
                """
            ),
            params,
        ).scalar_one()
    )
    artifact_group_count = int(
        conn.execute(
            text(
                """
                SELECT COUNT(*)::int
                FROM call_artifacts ca
                WHERE COALESCE(ca.metadata_json->>'group_id', '') ~ '^[0-9]+$'
                  AND (ca.metadata_json->>'group_id')::int = :batch_id
                """
            ),
            params,
        ).scalar_one()
    )
    required_total = (
        metadata_count + config_count + source_count + parent_link_count + child_link_count
    )
    return {
        "provenanced_runs_metadata_ref_count": metadata_count,
        "provenanced_runs_config_ref_count": config_count,
        "provenanced_runs_source_group_ref_count": source_count,
        "group_links_parent_ref_count": parent_link_count,
        "group_links_child_ref_count": child_link_count,
        "call_artifacts_group_ref_count": artifact_group_count,
        "required_ref_total": required_total,
    }


def _migrate_refs_to_survivor(conn, *, from_batch_id: int, to_batch_id: int) -> dict[str, int]:
    params = {"old_id": int(from_batch_id), "new_id": int(to_batch_id)}
    metadata_updated = conn.execute(
        text(
            """
            UPDATE provenanced_runs
            SET metadata_json = jsonb_set(
                COALESCE(metadata_json, '{}'::jsonb),
                '{embedding_batch_group_id}',
                to_jsonb(CAST(:new_id AS int)),
                true
            )
            WHERE COALESCE(metadata_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
              AND (metadata_json->>'embedding_batch_group_id')::int = :old_id
            """
        ),
        params,
    ).rowcount
    config_updated = conn.execute(
        text(
            """
            UPDATE provenanced_runs
            SET config_json = jsonb_set(
                COALESCE(config_json, '{}'::jsonb),
                '{embedding_batch_group_id}',
                to_jsonb(CAST(:new_id AS int)),
                true
            )
            WHERE COALESCE(config_json->>'embedding_batch_group_id', '') ~ '^[0-9]+$'
              AND (config_json->>'embedding_batch_group_id')::int = :old_id
            """
        ),
        params,
    ).rowcount
    source_updated = conn.execute(
        text(
            """
            UPDATE provenanced_runs
            SET source_group_id = :new_id
            WHERE source_group_id = :old_id
            """
        ),
        params,
    ).rowcount
    parent_links_updated = conn.execute(
        text(
            """
            UPDATE group_links
            SET parent_group_id = :new_id
            WHERE parent_group_id = :old_id
            """
        ),
        params,
    ).rowcount
    child_links_updated = conn.execute(
        text(
            """
            UPDATE group_links
            SET child_group_id = :new_id
            WHERE child_group_id = :old_id
            """
        ),
        params,
    ).rowcount
    # Avoid survivor self-loop links introduced by parent/child rewrites.
    self_links_deleted = conn.execute(
        text(
            """
            DELETE FROM group_links
            WHERE parent_group_id = :new_id
              AND child_group_id = :new_id
            """
        ),
        params,
    ).rowcount
    artifact_group_updated = conn.execute(
        text(
            """
            UPDATE call_artifacts
            SET metadata_json = jsonb_set(
                COALESCE(metadata_json, '{}'::jsonb),
                '{group_id}',
                to_jsonb(CAST(:new_id AS int)),
                true
            )
            WHERE COALESCE(metadata_json->>'group_id', '') ~ '^[0-9]+$'
              AND (metadata_json->>'group_id')::int = :old_id
            """
        ),
        params,
    ).rowcount
    return {
        "metadata_embedding_batch_group_id_updates": int(metadata_updated or 0),
        "config_embedding_batch_group_id_updates": int(config_updated or 0),
        "source_group_id_updates": int(source_updated or 0),
        "group_links_parent_updates": int(parent_links_updated or 0),
        "group_links_child_updates": int(child_links_updated or 0),
        "group_links_self_loops_deleted": int(self_links_deleted or 0),
        "call_artifacts_group_id_updates": int(artifact_group_updated or 0),
    }


def _delete_embedding_batch_group(conn, batch_id: int) -> int:
    deleted = conn.execute(
        text(
            """
            DELETE FROM groups
            WHERE id = :batch_id
              AND group_type = 'embedding_batch'
            """
        ),
        {"batch_id": int(batch_id)},
    ).rowcount
    return int(deleted or 0)


def _select_survivor(
    *,
    candidate_ids: list[int],
    analysis_ref_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    referenced_ids = sorted({int(row["batch_id"]) for row in analysis_ref_rows})
    if len(referenced_ids) > 1:
        return {
            "selection_status": "conflict_multiple_referenced_ids",
            "survivor_batch_id": None,
            "referenced_candidate_batch_ids": referenced_ids,
            "selection_reason": "multiple completed analysis references found in duplicate set",
        }
    if len(referenced_ids) == 1:
        survivor = int(referenced_ids[0])
        return {
            "selection_status": "selected_from_completed_analysis_reference",
            "survivor_batch_id": survivor,
            "referenced_candidate_batch_ids": referenced_ids,
            "selection_reason": "single completed analysis reference found; keep referenced batch",
        }
    survivor = int(max(candidate_ids))
    return {
        "selection_status": "selected_newest_no_completed_reference",
        "survivor_batch_id": survivor,
        "referenced_candidate_batch_ids": [],
        "selection_reason": "no completed analysis reference found; keep newest batch id",
    }


def _set_in_target_scope(group_key: dict[str, Any], target_lineages: set[tuple[int, int]]) -> bool:
    try:
        key = (int(group_key["source_dataframe_group_id"]), int(group_key["entry_max"]))
    except (KeyError, TypeError, ValueError):
        return False
    return key in target_lineages


def _report_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cleanup same-key duplicate embedding_batch groups with dry-run/apply modes."
    )
    parser.add_argument("--run-stamp", default=DEFAULT_RUN_STAMP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--target-lineage",
        action="append",
        default=[],
        help="Lineage key in <source_dataframe_group_id>:<entry_max> form (repeatable).",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Output report path. Default: campaign dry_run or post_cleanup apply path.",
    )
    parser.add_argument(
        "--receipt-path",
        default=None,
        help="Backup receipt path. Required for --apply.",
    )
    parser.add_argument(
        "--confirm-token",
        default=None,
        help=f"Required with --apply. Must equal {REQUIRED_CONFIRMATION_TOKEN!r}.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if bool(args.dry_run) == bool(args.apply):
        raise SystemExit("Exactly one of --dry-run or --apply must be provided.")

    mode = "apply" if args.apply else "dry_run"
    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else _default_report_path(run_stamp=str(args.run_stamp), mode=mode)
    )
    receipt_path = Path(args.receipt_path).resolve() if args.receipt_path else None
    target_lineages = _parse_lineages(
        args.target_lineage if args.target_lineage else list(DEFAULT_TARGET_LINEAGES)
    )

    if args.apply:
        if receipt_path is None:
            raise SystemExit("--apply requires --receipt-path.")
        if str(args.confirm_token or "").strip() != REQUIRED_CONFIRMATION_TOKEN:
            raise SystemExit(
                "--apply requires --confirm-token matching "
                f"{REQUIRED_CONFIRMATION_TOKEN!r}."
            )

    receipt_info = _receipt_status(receipt_path)
    if args.apply and not bool(receipt_info.get("valid")):
        raise SystemExit(
            "Backup receipt validation failed; apply is blocked. "
            f"receipt_status={receipt_info}"
        )

    inventory_path = _inventory_path_for(str(args.run_stamp))
    inventory_payload = _load_duplicate_inventory(inventory_path)
    inventory_sets = list(inventory_payload.get("duplicate_sets") or [])
    target_inventory_sets = [
        raw_set
        for raw_set in inventory_sets
        if _set_in_target_scope(dict(raw_set.get("group_key") or {}), target_lineages)
    ]
    all_candidate_ids = sorted(
        {
            int(batch_id)
            for raw_set in target_inventory_sets
            for batch_id in list(raw_set.get("batch_ids") or [])
        }
    )

    db_url = _load_database_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    set_reports: list[dict[str, Any]] = []
    total_delete_candidates = 0
    total_applied_deletes = 0
    conflicts = 0
    blocked = 0
    applied_sets = 0

    with engine.connect() as conn:
        reference_index = _build_bulk_reference_index(conn, all_candidate_ids)
    analysis_rows_by_batch: dict[int, list[dict[str, Any]]] = dict(
        reference_index["analysis_rows_by_batch"]
    )
    required_counts_by_batch: dict[int, dict[str, int]] = dict(
        reference_index["required_counts_by_batch"]
    )
    reference_index_stats = dict(reference_index["stats"])

    total_sets = len(target_inventory_sets)
    for set_idx, raw_set in enumerate(target_inventory_sets, start=1):
        duplicate_set_id = str(raw_set.get("duplicate_set_id") or "")
        group_key = dict(raw_set.get("group_key") or {})
        candidate_ids = sorted(int(x) for x in list(raw_set.get("batch_ids") or []))
        if len(candidate_ids) <= 1:
            continue
        _progress(
            "duplicate set "
            f"{set_idx}/{total_sets}: key={group_key} candidates={candidate_ids}"
        )

        analysis_ref_rows: list[dict[str, Any]] = []
        for batch_id in candidate_ids:
            _progress(f"checking references for batch {int(batch_id)}")
            analysis_ref_rows.extend(list(analysis_rows_by_batch.get(int(batch_id), [])))
        analysis_ref_rows.sort(
            key=lambda row: (
                str(row.get("created_at") or ""),
                int(row.get("provenanced_run_id") or 0),
                str(row.get("ref_field") or ""),
            )
        )

        survivor_decision = _select_survivor(
            candidate_ids=candidate_ids,
            analysis_ref_rows=analysis_ref_rows,
        )
        survivor_id = survivor_decision.get("survivor_batch_id")
        delete_candidates = (
            [batch_id for batch_id in candidate_ids if batch_id != int(survivor_id)]
            if survivor_id is not None
            else []
        )
        total_delete_candidates += len(delete_candidates)

        candidate_reports: list[dict[str, Any]] = []
        for batch_id in candidate_ids:
            ref_counts = dict(required_counts_by_batch.get(int(batch_id)) or _zero_required_counts())
            candidate_reports.append(
                {
                    "batch_id": int(batch_id),
                    "is_survivor": bool(survivor_id is not None and int(batch_id) == int(survivor_id)),
                    "delete_candidate": bool(batch_id in delete_candidates),
                    "required_reference_counts": ref_counts,
                    "reference_migration_plan": (
                        {
                            "to_survivor_batch_id": int(survivor_id),
                            "actions": [
                                "migrate provenanced_runs.metadata_json.embedding_batch_group_id",
                                "migrate provenanced_runs.config_json.embedding_batch_group_id",
                                "migrate provenanced_runs.source_group_id",
                                "migrate group_links parent/child endpoints",
                                "migrate call_artifacts.metadata_json.group_id",
                            ],
                        }
                        if batch_id in delete_candidates and survivor_id is not None
                        else None
                    ),
                }
            )

        set_status = "planned"
        set_reason = "ready"
        apply_actions: list[dict[str, Any]] = []

        if survivor_decision["selection_status"] == "conflict_multiple_referenced_ids":
            set_status = "blocked_conflict"
            set_reason = "multiple referenced candidate batches; operator adjudication required"
            conflicts += 1
            blocked += 1
        elif args.apply:
            # Apply per duplicate set in an isolated transaction.
            with engine.begin() as conn:
                try:
                    for delete_batch_id in delete_candidates:
                        migration_counts = _migrate_refs_to_survivor(
                            conn,
                            from_batch_id=int(delete_batch_id),
                            to_batch_id=int(survivor_id),
                        )
                        post_counts = _required_ref_counts(conn, int(delete_batch_id))
                        unresolved_required = int(post_counts["required_ref_total"])
                        if unresolved_required > 0:
                            raise RuntimeError(
                                "unresolved_required_references_after_migration:"
                                f"batch_id={delete_batch_id} "
                                f"required_ref_total={unresolved_required}"
                            )
                        delete_count = _delete_embedding_batch_group(conn, int(delete_batch_id))
                        if delete_count != 1:
                            raise RuntimeError(
                                f"expected to delete exactly one embedding_batch group id={delete_batch_id}, "
                                f"deleted={delete_count}"
                            )
                        apply_actions.append(
                            {
                                "delete_batch_id": int(delete_batch_id),
                                "migrations": migration_counts,
                                "post_migration_required_counts": post_counts,
                                "group_delete_count": int(delete_count),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    set_status = "blocked_apply_error"
                    set_reason = f"{type(exc).__name__}: {exc}"
                    blocked += 1
                    # Raising inside engine.begin() rolls back this set transaction.
                    raise
            # If transaction succeeded, mark as applied.
            set_status = "applied"
            set_reason = "migrated references and deleted non-survivor duplicate groups"
            total_applied_deletes += len(delete_candidates)
            applied_sets += 1

        set_reports.append(
            {
                "duplicate_set_id": duplicate_set_id,
                "group_key": group_key,
                "candidate_batch_ids": candidate_ids,
                "analysis_reference_rows": analysis_ref_rows,
                "survivor_selection": survivor_decision,
                "delete_candidates": delete_candidates,
                "candidates": candidate_reports,
                "set_status": set_status,
                "set_reason": set_reason,
                "apply_actions": apply_actions,
            }
        )

    summary = {
        "total_sets_in_scope": len(set_reports),
        "total_delete_candidates": int(total_delete_candidates),
        "conflict_set_count": int(conflicts),
        "blocked_set_count": int(blocked),
        "applied_set_count": int(applied_sets),
        "applied_delete_count": int(total_applied_deletes),
    }
    report = {
        "generated_at_utc": _utc_iso(),
        "mode": mode,
        "run_stamp": str(args.run_stamp),
        "inventory_path": str(inventory_path),
        "reference_index_stats": reference_index_stats,
        "target_lineages": [
            {"source_dataframe_group_id": int(sdf), "entry_max": int(entry)}
            for sdf, entry in sorted(target_lineages)
        ],
        "receipt_validation": receipt_info,
        "required_confirmation_token_for_apply": REQUIRED_CONFIRMATION_TOKEN,
        "summary": summary,
        "duplicate_sets": set_reports,
    }

    _report_write(report_path, report)
    print(f"REPORT_PATH={report_path}")
    print(f"MODE={mode}")
    print(f"SETS_IN_SCOPE={summary['total_sets_in_scope']}")
    print(f"DELETE_CANDIDATES={summary['total_delete_candidates']}")
    print(f"CONFLICT_SET_COUNT={summary['conflict_set_count']}")
    if mode == "apply":
        print(f"APPLIED_SET_COUNT={summary['applied_set_count']}")
        print(f"APPLIED_DELETE_COUNT={summary['applied_delete_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

