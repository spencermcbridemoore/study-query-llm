#!/usr/bin/env python3
"""
Export MCQ-related v2 Postgres rows to a JSON backup + manifest.

Surface exported (per data-pipeline rebuild plan):
- groups where group_type in (mcq_run, mcq_sweep, mcq_sweep_request)
- provenanced_runs rows pointing at those groups
- linked analysis_results rows
- linked call_artifacts metadata (URIs only, no blob bytes)

Usage:
  python scripts/backup_mcq_db_to_json.py
  python scripts/backup_mcq_db_to_json.py --database-url postgresql://...
  python scripts/backup_mcq_db_to_json.py --output backup_pg_dumps/mcq_export_20260417.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, sessionmaker

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

MCQ_GROUP_TYPES = ("mcq_run", "mcq_sweep", "mcq_sweep_request")
BACKUP_SCHEMA_VERSION = "study-query-llm/mcq_db_backup/v2"


def _mask_database_url(url: str) -> str:
    if "@" in url and "://" in url:
        head, tail = url.rsplit("@", 1)
        if "://" in head and ":" in head.split("://", 1)[1]:
            scheme_user, _, rest = head.partition("://")
            user, _, _password = rest.partition(":")
            return f"{scheme_user}://{user}:***@{tail}"
    return url


def _tables_explained() -> dict[str, str]:
    return {
        "groups": (
            "MCQ lineage groups only (group_type in mcq_run/mcq_sweep/mcq_sweep_request)."
        ),
        "provenanced_runs": (
            "Canonical execution rows whose request/source/result/input snapshot references "
            "an MCQ lineage group."
        ),
        "analysis_results": (
            "Per-metric scalar/JSON analysis rows linked to MCQ lineage (by source_group_id "
            "or analysis_group_id)."
        ),
        "call_artifacts": (
            "CallArtifact metadata for calls linked to MCQ groups via group_members; includes "
            "URIs but never blob bytes."
        ),
    }


def _group_index(mcq_groups: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": group.id,
            "group_type": group.group_type,
            "name": group.name,
            "created_at": group.created_at.isoformat() if group.created_at else None,
        }
        for group in mcq_groups
    ]


def _collect(session: Session) -> dict[str, Any]:
    from study_query_llm.db.models_v2 import (
        AnalysisResult,
        CallArtifact,
        Group,
        GroupMember,
        ProvenancedRun,
    )

    mcq_groups = (
        session.query(Group)
        .filter(Group.group_type.in_(MCQ_GROUP_TYPES))
        .order_by(Group.id)
        .all()
    )
    group_ids = {int(group.id) for group in mcq_groups}
    if not group_ids:
        counts = {key: 0 for key in _tables_explained()}
        return {
            "groups": [],
            "provenanced_runs": [],
            "analysis_results": [],
            "call_artifacts": [],
            "counts": counts,
            "mcq_group_index": [],
        }

    provenanced_runs = (
        session.query(ProvenancedRun)
        .filter(
            or_(
                ProvenancedRun.request_group_id.in_(group_ids),
                ProvenancedRun.source_group_id.in_(group_ids),
                ProvenancedRun.result_group_id.in_(group_ids),
                ProvenancedRun.input_snapshot_group_id.in_(group_ids),
            )
        )
        .order_by(ProvenancedRun.id)
        .all()
    )

    analysis_results = (
        session.query(AnalysisResult)
        .filter(
            or_(
                AnalysisResult.source_group_id.in_(group_ids),
                AnalysisResult.analysis_group_id.in_(group_ids),
            )
        )
        .order_by(AnalysisResult.id)
        .all()
    )

    members = (
        session.query(GroupMember)
        .filter(GroupMember.group_id.in_(group_ids))
        .order_by(GroupMember.id)
        .all()
    )
    call_ids = {int(member.call_id) for member in members}
    call_artifacts = (
        session.query(CallArtifact)
        .filter(CallArtifact.call_id.in_(call_ids))
        .order_by(CallArtifact.id)
        .all()
        if call_ids
        else []
    )

    counts = {
        "groups": len(mcq_groups),
        "provenanced_runs": len(provenanced_runs),
        "analysis_results": len(analysis_results),
        "call_artifacts": len(call_artifacts),
    }
    return {
        "groups": [group.to_dict() for group in mcq_groups],
        "provenanced_runs": [run.to_dict() for run in provenanced_runs],
        "analysis_results": [result.to_dict() for result in analysis_results],
        "call_artifacts": [artifact.to_dict() for artifact in call_artifacts],
        "counts": counts,
        "mcq_group_index": _group_index(mcq_groups),
    }


def _build_documents(
    *,
    payload: dict[str, Any],
    source_url_redacted: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = payload["counts"]
    group_index = payload["mcq_group_index"]
    exported_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "backup_schema": BACKUP_SCHEMA_VERSION,
        "exported_at_utc": exported_at,
        "source_database_url_redacted": source_url_redacted,
        "row_counts": counts,
        "tables_explained": _tables_explained(),
        "mcq_group_index": group_index,
        "script": "scripts/backup_mcq_db_to_json.py",
    }
    export_doc = {
        "backup_metadata": {
            "backup_schema": BACKUP_SCHEMA_VERSION,
            "exported_at_utc": exported_at,
            "source_database_url_redacted": source_url_redacted,
            "row_counts": counts,
            "script": "scripts/backup_mcq_db_to_json.py",
        },
        "groups": payload["groups"],
        "provenanced_runs": payload["provenanced_runs"],
        "analysis_results": payload["analysis_results"],
        "call_artifacts": payload["call_artifacts"],
    }
    return export_doc, manifest


def _resolve_output_path(*, output: Path | None, output_dir: Path | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output and output_dir:
        raise ValueError("Use either --output or --output-dir, not both.")
    if output is not None:
        resolved = output if output.is_absolute() else REPO / output
        return resolved
    if output_dir is None:
        default_dir = REPO / "backup_pg_dumps"
        return default_dir / f"mcq_export_{stamp}.json"
    resolved_dir = output_dir if output_dir.is_absolute() else REPO / output_dir
    return resolved_dir / f"mcq_export_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MCQ-related v2 rows to JSON backup.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: LOCAL_DATABASE_URL, then DATABASE_URL from .env)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: backup_pg_dumps/mcq_export_<timestamp>.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Compatibility option: output directory, filename remains timestamped.",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    db_url = (args.database_url or "").strip() or (
        os.environ.get("LOCAL_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    ).strip()
    if not db_url:
        print(
            "ERROR: Set LOCAL_DATABASE_URL or DATABASE_URL in .env, or pass --database-url.",
            file=sys.stderr,
        )
        return 1

    try:
        output_path = _resolve_output_path(output=args.output, output_dir=args.output_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".manifest.json")

    engine = create_engine(db_url, pool_pre_ping=True)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()
    try:
        payload = _collect(session)
    finally:
        session.close()
        engine.dispose()

    redacted = _mask_database_url(db_url)
    export_doc, manifest = _build_documents(
        payload=payload,
        source_url_redacted=redacted,
    )
    output_path.write_text(
        json.dumps(export_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote export: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Row counts: {manifest['row_counts']}")
    print(f"Source (redacted): {redacted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
