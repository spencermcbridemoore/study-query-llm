#!/usr/bin/env python3
"""
Export MCQ-related v2 Postgres rows to JSON files under scratch/mcq_db_backups/.

Includes ``groups`` (mcq_run / mcq_sweep / mcq_sweep_request), ``group_links`` touching
those groups, ``group_members``, linked ``raw_calls``, ``call_artifacts``,
``embedding_vectors``, ``sweep_run_claims`` for those requests/runs, and
``orchestration_jobs`` for those request groups when present.

Prerequisites:
  pip install -e ".[dev]"  (SQLAlchemy + python-dotenv)
  Local Postgres up (e.g. docker compose --profile postgres up -d db)

Usage (repo root):
  python scripts/backup_mcq_db_to_json.py
  python scripts/backup_mcq_db_to_json.py --database-url postgresql://...
  python scripts/backup_mcq_db_to_json.py --output-dir scratch/mcq_db_backups
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, sessionmaker

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MCQ_GROUP_TYPES = ("mcq_run", "mcq_sweep", "mcq_sweep_request")
BACKUP_SCHEMA_VERSION = "study-query-llm/mcq_db_backup/v1"


def _mask_database_url(url: str) -> str:
    if "@" in url and "://" in url:
        head, tail = url.rsplit("@", 1)
        if "://" in head and ":" in head.split("://", 1)[1]:
            scheme_user, _, rest = head.partition("://")
            user, _, _pass = rest.partition(":")
            return f"{scheme_user}://{user}:***@{tail}"
    return url


def _tables_explained() -> Dict[str, str]:
    return {
        "groups": (
            "Mutable experiment batch rows. MCQ probe results live in group_type "
            "'mcq_run' with metadata_json (run_key, probe_details, result_summary). "
            "'mcq_sweep' parents batch many runs; 'mcq_sweep_request' is the request-side type."
        ),
        "group_links": (
            "Parent/child edges between groups (e.g. sweep contains run). Included when "
            "either endpoint is an MCQ-typed group in this export."
        ),
        "group_members": (
            "Links RawCall rows to Group rows. Each mcq_run group typically references "
            "the underlying LLM raw_calls for the probe."
        ),
        "raw_calls": (
            "Immutable provider request/response capture (JSON columns). Restoring these "
            "elsewhere requires matching schema and id remap or insert-new-id strategy."
        ),
        "call_artifacts": (
            "URIs / metadata for multimodal artifacts tied to a raw_call (no blob bytes)."
        ),
        "embedding_vectors": (
            "Embedding vectors stored per raw_call when embeddings were recorded (JSON vector)."
        ),
        "sweep_run_claims": (
            "Worker claim rows for request-driven sweeps tied to request_group_id / run_group_id."
        ),
        "orchestration_jobs": (
            "Durable job queue rows scoped by request_group_id (e.g. mcq_run jobs); often empty."
        ),
    }


def _build_metadata(
    *,
    source_url_redacted: str,
    counts: Dict[str, int],
    group_index: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "backup_schema": BACKUP_SCHEMA_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_database_url_redacted": source_url_redacted,
        "purpose": (
            "Offline backup of MCQ sweep/run graph and linked raw LLM captures from a "
            "Study Query LLM v2 PostgreSQL database. Not a pg_dump; use for archival / "
            "analysis or as a human-readable record. Restore to another DB is not automatic."
        ),
        "row_counts": counts,
        "tables_explained": _tables_explained(),
        "mcq_group_index": group_index,
        "script": "scripts/backup_mcq_db_to_json.py",
    }


def _collect(session: Session) -> Dict[str, Any]:
    from study_query_llm.db.models_v2 import (
        CallArtifact,
        EmbeddingVector,
        Group,
        GroupLink,
        GroupMember,
        OrchestrationJob,
        RawCall,
        SweepRunClaim,
    )

    mcq_groups: List[Group] = (
        session.query(Group)
        .filter(Group.group_type.in_(MCQ_GROUP_TYPES))
        .order_by(Group.id)
        .all()
    )
    g_ids: Set[int] = {g.id for g in mcq_groups}
    if not g_ids:
        return {
            "groups": [],
            "group_links": [],
            "group_members": [],
            "raw_calls": [],
            "call_artifacts": [],
            "embedding_vectors": [],
            "sweep_run_claims": [],
            "orchestration_jobs": [],
            "counts": {k: 0 for k in _tables_explained().keys()},
            "group_index": [],
        }

    links: List[GroupLink] = (
        session.query(GroupLink)
        .filter(
            or_(
                GroupLink.parent_group_id.in_(g_ids),
                GroupLink.child_group_id.in_(g_ids),
            )
        )
        .order_by(GroupLink.id)
        .all()
    )

    members: List[GroupMember] = (
        session.query(GroupMember)
        .filter(GroupMember.group_id.in_(g_ids))
        .order_by(GroupMember.id)
        .all()
    )
    call_ids: Set[int] = {m.call_id for m in members}

    raw_calls: List[RawCall] = (
        session.query(RawCall).filter(RawCall.id.in_(call_ids)).order_by(RawCall.id).all()
        if call_ids
        else []
    )

    artifacts: List[CallArtifact] = (
        session.query(CallArtifact)
        .filter(CallArtifact.call_id.in_(call_ids))
        .order_by(CallArtifact.id)
        .all()
        if call_ids
        else []
    )

    vectors: List[EmbeddingVector] = (
        session.query(EmbeddingVector)
        .filter(EmbeddingVector.call_id.in_(call_ids))
        .order_by(EmbeddingVector.id)
        .all()
        if call_ids
        else []
    )

    claims: List[SweepRunClaim] = (
        session.query(SweepRunClaim)
        .filter(
            or_(
                SweepRunClaim.request_group_id.in_(g_ids),
                SweepRunClaim.run_group_id.in_(g_ids),
            )
        )
        .order_by(SweepRunClaim.id)
        .all()
    )

    jobs: List[OrchestrationJob] = (
        session.query(OrchestrationJob)
        .filter(OrchestrationJob.request_group_id.in_(g_ids))
        .order_by(OrchestrationJob.id)
        .all()
    )

    group_index = [
        {
            "id": g.id,
            "group_type": g.group_type,
            "name": g.name,
            "created_at": g.created_at.isoformat() if g.created_at else None,
        }
        for g in mcq_groups
    ]

    counts = {
        "groups": len(mcq_groups),
        "group_links": len(links),
        "group_members": len(members),
        "raw_calls": len(raw_calls),
        "call_artifacts": len(artifacts),
        "embedding_vectors": len(vectors),
        "sweep_run_claims": len(claims),
        "orchestration_jobs": len(jobs),
    }

    return {
        "groups": [g.to_dict() for g in mcq_groups],
        "group_links": [x.to_dict() for x in links],
        "group_members": [m.to_dict() for m in members],
        "raw_calls": [r.to_dict() for r in raw_calls],
        "call_artifacts": [a.to_dict() for a in artifacts],
        "embedding_vectors": [v.to_dict() for v in vectors],
        "sweep_run_claims": [c.to_dict() for c in claims],
        "orchestration_jobs": [j.to_dict() for j in jobs],
        "counts": counts,
        "group_index": group_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup MCQ-related v2 DB rows to JSON.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: LOCAL_DATABASE_URL, then DATABASE_URL from .env)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "scratch" / "mcq_db_backups",
        help="Directory for output JSON files (created if missing)",
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

    out_dir: Path = args.output_dir
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"mcq_v2_backup_{stamp}"
    full_path = out_dir / f"{base}_full.json"
    summary_path = out_dir / f"{base}_summary.json"

    engine = create_engine(db_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        payload = _collect(session)
    finally:
        session.close()
    engine.dispose()

    counts = payload.pop("counts")
    group_index = payload.pop("group_index")
    redacted = _mask_database_url(db_url)

    backup_metadata = _build_metadata(
        source_url_redacted=redacted,
        counts=counts,
        group_index=group_index,
    )

    full_doc = {"backup_metadata": backup_metadata, **payload}
    summary_doc = {
        "backup_metadata": {
            **backup_metadata,
            "companion_file": full_path.name,
            "note": (
                "Summary backup: same backup_metadata and mcq_group_index as the full export; "
                "no raw_calls / artifacts / vectors / links / members payloads."
            ),
        },
        "mcq_group_index": group_index,
    }

    for path, doc in ((full_path, full_doc), (summary_path, summary_doc)):
        path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {full_path} ({full_path.stat().st_size // 1024} KiB)")
    print(f"Wrote {summary_path} ({summary_path.stat().st_size // 1024} KiB)")
    print(f"Source (redacted): {redacted}")
    print(f"Row counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
