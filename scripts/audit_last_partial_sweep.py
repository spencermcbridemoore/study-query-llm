#!/usr/bin/env python3
"""Read-only DB audit for last partial sweep migration readiness.

Reports:
- duplicate clustering_run run_key values
- missing clustering_run run_key
- duplicate group_links for (parent_group_id, child_group_id, link_type)
- newest request candidate matching partial-sweep migration heuristics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, text as sa_text

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group, GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.sweep_query_service import SweepQueryService


@dataclass
class RequestCandidate:
    request_id: int
    name: str
    created_at: str
    request_status: str
    expected_count: int
    completed_count: int
    missing_count: int


def _collect_run_key_quality(session) -> Dict[str, Any]:
    total_runs = (
        session.query(func.count(Group.id))
        .filter(Group.group_type == "clustering_run")
        .scalar()
    ) or 0
    missing_run_key = (
        session.query(func.count(Group.id))
        .filter(
            Group.group_type == "clustering_run",
            sa_text("(metadata_json->>'run_key') IS NULL"),
        )
        .scalar()
    ) or 0
    duplicate_rows = (
        session.query(
            sa_text("metadata_json->>'run_key' as run_key"),
            func.count(Group.id).label("n"),
        )
        .filter(
            Group.group_type == "clustering_run",
            sa_text("(metadata_json->>'run_key') IS NOT NULL"),
        )
        .group_by(sa_text("metadata_json->>'run_key'"))
        .having(func.count(Group.id) > 1)
        .order_by(func.count(Group.id).desc())
        .all()
    )
    return {
        "total_clustering_runs": total_runs,
        "missing_run_key_count": int(missing_run_key),
        "duplicate_run_key_count": len(duplicate_rows),
        "duplicate_run_keys": [
            {"run_key": row.run_key, "count": int(row.n)} for row in duplicate_rows
        ],
    }


def _collect_duplicate_links(session) -> Dict[str, Any]:
    rows = (
        session.query(
            GroupLink.parent_group_id,
            GroupLink.child_group_id,
            GroupLink.link_type,
            func.count(GroupLink.id).label("n"),
        )
        .group_by(
            GroupLink.parent_group_id, GroupLink.child_group_id, GroupLink.link_type
        )
        .having(func.count(GroupLink.id) > 1)
        .order_by(func.count(GroupLink.id).desc())
        .all()
    )
    return {
        "duplicate_link_triplet_count": len(rows),
        "duplicates": [
            {
                "parent_group_id": int(r.parent_group_id),
                "child_group_id": int(r.child_group_id),
                "link_type": str(r.link_type),
                "count": int(r.n),
            }
            for r in rows
        ],
    }


def _select_request_candidate(
    svc: SweepQueryService,
    *,
    expected_min: int,
    completed_min: int,
    completed_max: int,
) -> Optional[RequestCandidate]:
    requests = svc.list_clustering_sweep_requests(include_fulfilled=False)
    candidates: List[RequestCandidate] = []
    for req in requests:
        request_id = req["id"]
        summary = svc.get_request_progress_summary(request_id)
        if not summary:
            continue
        expected = int(summary.get("expected_count", 0))
        completed = int(summary.get("completed_count", 0))
        if not (
            expected >= expected_min and completed_min <= completed <= completed_max
        ):
            continue
        candidates.append(
            RequestCandidate(
                request_id=int(request_id),
                name=str(req.get("name", "?")),
                created_at=str(req.get("created_at")),
                request_status=str(summary.get("request_status", "?")),
                expected_count=expected,
                completed_count=completed,
                missing_count=int(summary.get("missing_count", 0)),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.created_at, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit DB readiness and select last partial sweep candidate",
    )
    parser.add_argument("--expected-min", type=int, default=100)
    parser.add_argument("--completed-min", type=int, default=1)
    parser.add_argument("--completed-max", type=int, default=25)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        sweep_svc = SweepQueryService(repo)

        run_key_quality = _collect_run_key_quality(session)
        duplicate_links = _collect_duplicate_links(session)
        candidate = _select_request_candidate(
            sweep_svc,
            expected_min=args.expected_min,
            completed_min=args.completed_min,
            completed_max=args.completed_max,
        )

        blockers = []
        if run_key_quality["missing_run_key_count"] > 0:
            blockers.append("clustering_run rows missing run_key")
        if run_key_quality["duplicate_run_key_count"] > 0:
            blockers.append("duplicate clustering_run run_key values")
        if duplicate_links["duplicate_link_triplet_count"] > 0:
            blockers.append("duplicate group_links triplets")
        if candidate is None:
            blockers.append("no matching partial request candidate found")

        result = {
            "candidate_request": asdict(candidate) if candidate else None,
            "run_key_quality": run_key_quality,
            "duplicate_group_links": duplicate_links,
            "blockers": blockers,
        }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print("=" * 80)
    print("Last Partial Sweep Migration Audit")
    print("=" * 80)
    if candidate:
        print("Candidate request:")
        print(f"  id={candidate.request_id}")
        print(f"  name={candidate.name}")
        print(f"  status={candidate.request_status}")
        print(f"  expected={candidate.expected_count}")
        print(f"  completed={candidate.completed_count}")
        print(f"  missing={candidate.missing_count}")
    else:
        print("Candidate request: NONE")

    print("\nRun key quality:")
    print(f"  total clustering_run: {run_key_quality['total_clustering_runs']}")
    print(f"  missing run_key: {run_key_quality['missing_run_key_count']}")
    print(f"  duplicate run_key values: {run_key_quality['duplicate_run_key_count']}")
    for item in run_key_quality["duplicate_run_keys"][:10]:
        print(f"    - {item['run_key']}: {item['count']}")

    print("\nDuplicate group links:")
    print(
        "  duplicate (parent, child, link_type) tuples: "
        f"{duplicate_links['duplicate_link_triplet_count']}"
    )
    for item in duplicate_links["duplicates"][:10]:
        print(
            "    - "
            f"({item['parent_group_id']}, {item['child_group_id']}, "
            f"{item['link_type']}): {item['count']}"
        )

    print("\nBlockers:")
    if result["blockers"]:
        for b in result["blockers"]:
            print(f"  - {b}")
    else:
        print("  - none")


if __name__ == "__main__":
    main()
