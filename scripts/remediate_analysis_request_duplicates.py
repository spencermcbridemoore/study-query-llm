#!/usr/bin/env python3
"""Remediate duplicate ``analysis_request`` Group rows before the unique index.

A get-or-create race let concurrent ``analyze()`` callers each create a
duplicate ``analysis_request`` group for the same identity
``(method_name, input_id, run_key)``. The DB-enforced partial unique index
(``db/migrations/add_analysis_request_unique_index.py``) cannot be created while
duplicates exist, so this script collapses each duplicated identity first.

For every duplicated identity it keeps the lowest group id (the first created)
and, for each remaining "loser" group, repoints every reference onto the keeper
before deleting the loser:

  * all foreign-key columns referencing ``groups.id`` (provenanced_runs,
    group_links, group_members, sweep_run_claims, orchestration_jobs,
    analysis_results) -- a colliding loser-side row (one that would violate a
    natural unique key on the keeper) is deleted instead of repointed;
  * JSON-embedded ``request_group_id`` lineage pointers in
    ``groups.metadata_json`` and ``provenanced_runs.metadata_json``/``config_json``;
  * ``group_links`` self-links and duplicate (parent, child, link_type) edges
    created by the repoint are cleaned up.

The core (``remediate_duplicates``) is dialect-agnostic and operates on a passed
session; ``main()`` adds canonical-target guards. Default is a DRY RUN.

CANONICAL WRITE: ``--apply`` mutates the canonical database. Get explicit human
approval first. Run the dry run, review the plan, then re-run with ``--apply``.

Usage:
    python scripts/remediate_analysis_request_duplicates.py            # dry run
    python scripts/remediate_analysis_request_duplicates.py --apply    # writes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.lane import Lane
from study_query_llm.db.models_v2 import (
    AnalysisResult,
    Group,
    GroupLink,
    GroupMember,
    OrchestrationJob,
    ProvenancedRun,
    SweepRunClaim,
)
from study_query_llm.db.write_intent import WriteIntent

ANALYSIS_REQUEST = "analysis_request"

# Every foreign-key column that references groups.id (see models_v2). Repointing
# all of them off a loser empties its reference set so the row can be deleted
# without cascading data loss (CASCADE columns) or orphaning links (SET NULL).
_GROUP_REFERENCES: list[tuple[type, str]] = [
    (GroupMember, "group_id"),
    (GroupLink, "parent_group_id"),
    (GroupLink, "child_group_id"),
    (SweepRunClaim, "request_group_id"),
    (SweepRunClaim, "run_group_id"),
    (OrchestrationJob, "request_group_id"),
    (AnalysisResult, "source_group_id"),
    (AnalysisResult, "analysis_group_id"),
    (ProvenancedRun, "request_group_id"),
    (ProvenancedRun, "source_group_id"),
    (ProvenancedRun, "result_group_id"),
    (ProvenancedRun, "input_snapshot_group_id"),
]


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_duplicate_identities(session) -> dict[tuple[str, str, str], list[int]]:
    """Return {identity: [group_ids sorted asc]} for identities with > 1 group.

    Identity is (method_name, input_id, run_key) read from metadata_json. Rows
    with an incomplete identity (e.g. empty-metadata container groups) are never
    treated as duplicates.
    """
    groups = (
        session.query(Group)
        .filter(Group.group_type == ANALYSIS_REQUEST)
        .order_by(Group.id.asc())
        .all()
    )
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for group in groups:
        meta = dict(group.metadata_json or {})
        method_name = meta.get("method_name")
        input_id = meta.get("input_id")
        run_key = meta.get("run_key")
        if method_name is None or input_id is None or run_key is None:
            continue
        key = (str(method_name), str(input_id), str(run_key))
        buckets.setdefault(key, []).append(int(group.id))
    return {key: sorted(ids) for key, ids in buckets.items() if len(ids) > 1}


def _repoint_fk_references(session, *, loser_id, keeper_id, apply, report) -> None:
    for model, column in _GROUP_REFERENCES:
        rows = session.query(model).filter(getattr(model, column) == loser_id).all()
        for row in rows:
            entry = {"table": model.__tablename__, "column": column, "pk": int(row.id)}
            if not apply:
                entry["action"] = "repoint"
                report["fk_repoints"].append(entry)
                continue
            try:
                with session.begin_nested():
                    setattr(row, column, keeper_id)
                    session.flush()
                entry["action"] = "repoint"
            except IntegrityError:
                # Repoint collides with an existing keeper-side row on a natural
                # unique key: the loser-side row is a genuine duplicate -> drop.
                session.expire(row)
                session.delete(row)
                session.flush()
                entry["action"] = "delete_conflict"
            report["fk_repoints"].append(entry)


def _repoint_json_request_group_id(session, *, loser_id, keeper_id, apply, report) -> None:
    for group in session.query(Group).filter(Group.id != loser_id).all():
        meta = group.metadata_json
        if isinstance(meta, dict) and _as_int(meta.get("request_group_id")) == loser_id:
            report["json_repoints"].append(
                {"table": "groups", "id": int(group.id), "field": "metadata_json.request_group_id"}
            )
            if apply:
                updated = dict(meta)
                updated["request_group_id"] = keeper_id
                group.metadata_json = updated
    for run in session.query(ProvenancedRun).all():
        for attr in ("metadata_json", "config_json"):
            payload = getattr(run, attr)
            if isinstance(payload, dict) and _as_int(payload.get("request_group_id")) == loser_id:
                report["json_repoints"].append(
                    {"table": "provenanced_runs", "id": int(run.id), "field": f"{attr}.request_group_id"}
                )
                if apply:
                    updated = dict(payload)
                    updated["request_group_id"] = keeper_id
                    setattr(run, attr, updated)
    if apply:
        session.flush()


def _cleanup_group_links_for_keeper(session, *, keeper_id, apply, report) -> None:
    """Drop self-links and duplicate (parent, child, link_type) edges on keeper."""
    links = (
        session.query(GroupLink)
        .filter(
            (GroupLink.parent_group_id == keeper_id)
            | (GroupLink.child_group_id == keeper_id)
        )
        .order_by(GroupLink.id.asc())
        .all()
    )
    seen: set[tuple[int, int, str]] = set()
    for link in links:
        if link.parent_group_id == link.child_group_id:
            report["link_cleanups"].append({"id": int(link.id), "action": "delete_self_link"})
            if apply:
                session.delete(link)
            continue
        key = (link.parent_group_id, link.child_group_id, link.link_type)
        if key in seen:
            report["link_cleanups"].append({"id": int(link.id), "action": "delete_duplicate"})
            if apply:
                session.delete(link)
        else:
            seen.add(key)
    if apply:
        session.flush()


def remediate_duplicates(session, *, apply: bool) -> dict:
    """Collapse duplicate analysis_request identities; return an action report.

    Dialect-agnostic: operates only through the ORM session. The caller owns the
    transaction (commit/rollback).
    """
    duplicates = find_duplicate_identities(session)
    report: dict = {
        "duplicate_identities": len(duplicates),
        "groups": [],
        "fk_repoints": [],
        "json_repoints": [],
        "link_cleanups": [],
        "deleted_groups": [],
        "apply": bool(apply),
    }
    for key, ids in sorted(duplicates.items(), key=lambda kv: kv[1][0]):
        keeper_id = ids[0]
        losers = ids[1:]
        report["groups"].append(
            {"identity": list(key), "keeper": keeper_id, "losers": list(losers)}
        )
        for loser_id in losers:
            _repoint_fk_references(
                session, loser_id=loser_id, keeper_id=keeper_id, apply=apply, report=report
            )
            _repoint_json_request_group_id(
                session, loser_id=loser_id, keeper_id=keeper_id, apply=apply, report=report
            )
            _cleanup_group_links_for_keeper(
                session, keeper_id=keeper_id, apply=apply, report=report
            )
            report["deleted_groups"].append(loser_id)
            if apply:
                session.execute(text("DELETE FROM groups WHERE id = :id"), {"id": loser_id})
                session.flush()

    report["remaining_duplicates"] = len(find_duplicate_identities(session))
    return report


def _resolve_database_url(explicit_url: str | None, env_var: str) -> str:
    if explicit_url:
        return explicit_url.strip()
    for key in (env_var, "DATABASE_URL", "JETSTREAM_DATABASE_URL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    raise ValueError(
        f"No database URL found. Set --database-url, {env_var}, DATABASE_URL, or "
        "JETSTREAM_DATABASE_URL."
    )


def _print_report(report: dict) -> None:
    print(f"duplicate_identities={report['duplicate_identities']}")
    for group in report["groups"]:
        print(
            f"  identity={group['identity']} keeper={group['keeper']} losers={group['losers']}"
        )
    print(f"fk_repoints={len(report['fk_repoints'])}")
    for entry in report["fk_repoints"]:
        print(f"  fk {entry['table']}.{entry['column']} pk={entry['pk']} -> {entry['action']}")
    print(f"json_repoints={len(report['json_repoints'])}")
    for entry in report["json_repoints"]:
        print(f"  json {entry['table']}#{entry['id']} {entry['field']}")
    print(f"link_cleanups={len(report['link_cleanups'])}")
    for entry in report["link_cleanups"]:
        print(f"  link id={entry['id']} -> {entry['action']}")
    print(f"deleted_groups={report['deleted_groups']}")
    print(f"remaining_duplicates={report['remaining_duplicates']}")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Collapse duplicate analysis_request groups before the unique index."
    )
    parser.add_argument("--database-url", type=str, default=None, help="Explicit target database URL.")
    parser.add_argument(
        "--env-var",
        type=str,
        default="CANONICAL_DATABASE_URL",
        help="Primary env var used to resolve the target URL.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the remediation (default is dry-run; CANONICAL WRITE).",
    )
    args = parser.parse_args()

    try:
        resolved_url = _resolve_database_url(args.database_url, args.env_var)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    db = DatabaseConnectionV2(
        resolved_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    if db.engine.dialect.name != "postgresql":
        print("ERROR: remediation is Postgres-only.", file=sys.stderr)
        return 2
    if db.lane is not Lane.CANONICAL:
        print(f"ERROR: refusing remediation on non-canonical lane {db.lane.name}.", file=sys.stderr)
        return 2

    with db.session_scope() as session:
        report = remediate_duplicates(session, apply=args.apply)
        _print_report(report)
        if args.apply and report["remaining_duplicates"] != 0:
            raise RuntimeError(
                "Remediation incomplete: duplicates remain after apply; transaction rolled back."
            )

    if not args.apply:
        print("DRY RUN - no changes applied. Re-run with --apply (CANONICAL WRITE) after review.")
        return 0

    # Authoritative post-commit confirmation in a fresh session.
    with db.session_scope() as session:
        remaining = len(find_duplicate_identities(session))
    print(f"post_commit_remaining_duplicates={remaining}")
    if remaining != 0:
        print("ERROR: duplicates still present after commit.", file=sys.stderr)
        return 3
    print("OK: zero duplicate analysis_request identities remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
