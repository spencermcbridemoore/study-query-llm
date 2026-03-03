#!/usr/bin/env python3
"""
Archive pre-fix clustering_run groups (and their step children + GroupLinks)
from the online Neon DB to the local backup DB, then delete from Neon.

Only groups with metadata_json->>'centroid_fix_era' = 'pre_fix' are selected.
Run label_pre_fix_runs.py first to mark the runs.

Prerequisites:
    docker compose --profile postgres up -d db
    python scripts/init_local_db.py  (first time only)
    python scripts/label_pre_fix_runs.py  (label runs before archiving)

Usage:
    python scripts/archive_pre_fix_runs.py --dry-run   # preview, no changes
    python scripts/archive_pre_fix_runs.py              # execute
    python scripts/archive_pre_fix_runs.py --online-url <url> --local-url <url>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def copy_run_groups_to_local(
    online_session, local_session, run_groups, dry_run: bool
) -> dict[int, int]:
    """
    Copy clustering_run Group rows to local DB.
    Dedupes by (group_type, name). Returns online_id -> local_id map.
    """
    from study_query_llm.db.models_v2 import Group

    id_map: dict[int, int] = {}
    for g in run_groups:
        existing = local_session.query(Group).filter_by(
            group_type=g.group_type,
            name=g.name,
        ).first()

        if existing:
            id_map[g.id] = existing.id
            continue

        if dry_run:
            id_map[g.id] = g.id
            continue

        local_group = Group(
            group_type=g.group_type,
            name=g.name,
            description=g.description,
            created_at=g.created_at,
            metadata_json=g.metadata_json,
        )
        local_session.add(local_group)
        local_session.flush()
        local_session.refresh(local_group)
        id_map[g.id] = local_group.id

    return id_map


def copy_step_groups_to_local(
    online_session, local_session, step_groups, dry_run: bool
) -> dict[int, int]:
    """
    Copy clustering_step Group rows to local DB.
    Step names are NOT unique across runs, so every step is inserted as a new row
    (no name-based dedup). Returns online_id -> local_id map.
    """
    from study_query_llm.db.models_v2 import Group

    id_map: dict[int, int] = {}
    for g in step_groups:
        if dry_run:
            id_map[g.id] = g.id
            continue

        local_group = Group(
            group_type=g.group_type,
            name=g.name,
            description=g.description,
            created_at=g.created_at,
            metadata_json=g.metadata_json,
        )
        local_session.add(local_group)
        local_session.flush()
        local_session.refresh(local_group)
        id_map[g.id] = local_group.id

    return id_map


def copy_group_links_to_local(
    online_session, local_session, links, id_map: dict[int, int], dry_run: bool
) -> int:
    """
    Copy GroupLink rows to local DB, remapping parent/child IDs via id_map.
    Skips links whose parent or child is not in id_map.
    Returns count of links copied.
    """
    from study_query_llm.db.models_v2 import GroupLink

    copied = 0
    for lnk in links:
        local_parent = id_map.get(lnk.parent_group_id)
        local_child = id_map.get(lnk.child_group_id)
        if local_parent is None or local_child is None:
            continue

        if dry_run:
            copied += 1
            continue

        exists = local_session.query(GroupLink).filter_by(
            parent_group_id=local_parent,
            child_group_id=local_child,
            link_type=lnk.link_type,
        ).first()
        if exists:
            continue

        local_session.add(GroupLink(
            parent_group_id=local_parent,
            child_group_id=local_child,
            link_type=lnk.link_type,
            position=lnk.position,
            metadata_json=lnk.metadata_json,
            created_at=lnk.created_at,
        ))
        copied += 1

    if not dry_run:
        local_session.flush()

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive pre-fix clustering_run groups from Neon to local DB"
    )
    parser.add_argument("--online-url", default=None, help="Online DB URL (default: DATABASE_URL)")
    parser.add_argument("--local-url", default=None, help="Local DB URL (default: LOCAL_DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing or deleting")
    args = parser.parse_args()

    online_url = args.online_url or os.environ.get("DATABASE_URL")
    local_url = args.local_url or os.environ.get("LOCAL_DATABASE_URL")

    if not online_url:
        print("ERROR: DATABASE_URL not set.")
        return 1
    if not local_url:
        print("ERROR: LOCAL_DATABASE_URL not set. Add it to .env or pass --local-url.")
        return 1

    from sqlalchemy import text as sa_text
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.models_v2 import Group, GroupLink

    print(f"Online DB: ...@{online_url.split('@')[-1] if '@' in online_url else online_url}")
    print(f"Local  DB: ...@{local_url.split('@')[-1] if '@' in local_url else local_url}")
    if args.dry_run:
        print("[DRY RUN] No changes will be written or deleted.\n")

    online_db = DatabaseConnectionV2(online_url, enable_pgvector=False)
    local_db = DatabaseConnectionV2(local_url, enable_pgvector=False)
    local_db.init_db()

    # --- Step 1: Find pre_fix runs ---
    with online_db.session_scope() as online_session:
        pre_fix_runs = (
            online_session.query(Group)
            .filter(
                Group.group_type == "clustering_run",
                sa_text("metadata_json->>'centroid_fix_era' = 'pre_fix'"),
            )
            .order_by(Group.id)
            .all()
        )

        if not pre_fix_runs:
            print("No clustering_run groups labeled 'pre_fix'. Nothing to archive.")
            print("Run label_pre_fix_runs.py first.")
            return 0

        run_ids = [r.id for r in pre_fix_runs]
        print(f"Found {len(run_ids)} pre_fix clustering_run group(s).")

        # --- Step 2: Resolve step groups via GroupLinks ---
        step_links = (
            online_session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id.in_(run_ids),
                GroupLink.link_type == "clustering_step",
            )
            .order_by(GroupLink.parent_group_id, GroupLink.position)
            .all()
        )

        step_ids = [lnk.child_group_id for lnk in step_links]
        step_groups = []
        if step_ids:
            step_groups = (
                online_session.query(Group)
                .filter(Group.id.in_(step_ids))
                .order_by(Group.id)
                .all()
            )

        print(f"Found {len(step_groups)} clustering_step group(s) across those runs.")
        print(f"Found {len(step_links)} GroupLink(s) (run -> step).")

        # --- Step 3: Copy to local DB ---
        print("\nCopying to local DB...")
        with local_db.session_scope() as local_session:
            run_id_map = copy_run_groups_to_local(
                online_session, local_session, pre_fix_runs, args.dry_run,
            )
            step_id_map = copy_step_groups_to_local(
                online_session, local_session, step_groups, args.dry_run,
            )

            combined_map = {**run_id_map, **step_id_map}

            links_copied = copy_group_links_to_local(
                online_session, local_session, step_links, combined_map, args.dry_run,
            )

            if not args.dry_run:
                local_session.commit()

        print(f"  {'Would copy' if args.dry_run else 'Copied'} {len(run_id_map)} run group(s).")
        print(f"  {'Would copy' if args.dry_run else 'Copied'} {len(step_id_map)} step group(s).")
        print(f"  {'Would copy' if args.dry_run else 'Copied'} {links_copied} GroupLink(s).")

    # --- Step 4: Delete from Neon ---
    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {len(step_ids)} step group(s) from Neon.")
        print(f"[DRY RUN] Would delete {len(run_ids)} run group(s) from Neon.")
        print("\nSummary (dry run):")
        print(f"  Pre-fix runs found    : {len(run_ids)}")
        print(f"  Step groups found     : {len(step_ids)}")
        print(f"  GroupLinks found      : {len(step_links)}")
        print(f"  Would copy runs       : {len(run_id_map)}")
        print(f"  Would copy steps      : {len(step_id_map)}")
        print(f"  Would copy links      : {links_copied}")
        print(f"  Would delete steps    : {len(step_ids)}")
        print(f"  Would delete runs     : {len(run_ids)}")
        print("\nRun without --dry-run to execute.")
        return 0

    print("\nDeleting from online DB...")
    deleted_steps = 0
    deleted_runs = 0
    chunk_size = 100

    with online_db.session_scope() as online_session:
        # Delete step groups first (child end of links; cascade removes GroupLinks)
        for i in range(0, len(step_ids), chunk_size):
            chunk = step_ids[i : i + chunk_size]
            rows = online_session.query(Group).filter(Group.id.in_(chunk)).all()
            for row in rows:
                online_session.delete(row)
                deleted_steps += 1
            online_session.flush()

        # Then delete run groups (cascade removes sweep->run GroupLinks)
        for i in range(0, len(run_ids), chunk_size):
            chunk = run_ids[i : i + chunk_size]
            rows = online_session.query(Group).filter(Group.id.in_(chunk)).all()
            for row in rows:
                online_session.delete(row)
                deleted_runs += 1
            online_session.flush()

        online_session.commit()

    print(f"  Deleted {deleted_steps} step group(s) from Neon.")
    print(f"  Deleted {deleted_runs} run group(s) from Neon.")

    print("\nSummary:")
    print(f"  Pre-fix runs archived : {len(run_id_map)}")
    print(f"  Step groups archived  : {len(step_id_map)}")
    print(f"  GroupLinks archived   : {links_copied}")
    print(f"  Steps deleted (Neon)  : {deleted_steps}")
    print(f"  Runs deleted (Neon)   : {deleted_runs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
