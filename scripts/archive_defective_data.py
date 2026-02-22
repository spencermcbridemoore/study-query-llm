#!/usr/bin/env python3
"""
Archive defective RawCall records from the online Neon DB to local backup,
then delete them from Neon.

Records in the "defective_data" label group are considered invalid and should
not pollute the online DB or downstream analysis. This script:
  1. Finds all RawCall IDs tagged as defective_data in the online DB
  2. Copies those records (+ GroupMembers, EmbeddingVectors, CallArtifacts, Groups)
     to the local backup DB
  3. Deletes the RawCall rows from Neon (cascades remove related rows)

Run sync_from_online.py first to ensure other valid records are backed up.

Prerequisites:
    docker compose --profile postgres up -d db
    python scripts/init_local_db.py  (first time only)

Usage:
    python scripts/archive_defective_data.py --dry-run   # preview, no changes
    python scripts/archive_defective_data.py             # execute
    python scripts/archive_defective_data.py --online-url <url> --local-url <url>
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def copy_groups_to_local(online_session, local_session, group_ids: list[int], dry_run: bool) -> dict[int, int]:
    """
    Copy Group rows to local DB. Returns online_group_id -> local_group_id mapping.
    Matches on (group_type, name) to avoid duplicates.
    """
    from study_query_llm.db.models_v2 import Group

    if not group_ids:
        return {}

    online_groups = online_session.query(Group).filter(Group.id.in_(group_ids)).all()
    id_map: dict[int, int] = {}

    for g in online_groups:
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


def copy_calls_to_local(online_session, local_session, call_ids: list[int], dry_run: bool) -> int:
    """
    Copy RawCall rows + all related rows to local DB.
    Returns number of RawCall rows actually copied (skipping already-present ones).
    """
    from study_query_llm.db.models_v2 import (
        RawCall, GroupMember, EmbeddingVector, CallArtifact
    )

    copied = 0

    # --- RawCalls ---
    online_calls = online_session.query(RawCall).filter(RawCall.id.in_(call_ids)).all()
    for c in online_calls:
        if local_session.query(RawCall).filter_by(id=c.id).first():
            continue  # already archived
        if dry_run:
            copied += 1
            continue
        local_call = RawCall(
            id=c.id,
            provider=c.provider,
            model=c.model,
            modality=c.modality,
            status=c.status,
            request_json=c.request_json,
            response_json=c.response_json,
            error_json=c.error_json,
            latency_ms=c.latency_ms,
            tokens_json=c.tokens_json,
            metadata_json=c.metadata_json,
            created_at=c.created_at,
        )
        local_session.add(local_call)
        copied += 1

    if not dry_run:
        local_session.flush()

    # --- Groups & GroupMembers ---
    members = online_session.query(GroupMember).filter(GroupMember.call_id.in_(call_ids)).all()
    group_ids = list({m.group_id for m in members})
    group_id_map = copy_groups_to_local(online_session, local_session, group_ids, dry_run)

    if not dry_run:
        for m in members:
            local_group_id = group_id_map.get(m.group_id, m.group_id)
            exists = local_session.query(GroupMember).filter_by(
                group_id=local_group_id, call_id=m.call_id
            ).first()
            if exists:
                continue
            local_session.add(GroupMember(
                group_id=local_group_id,
                call_id=m.call_id,
                added_at=m.added_at,
                position=m.position,
                role=m.role,
            ))

    # --- EmbeddingVectors ---
    vectors = online_session.query(EmbeddingVector).filter(EmbeddingVector.call_id.in_(call_ids)).all()
    if not dry_run:
        for v in vectors:
            if local_session.query(EmbeddingVector).filter_by(call_id=v.call_id).first():
                continue
            local_session.add(EmbeddingVector(
                call_id=v.call_id,
                vector=v.vector,
                dimension=v.dimension,
                norm=v.norm,
                metadata_json=v.metadata_json,
            ))

    # --- CallArtifacts ---
    artifacts = online_session.query(CallArtifact).filter(CallArtifact.call_id.in_(call_ids)).all()
    if not dry_run:
        for a in artifacts:
            if local_session.query(CallArtifact).filter_by(call_id=a.call_id, uri=a.uri).first():
                continue
            local_session.add(CallArtifact(
                call_id=a.call_id,
                artifact_type=a.artifact_type,
                uri=a.uri,
                content_type=a.content_type,
                byte_size=a.byte_size,
                metadata_json=a.metadata_json,
            ))

    if not dry_run:
        local_session.flush()

    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Archive defective records from Neon to local DB, then delete from Neon"
    )
    parser.add_argument("--online-url", default=None, help="Online DB URL (default: DATABASE_URL)")
    parser.add_argument("--local-url", default=None, help="Local DB URL (default: LOCAL_DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing or deleting")
    args = parser.parse_args()

    online_url = args.online_url or os.environ.get("DATABASE_URL")
    local_url = args.local_url or os.environ.get("LOCAL_DATABASE_URL")

    if not online_url:
        print("ERROR: DATABASE_URL not set.")
        sys.exit(1)
    if not local_url:
        print("ERROR: LOCAL_DATABASE_URL not set. Add it to .env or pass --local-url.")
        sys.exit(1)

    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository
    from study_query_llm.services.data_quality_service import DataQualityService
    from study_query_llm.db.models_v2 import RawCall, GroupMember, Group

    print(f"Online DB: ...@{online_url.split('@')[-1] if '@' in online_url else online_url}")
    print(f"Local  DB: ...@{local_url.split('@')[-1] if '@' in local_url else local_url}")
    if args.dry_run:
        print("[DRY RUN] No changes will be written or deleted.\n")

    online_db = DatabaseConnectionV2(online_url, enable_pgvector=False)
    local_db = DatabaseConnectionV2(local_url, enable_pgvector=False)

    # --- Step 1: Find defective call IDs on the online DB ---
    with online_db.session_scope() as online_session:
        repo = RawCallRepository(online_session)
        quality_svc = DataQualityService(repo)
        defective_group_id = quality_svc.get_or_create_defective_group()

        defective_members = online_session.query(GroupMember).filter_by(
            group_id=defective_group_id
        ).all()
        defective_call_ids = [m.call_id for m in defective_members]

    if not defective_call_ids:
        print("No records tagged as defective_data in the online DB. Nothing to do.")
        return

    print(f"Found {len(defective_call_ids)} defective record(s) in online DB (group id: {defective_group_id})")

    # --- Step 2: Copy to local DB ---
    print("Copying to local DB...")
    with online_db.session_scope() as online_session:
        with local_db.session_scope() as local_session:
            copied = copy_calls_to_local(
                online_session, local_session, defective_call_ids, args.dry_run
            )
            if not args.dry_run:
                local_session.commit()

    print(f"  {'Would copy' if args.dry_run else 'Copied'} {copied} record(s) to local DB.")

    # --- Step 3: Delete from online DB ---
    if args.dry_run:
        print(f"  [DRY RUN] Would delete {len(defective_call_ids)} RawCall row(s) from Neon.")
        print("\nSummary (dry run):")
        print(f"  Defective records found : {len(defective_call_ids)}")
        print(f"  Would copy to local     : {copied}")
        print(f"  Would delete from Neon  : {len(defective_call_ids)}")
        print("\nRun without --dry-run to execute.")
        return

    print("Deleting from online DB (cascades will clean up GroupMembers, EmbeddingVectors, etc.)...")
    deleted = 0
    with online_db.session_scope() as online_session:
        # Delete in chunks to avoid very large IN clauses
        chunk_size = 100
        for i in range(0, len(defective_call_ids), chunk_size):
            chunk = defective_call_ids[i : i + chunk_size]
            rows = online_session.query(RawCall).filter(RawCall.id.in_(chunk)).all()
            for row in rows:
                online_session.delete(row)
                deleted += 1
            online_session.flush()
        online_session.commit()

    print(f"  Deleted {deleted} record(s) from Neon.")

    print("\nSummary:")
    print(f"  Defective records found : {len(defective_call_ids)}")
    print(f"  Copied to local DB      : {copied}")
    print(f"  Deleted from Neon       : {deleted}")


if __name__ == "__main__":
    main()
