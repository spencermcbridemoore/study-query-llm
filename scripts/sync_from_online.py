#!/usr/bin/env python3
"""
Incremental sync from online Neon PostgreSQL to local backup PostgreSQL.

Pulls all records from the online DB that are newer than what's already in
the local DB. Only downloads, never uploads. Safe to run repeatedly.

Prerequisites:
    docker compose --profile postgres up -d db
    python scripts/init_local_db.py  (first time only)

Usage:
    python scripts/sync_from_online.py
    python scripts/sync_from_online.py --dry-run
    python scripts/sync_from_online.py --batch-size 500
    python scripts/sync_from_online.py --online-url <url> --local-url <url>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

load_dotenv()


def get_max_local_id(local_session) -> int:
    """Return the highest raw_call id already in the local DB, or 0 if empty."""
    from study_query_llm.db.models_v2 import RawCall
    result = local_session.query(func.max(RawCall.id)).scalar()
    return result or 0


def sync_groups(online_session, local_session, group_ids: list[int], dry_run: bool) -> dict[int, int]:
    """
    Copy groups from online to local by online id.

    Returns a mapping of online_group_id -> local_group_id so GroupMember
    rows can be remapped correctly (local auto-increment may differ).
    """
    from study_query_llm.db.models_v2 import Group

    if not group_ids:
        return {}

    online_groups = online_session.query(Group).filter(Group.id.in_(group_ids)).all()
    id_map: dict[int, int] = {}

    for g in online_groups:
        # Check if already exists locally by (group_type, name)
        existing = local_session.query(Group).filter_by(
            group_type=g.group_type,
            name=g.name,
        ).first()

        if existing:
            id_map[g.id] = existing.id
            continue

        if dry_run:
            id_map[g.id] = g.id  # placeholder for dry-run reporting
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


def sync_batch(
    online_session,
    local_session,
    min_id: int,
    batch_size: int,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Pull one batch of RawCalls (id > min_id) from online and insert into local.

    Returns (records_synced, next_min_id).
    """
    from study_query_llm.db.models_v2 import (
        RawCall, GroupMember, EmbeddingVector, CallArtifact
    )

    # Fetch the next batch of calls from online
    online_calls = (
        online_session.query(RawCall)
        .filter(RawCall.id > min_id)
        .order_by(RawCall.id)
        .limit(batch_size)
        .all()
    )

    if not online_calls:
        return 0, min_id

    call_ids = [c.id for c in online_calls]
    next_min = call_ids[-1]

    if not dry_run:
        # Insert RawCalls — skip any that somehow already exist
        for c in online_calls:
            exists = local_session.query(RawCall).filter_by(id=c.id).first()
            if exists:
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
        local_session.flush()

    # Pull associated GroupMembers and sync their Groups first
    members = (
        online_session.query(GroupMember)
        .filter(GroupMember.call_id.in_(call_ids))
        .all()
    )
    group_ids = list({m.group_id for m in members})
    group_id_map = sync_groups(online_session, local_session, group_ids, dry_run)

    if not dry_run:
        for m in members:
            local_group_id = group_id_map.get(m.group_id, m.group_id)
            exists = local_session.query(GroupMember).filter_by(
                group_id=local_group_id, call_id=m.call_id
            ).first()
            if exists:
                continue
            local_member = GroupMember(
                group_id=local_group_id,
                call_id=m.call_id,
                added_at=m.added_at,
                position=m.position,
                role=m.role,
            )
            local_session.add(local_member)

    # Pull EmbeddingVectors
    vectors = (
        online_session.query(EmbeddingVector)
        .filter(EmbeddingVector.call_id.in_(call_ids))
        .all()
    )
    if not dry_run:
        for v in vectors:
            exists = local_session.query(EmbeddingVector).filter_by(
                call_id=v.call_id
            ).first()
            if exists:
                continue
            local_vec = EmbeddingVector(
                call_id=v.call_id,
                vector=v.vector,
                dimension=v.dimension,
                norm=v.norm,
                metadata_json=v.metadata_json,
            )
            local_session.add(local_vec)

    # Pull CallArtifacts
    artifacts = (
        online_session.query(CallArtifact)
        .filter(CallArtifact.call_id.in_(call_ids))
        .all()
    )
    if not dry_run:
        for a in artifacts:
            exists = local_session.query(CallArtifact).filter_by(
                call_id=a.call_id, uri=a.uri
            ).first()
            if exists:
                continue
            local_art = CallArtifact(
                call_id=a.call_id,
                artifact_type=a.artifact_type,
                uri=a.uri,
                content_type=a.content_type,
                byte_size=a.byte_size,
                metadata_json=a.metadata_json,
            )
            local_session.add(local_art)

    if not dry_run:
        local_session.flush()

    return len(online_calls), next_min


def main():
    parser = argparse.ArgumentParser(description="Sync records from online Neon DB to local backup DB")
    parser.add_argument("--online-url", default=None, help="Online DB URL (default: DATABASE_URL)")
    parser.add_argument("--local-url", default=None, help="Local DB URL (default: LOCAL_DATABASE_URL)")
    parser.add_argument("--batch-size", type=int, default=200, help="Records per batch (default: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
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

    print(f"Online DB: ...@{online_url.split('@')[-1] if '@' in online_url else online_url}")
    print(f"Local  DB: ...@{local_url.split('@')[-1] if '@' in local_url else local_url}")
    if args.dry_run:
        print("[DRY RUN] No changes will be written.\n")

    online_db = DatabaseConnectionV2(online_url, enable_pgvector=False)
    local_db = DatabaseConnectionV2(local_url, enable_pgvector=False)

    total_synced = 0

    with online_db.session_scope() as online_session:
        with local_db.session_scope() as local_session:
            max_local = get_max_local_id(local_session)

            # Count how many records online have id > max_local
            from study_query_llm.db.models_v2 import RawCall
            online_count = (
                online_session.query(func.count(RawCall.id))
                .filter(RawCall.id > max_local)
                .scalar()
            )

            print(f"Local max id : {max_local}")
            print(f"Records to sync: {online_count}")

            if online_count == 0:
                print("Already up to date.")
                return

            current_min = max_local
            while True:
                count, current_min = sync_batch(
                    online_session,
                    local_session,
                    min_id=current_min,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                )
                if count == 0:
                    break
                total_synced += count
                verb = "Would sync" if args.dry_run else "Synced"
                print(f"  {verb} {total_synced} / {online_count} records (last id: {current_min})")

            if not args.dry_run:
                local_session.commit()

    action = "Would sync" if args.dry_run else "Synced"
    print(f"\n{action} {total_synced} records from online to local DB.")


if __name__ == "__main__":
    main()
