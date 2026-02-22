#!/usr/bin/env python3
"""
Incremental sync from online Neon PostgreSQL to local backup PostgreSQL.

Pulls all records from the online DB that are newer than what's already in
the local DB. Only downloads, never uploads. Safe to run repeatedly
(uses ON CONFLICT DO NOTHING so duplicates are skipped automatically).

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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import create_engine, func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker, make_transient
from sqlalchemy.pool import NullPool

load_dotenv()


def get_max_local_id(local_engine) -> int:
    """Return the highest raw_call id already in the local DB, or 0 if empty."""
    from study_query_llm.db.models_v2 import RawCall
    Session = sessionmaker(bind=local_engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        result = session.query(func.max(RawCall.id)).scalar()
        return result or 0
    finally:
        session.close()


def warmup_neon(online_engine) -> None:
    """
    Send a trivial query to wake up Neon's serverless compute.

    Neon free-tier compute sleeps after ~5 min of inactivity and can take
    30-120 s to cold-start. This warmup prevents the first batch query from
    appearing to hang silently.
    """
    print("Warming up Neon connection (may take up to 120s on cold start)...", flush=True)
    t0 = time.time()
    with online_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print(f"Neon connected in {time.time()-t0:.1f}s", flush=True)


def fetch_batch(online_engine, min_id: int, batch_size: int) -> dict:
    """
    Open a fresh short-lived Neon connection, fetch one batch, close it.

    Returns dict with: calls, members, group_ids, vectors, artifacts.
    Empty dict means no more records.
    """
    from study_query_llm.db.models_v2 import (
        RawCall, GroupMember, EmbeddingVector, CallArtifact
    )

    Session = sessionmaker(bind=online_engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        online_calls = (
            session.query(RawCall)
            .filter(RawCall.id > min_id)
            .order_by(RawCall.id)
            .limit(batch_size)
            .all()
        )
        if not online_calls:
            return {}

        call_ids = [c.id for c in online_calls]

        members = (
            session.query(GroupMember)
            .filter(GroupMember.call_id.in_(call_ids))
            .all()
        )
        group_ids = list({m.group_id for m in members})

        vectors = (
            session.query(EmbeddingVector)
            .filter(EmbeddingVector.call_id.in_(call_ids))
            .all()
        )
        artifacts = (
            session.query(CallArtifact)
            .filter(CallArtifact.call_id.in_(call_ids))
            .all()
        )

        # Detach all objects from session so data is accessible after close
        for obj in online_calls + members + vectors + artifacts:
            session.expunge(obj)
            make_transient(obj)

        return {
            "calls": online_calls,
            "members": members,
            "group_ids": group_ids,
            "vectors": vectors,
            "artifacts": artifacts,
        }
    finally:
        session.close()


def fetch_groups(online_engine, group_ids: list[int]) -> list:
    """Fetch Group rows from Neon in a fresh connection."""
    if not group_ids:
        return []
    from study_query_llm.db.models_v2 import Group
    Session = sessionmaker(bind=online_engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        groups = session.query(Group).filter(Group.id.in_(group_ids)).all()
        for g in groups:
            session.expunge(g)
            make_transient(g)
        return groups
    finally:
        session.close()


def write_batch(local_engine, batch: dict, online_groups: list) -> None:
    """
    Insert a fetched batch into the local DB using ON CONFLICT DO NOTHING.

    This makes the sync idempotent — safe to re-run if interrupted.
    """
    from study_query_llm.db.models_v2 import (
        RawCall, Group, GroupMember, EmbeddingVector, CallArtifact
    )

    Session = sessionmaker(bind=local_engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        # Upsert Groups (match on group_type + name)
        group_id_map: dict[int, int] = {}
        for g in online_groups:
            existing = session.query(Group).filter_by(
                group_type=g.group_type, name=g.name
            ).first()
            if existing:
                group_id_map[g.id] = existing.id
            else:
                local_group = Group(
                    group_type=g.group_type, name=g.name,
                    description=g.description, created_at=g.created_at,
                    metadata_json=g.metadata_json,
                )
                session.add(local_group)
                session.flush()
                session.refresh(local_group)
                group_id_map[g.id] = local_group.id

        # Insert RawCalls — ON CONFLICT DO NOTHING via raw SQL for efficiency
        if batch["calls"]:
            stmt = pg_insert(RawCall).values([
                dict(
                    id=c.id, provider=c.provider, model=c.model,
                    modality=c.modality, status=c.status,
                    request_json=c.request_json, response_json=c.response_json,
                    error_json=c.error_json, latency_ms=c.latency_ms,
                    tokens_json=c.tokens_json, metadata_json=c.metadata_json,
                    created_at=c.created_at,
                )
                for c in batch["calls"]
            ]).on_conflict_do_nothing(index_elements=["id"])
            session.execute(stmt)

        # Insert GroupMembers — ON CONFLICT DO NOTHING
        if batch["members"]:
            stmt = pg_insert(GroupMember).values([
                dict(
                    group_id=group_id_map.get(m.group_id, m.group_id),
                    call_id=m.call_id,
                    added_at=m.added_at,
                    position=m.position,
                    role=m.role,
                )
                for m in batch["members"]
            ]).on_conflict_do_nothing(index_elements=["group_id", "call_id"])
            session.execute(stmt)

        # Insert EmbeddingVectors
        for v in batch["vectors"]:
            existing = session.query(EmbeddingVector).filter_by(call_id=v.call_id).first()
            if not existing:
                session.add(EmbeddingVector(
                    call_id=v.call_id, vector=v.vector,
                    dimension=v.dimension, norm=v.norm,
                    metadata_json=v.metadata_json,
                ))

        # Insert CallArtifacts
        for a in batch["artifacts"]:
            existing = session.query(CallArtifact).filter_by(
                call_id=a.call_id, uri=a.uri
            ).first()
            if not existing:
                session.add(CallArtifact(
                    call_id=a.call_id, artifact_type=a.artifact_type,
                    uri=a.uri, content_type=a.content_type,
                    byte_size=a.byte_size, metadata_json=a.metadata_json,
                ))

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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

    print(f"Online DB: ...@{online_url.split('@')[-1] if '@' in online_url else online_url}")
    print(f"Local  DB: ...@{local_url.split('@')[-1] if '@' in local_url else local_url}", flush=True)
    if args.dry_run:
        print("[DRY RUN] No changes will be written.\n", flush=True)

    # NullPool: fresh connection per operation, avoids long-lived transaction issues
    # with Neon's PgBouncer transaction-mode pooler
    online_engine = create_engine(
        online_url,
        poolclass=NullPool,
        connect_args={"connect_timeout": 120},
    )
    local_engine = create_engine(local_url, poolclass=NullPool)

    # Wake up Neon's serverless compute before the batch loop
    warmup_neon(online_engine)

    current_min = get_max_local_id(local_engine)
    print(f"Local max id : {current_min}", flush=True)
    print("Starting sync...", flush=True)

    total_synced = 0
    t_start = time.time()

    while True:
        batch = fetch_batch(online_engine, current_min, args.batch_size)
        if not batch:
            break

        count = len(batch["calls"])
        next_min = batch["calls"][-1].id

        if not args.dry_run:
            online_groups = fetch_groups(online_engine, batch["group_ids"])
            write_batch(local_engine, batch, online_groups)

        total_synced += count
        current_min = next_min
        elapsed = time.time() - t_start
        rate = total_synced / elapsed if elapsed > 0 else 0
        verb = "Would sync" if args.dry_run else "Synced"
        print(
            f"  {verb} {total_synced:,} records (last id: {current_min}, "
            f"{rate:.0f} rec/s)",
            flush=True,
        )

    action = "Would sync" if args.dry_run else "Synced"
    elapsed = time.time() - t_start
    print(f"\n{action} {total_synced:,} records in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
