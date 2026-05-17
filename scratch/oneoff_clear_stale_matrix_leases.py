#!/usr/bin/env python
"""One-off cleanup for stale matrix-level embedding cache lease rows.

Default mode is --dry-run. Use --apply to delete candidates after review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import EmbeddingCacheLease
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent


@dataclass
class LeaseRow:
    cache_key: str
    lease_owner: str
    lease_expires_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "lease_owner": self.lease_owner,
            "lease_expires_at": (
                self.lease_expires_at.isoformat() if self.lease_expires_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def _resolve_database_url(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    for key in ("DATABASE_URL", "CANONICAL_DATABASE_URL", "JETSTREAM_DATABASE_URL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    raise ValueError(
        "database URL is required (pass --database-url or set DATABASE_URL/CANONICAL_DATABASE_URL/JETSTREAM_DATABASE_URL)"
    )


def _stale_per_repository_semantics(row: EmbeddingCacheLease, *, now_utc: datetime) -> bool:
    # Mirror try_acquire_embedding_cache_lease takeover predicate for staleness.
    expiry = RawCallRepository._coerce_utc_aware(row.lease_expires_at)
    owner = str(row.lease_owner or "").strip()
    return bool(expiry is None or not owner or expiry <= now_utc)


def _list_matrix_stale_rows(db: DatabaseConnectionV2) -> list[LeaseRow]:
    now_utc = datetime.now(timezone.utc)
    stale_rows: list[LeaseRow] = []
    with db.session_scope() as session:
        rows = (
            session.query(EmbeddingCacheLease)
            .filter(EmbeddingCacheLease.cache_key.like("embed_matrix:%"))
            .order_by(EmbeddingCacheLease.lease_expires_at.asc())
            .all()
        )
        for row in rows:
            if not _stale_per_repository_semantics(row, now_utc=now_utc):
                continue
            stale_rows.append(
                LeaseRow(
                    cache_key=str(row.cache_key),
                    lease_owner=str(row.lease_owner or ""),
                    lease_expires_at=RawCallRepository._coerce_utc_aware(row.lease_expires_at),
                    created_at=RawCallRepository._coerce_utc_aware(row.created_at),
                    updated_at=RawCallRepository._coerce_utc_aware(row.updated_at),
                )
            )
    return stale_rows


def _delete_rows(db: DatabaseConnectionV2, rows: list[LeaseRow]) -> int:
    keys = [row.cache_key for row in rows]
    if not keys:
        return 0
    with db.session_scope() as session:
        deleted = (
            session.query(EmbeddingCacheLease)
            .filter(EmbeddingCacheLease.cache_key.in_(keys))
            .delete(synchronize_session=False)
        )
    return int(deleted or 0)


def _print_human_report(
    *,
    mode: str,
    db_url: str,
    stale_rows: list[LeaseRow],
    deleted_count: int | None,
) -> None:
    print(f"mode={mode}")
    print(f"database_url={db_url}")
    print(f"stale_candidate_count={len(stale_rows)}")
    if deleted_count is not None:
        print(f"deleted_count={deleted_count}")
    if not stale_rows:
        print("rows=[]")
        return
    print("rows:")
    for row in stale_rows:
        payload = row.to_dict()
        print(
            f"  - cache_key={payload['cache_key']} owner={payload['lease_owner']} "
            f"expires_at={payload['lease_expires_at']}"
        )


def _write_json_report(
    *,
    output_path: Path | None,
    mode: str,
    stale_rows: list[LeaseRow],
    deleted_count: int | None,
) -> None:
    if output_path is None:
        return
    payload = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale_candidate_count": len(stale_rows),
        "deleted_count": deleted_count,
        "rows": [row.to_dict() for row in stale_rows],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="List stale embed_matrix lease rows (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Delete stale embed_matrix lease rows discovered by current criteria.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="DB URL override (defaults to DATABASE_URL/CANONICAL_DATABASE_URL/JETSTREAM_DATABASE_URL).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply = bool(args.apply)
    mode = "apply" if apply else "dry-run"
    db_url = _resolve_database_url(args.database_url)
    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()
    stale_rows = _list_matrix_stale_rows(db)
    deleted_count = _delete_rows(db, stale_rows) if apply else None
    _print_human_report(
        mode=mode,
        db_url=db_url,
        stale_rows=stale_rows,
        deleted_count=deleted_count,
    )
    output_path = Path(args.output_json).resolve() if args.output_json else None
    _write_json_report(
        output_path=output_path,
        mode=mode,
        stale_rows=stale_rows,
        deleted_count=deleted_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
