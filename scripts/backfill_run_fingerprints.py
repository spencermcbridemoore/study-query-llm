#!/usr/bin/env python3
"""Backfill fingerprint_json/fingerprint_hash on existing provenanced_runs rows.

Idempotent: skips rows that already have a fingerprint_hash.

Usage:
    python scripts/backfill_run_fingerprints.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import MethodDefinition, ProvenancedRun
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.services.provenanced_run_service import canonical_run_fingerprint


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(description="Backfill run fingerprints")
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not write")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    args = parser.parse_args()

    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()

    updated = 0
    skipped = 0
    errors = 0

    with db.session_scope() as session:
        q = session.query(ProvenancedRun).filter(ProvenancedRun.fingerprint_hash.is_(None))
        if args.limit > 0:
            q = q.limit(args.limit)
        rows = q.all()

        method_cache: dict[int, tuple[str, str | None]] = {}

        for row in rows:
            try:
                method_name = None
                method_version = None
                if row.method_definition_id is not None:
                    if row.method_definition_id not in method_cache:
                        md = (
                            session.query(MethodDefinition)
                            .filter_by(id=row.method_definition_id)
                            .first()
                        )
                        if md:
                            method_cache[row.method_definition_id] = (
                                str(md.name),
                                str(md.version) if md.version else None,
                            )
                        else:
                            method_cache[row.method_definition_id] = (None, None)
                    method_name, method_version = method_cache[row.method_definition_id]

                meta = dict(row.metadata_json or {})
                fp_json, fp_hash = canonical_run_fingerprint(
                    method_name=method_name,
                    method_version=method_version,
                    config_json=dict(row.config_json or {}),
                    input_snapshot_group_id=row.input_snapshot_group_id,
                    manifest_hash=meta.get("manifest_hash"),
                    data_regime=meta.get("data_regime"),
                    determinism_class=str(row.determinism_class or "non_deterministic"),
                )

                if args.dry_run:
                    print(f"  [DRY] id={row.id} fp_hash={fp_hash[:16]}...")
                else:
                    row.fingerprint_json = fp_json
                    row.fingerprint_hash = fp_hash
                updated += 1
            except Exception as exc:
                print(f"  [ERROR] id={row.id}: {exc}", file=sys.stderr)
                errors += 1

        if not args.dry_run:
            session.flush()

    print(f"Done. updated={updated} skipped={skipped} errors={errors} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
