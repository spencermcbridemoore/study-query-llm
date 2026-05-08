#!/usr/bin/env python3
"""Idempotently register inference method definitions.

Registers stage-1 and stage-2 inference method identities so runtime writers
do not lazily create method rows during execution paths.

Usage:
    python scripts/register_inference_methods.py
    python scripts/register_inference_methods.py --dry-run
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

from study_query_llm.algorithms.inference_methods import (
    INFERENCE_METHODS,
    register_inference_methods,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.services.method_service import MethodService


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Register inference method identities for polymorphic execution "
            "lanes. Definitions only; no runtime invocation."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be registered without writing.",
    )
    args = parser.parse_args()

    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Registering inference methods")
    print(f"DATABASE_URL set: yes  dry_run={args.dry_run}")
    print("=" * 60)

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        if args.dry_run:
            print("[dry-run] Methods that would be registered if missing:")
            for spec in INFERENCE_METHODS:
                key = f"{spec['name']}@{spec['version']}"
                existing = method_svc.get_method(
                    spec["name"], version=spec["version"]
                )
                status = "present" if existing is not None else "missing"
                print(f"  - {key}: {status}")
            session.rollback()
            return 0

        registered = register_inference_methods(method_svc)
        print("Inference method ids:")
        for key, mid in sorted(registered.items()):
            print(f"  - {key}: id={mid}")

    print("=" * 60)
    print("Registration complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

