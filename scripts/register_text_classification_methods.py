#!/usr/bin/env python3
"""Idempotently register the register-only text-classification method catalog.

Registers the 5 register-only text-classification ``MethodDefinition`` rows
(``knn_prototype_classifier``, ``linear_probe_logreg``,
``label_embedding_zero_shot``, ``prompted_llm_classifier``,
``mixture_of_experts_classifier``). No composite recipes are registered;
none are defined yet for this family.

Definitions only: this script does not wire any execution path. It exists so
future runners (DB jobs, LangGraph nodes, notebooks) share canonical
``(name, version)`` identities and parameter schemas from the moment a
method first appears in the registry.

Safe to re-run: registration skips rows already present at the same
``(name, version)`` pair.

Usage:
    python scripts/register_text_classification_methods.py
    python scripts/register_text_classification_methods.py --dry-run
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

from study_query_llm.algorithms.text_classification_methods import (
    TEXT_CLASSIFICATION_METHODS,
    register_text_classification_methods,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Register the register-only text-classification methods. "
            "Definitions only; no execution path is wired."
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
    print("Registering text-classification methods (register-only subset)")
    print(f"DATABASE_URL set: yes  dry_run={args.dry_run}")
    print("=" * 60)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        if args.dry_run:
            print("[dry-run] Methods that would be registered if missing:")
            for spec in TEXT_CLASSIFICATION_METHODS:
                key = f"{spec['name']}@{spec['version']}"
                existing = method_svc.get_method(
                    spec["name"], version=spec["version"]
                )
                status = "present" if existing is not None else "missing"
                print(f"  - {key}: {status}")
            session.rollback()
            return 0

        registered = register_text_classification_methods(method_svc)
        print("Text-classification method ids:")
        for key, mid in sorted(registered.items()):
            print(f"  - {key}: id={mid}")

    print("=" * 60)
    print("Registration complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
