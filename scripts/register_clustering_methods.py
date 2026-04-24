#!/usr/bin/env python3
"""Idempotently register clustering component methods and the composite recipe.

Registers the 4 component MethodDefinition rows
(``mean_pool_tokens``, ``pca_svd_project``, ``kmeanspp_init``,
``k_llmmeans``) and ensures the composite ``cosine_kllmeans_no_pca`` method
carries its canonical recipe.

Safe to re-run: component registration skips rows already present at the
same ``(name, version)`` pair, and the composite recipe is only attached
when missing (matching recipes are no-ops; differing recipes warn rather
than overwrite -- bump the composite version explicitly if the recipe
needs to change).

Usage:
    python scripts/register_clustering_methods.py
    python scripts/register_clustering_methods.py --dry-run
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

from study_query_llm.algorithms.recipes import (
    CLUSTERING_COMPONENT_METHODS,
    COMPOSITE_RECIPES,
    ensure_composite_recipe,
    register_clustering_components,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Register clustering component methods and attach the "
            "composite recipe."
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
    print("Registering clustering component methods + composite recipe")
    print(f"DATABASE_URL set: yes  dry_run={args.dry_run}")
    print("=" * 60)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        if args.dry_run:
            print("[dry-run] Components that would be registered if missing:")
            for spec in CLUSTERING_COMPONENT_METHODS:
                key = f"{spec['name']}@{spec['version']}"
                existing = method_svc.get_method(
                    spec["name"], version=spec["version"]
                )
                status = "present" if existing is not None else "missing"
                print(f"  - {key}: {status}")
            print("[dry-run] Composite recipes that would be ensured:")
            for composite_name in COMPOSITE_RECIPES:
                existing = method_svc.get_method(composite_name, version="1.0")
                if existing is None:
                    status = "would register with recipe"
                elif not existing.recipe_json:
                    status = "would attach canonical recipe in place"
                else:
                    status = "already has a recipe (no change)"
                print(f"  - {composite_name}@1.0: {status}")
            session.rollback()
            return 0

        registered = register_clustering_components(method_svc)
        print("Component method ids:")
        for key, mid in sorted(registered.items()):
            print(f"  - {key}: id={mid}")

        print("Composite recipes:")
        for composite_name in COMPOSITE_RECIPES:
            mid = ensure_composite_recipe(
                method_svc,
                composite_name,
                composite_version="1.0",
                description=(
                    "Composite clustering pipeline "
                    f"({composite_name}); see method_definitions.recipe_json "
                    "for the stage list."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "k_range": {"type": "array", "items": {"type": "integer"}},
                        "selection_metric": {"type": "string"},
                        "selection_rule": {"type": "string"},
                        "hdbscan_min_cluster_size": {"type": "integer"},
                    },
                },
            )
            print(f"  - {composite_name}@1.0: id={mid}")

    print("=" * 60)
    print("Registration complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
