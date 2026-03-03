#!/usr/bin/env python3
"""
Label all existing clustering_run groups as pre-centroid-fix.

Sets metadata_json["centroid_fix_era"] = "pre_fix" on every clustering_run
that doesn't already have centroid_fix_era == "post_fix". This allows
downstream tools (SweepQueryService, check scripts, Panel UI) to exclude
pre-fix data when comparing summarizer results.

Usage:
    python scripts/label_pre_fix_runs.py --dry-run   # preview, no changes
    python scripts/label_pre_fix_runs.py              # execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as sa_text

from study_query_llm.config import config
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label existing clustering_run groups as pre-centroid-fix"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview which runs would be labeled without writing",
    )
    args = parser.parse_args()

    db = DatabaseConnectionV2(config.database.connection_string)
    db.init_db()

    with db.session_scope() as session:
        # All clustering_run groups with a cosine_kllmeans algorithm
        runs = (
            session.query(Group)
            .filter(
                Group.group_type == "clustering_run",
                sa_text("metadata_json->>'algorithm' LIKE :alg"),
            )
            .params(alg="cosine_kllmeans%")
            .order_by(Group.id)
            .all()
        )

        if not runs:
            print("No clustering_run groups found. Nothing to label.")
            return 0

        already_post = 0
        already_pre = 0
        to_label: list[Group] = []

        for run in runs:
            meta = run.metadata_json or {}
            era = meta.get("centroid_fix_era")
            if era == "post_fix":
                already_post += 1
            elif era == "pre_fix":
                already_pre += 1
            else:
                to_label.append(run)

        print(f"Total clustering_run groups: {len(runs)}")
        print(f"  Already labeled post_fix : {already_post}")
        print(f"  Already labeled pre_fix  : {already_pre}")
        print(f"  To be labeled pre_fix    : {len(to_label)}")

        if not to_label:
            print("\nNothing to label.")
            return 0

        if args.dry_run:
            print("\n[DRY RUN] Would label the following runs as pre_fix:")
            max_show = 20
            for run in to_label[:max_show]:
                meta = run.metadata_json or {}
                print(
                    f"  id={run.id}  {meta.get('dataset', '?')}/"
                    f"{meta.get('embedding_engine', '?')}/"
                    f"{meta.get('summarizer', '?')}  "
                    f"name={run.name!r}"
                )
            if len(to_label) > max_show:
                print(f"  ... and {len(to_label) - max_show} more.")
            print("\nRun without --dry-run to execute.")
            return 0

        labeled = 0
        for run in to_label:
            meta = dict(run.metadata_json or {})
            meta["centroid_fix_era"] = "pre_fix"
            run.metadata_json = meta
            labeled += 1

        session.flush()
        session.commit()

        print(f"\nLabeled {labeled} clustering_run group(s) as pre_fix.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
