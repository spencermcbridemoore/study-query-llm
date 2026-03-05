#!/usr/bin/env python3
"""Validate and optionally backfill dataset snapshot linkage on clustering runs.

Default behavior is read-only validation.
Use --apply to write metadata_json.dataset_snapshot_ids and create depends_on links.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_CLUSTERING_RUN,
    GROUP_TYPE_DATASET_SNAPSHOT,
    ProvenanceService,
)


def _load_snapshots(session) -> Dict[Tuple[str, int], List[int]]:
    snapshots = (
        session.query(Group)
        .filter(Group.group_type == GROUP_TYPE_DATASET_SNAPSHOT)
        .order_by(Group.created_at.desc())
        .all()
    )
    index: Dict[Tuple[str, int], List[int]] = {}
    for snap in snapshots:
        meta = snap.metadata_json or {}
        dataset = meta.get("source_dataset")
        sample_size = meta.get("sample_size")
        if not dataset or sample_size is None:
            continue
        key = (str(dataset), int(sample_size))
        index.setdefault(key, []).append(int(snap.id))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and optionally backfill run -> dataset snapshot linkage",
    )
    parser.add_argument("--apply", action="store_true", help="Apply backfill changes")
    parser.add_argument(
        "--datasets",
        type=str,
        default="dbpedia,estela",
        help="Comma-separated dataset names eligible for auto-backfill",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=286,
        help="Expected run sample size for backfill matching",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")

    target_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        snapshot_index = _load_snapshots(session)

        runs = (
            session.query(Group)
            .filter(Group.group_type == GROUP_TYPE_CLUSTERING_RUN)
            .all()
        )

        checked = 0
        already_linked = 0
        missing = 0
        backfilled = 0
        ambiguous = 0
        skipped = 0

        for run in runs:
            meta = dict(run.metadata_json or {})
            dataset = str(meta.get("dataset", ""))
            n_samples = int(meta.get("n_samples", 0) or 0)
            existing_ids = meta.get("dataset_snapshot_ids")

            if existing_ids:
                already_linked += 1
                continue

            checked += 1
            if dataset not in target_datasets or n_samples != args.sample_size:
                skipped += 1
                continue

            missing += 1
            candidates = snapshot_index.get((dataset, args.sample_size), [])
            if len(candidates) != 1:
                ambiguous += 1
                continue

            snapshot_id = int(candidates[0])
            if args.apply:
                meta["dataset_snapshot_ids"] = [snapshot_id]
                run.metadata_json = meta
                session.flush()
                provenance.link_run_to_dataset_snapshot(int(run.id), snapshot_id)
                backfilled += 1

        print("Run snapshot linkage validation")
        print(f"  checked_missing_or_unlinked_runs={checked}")
        print(f"  already_linked_runs={already_linked}")
        print(f"  eligible_missing_runs={missing}")
        print(f"  ambiguous_or_unmatched={ambiguous}")
        print(f"  skipped_non_target={skipped}")
        if args.apply:
            print(f"  backfilled={backfilled}")
        else:
            print("  backfilled=0 (dry-run; use --apply)")


if __name__ == "__main__":
    main()
