#!/usr/bin/env python3
"""Validate and optionally backfill dataset snapshot linkage on clustering runs.

Default behavior is read-only validation.
Use --apply to write metadata_json.dataset_snapshot_ids and create depends_on links
for unlinked/unannotated runs with unambiguous snapshot matches.
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
from study_query_llm.db.models_v2 import Group, GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
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


def _normalize_snapshot_ids(raw_ids) -> List[int]:
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, (list, tuple, set)):
        raw_ids = [raw_ids]
    out: List[int] = []
    for sid in raw_ids:
        try:
            out.append(int(sid))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _load_run_snapshot_links(session) -> Dict[int, List[int]]:
    rows = (
        session.query(
            GroupLink.parent_group_id,
            GroupLink.child_group_id,
        )
        .join(Group, Group.id == GroupLink.child_group_id)
        .filter(
            GroupLink.link_type == "depends_on",
            Group.group_type == GROUP_TYPE_DATASET_SNAPSHOT,
        )
        .all()
    )
    out: Dict[int, List[int]] = {}
    for parent_group_id, child_group_id in rows:
        run_id = int(parent_group_id)
        snapshot_id = int(child_group_id)
        out.setdefault(run_id, []).append(snapshot_id)
    for run_id in list(out.keys()):
        out[run_id] = sorted(set(out[run_id]))
    return out


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

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        snapshot_index = _load_snapshots(session)
        run_snapshot_links = _load_run_snapshot_links(session)

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
        mismatch_meta_without_link = 0
        mismatch_link_without_meta = 0
        mismatch_ids_disagree = 0

        for run in runs:
            meta = dict(run.metadata_json or {})
            dataset = str(meta.get("dataset", ""))
            n_samples = int(meta.get("n_samples", 0) or 0)
            metadata_snapshot_ids = _normalize_snapshot_ids(meta.get("dataset_snapshot_ids"))
            linked_snapshot_ids = _normalize_snapshot_ids(run_snapshot_links.get(int(run.id), []))
            has_meta = bool(metadata_snapshot_ids)
            has_links = bool(linked_snapshot_ids)
            if has_meta and not has_links:
                mismatch_meta_without_link += 1
                continue
            if has_links and not has_meta:
                mismatch_link_without_meta += 1
                continue
            if has_meta and has_links:
                if set(metadata_snapshot_ids) != set(linked_snapshot_ids):
                    mismatch_ids_disagree += 1
                    continue
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
        print(f"  mismatch_meta_without_link={mismatch_meta_without_link}")
        print(f"  mismatch_link_without_meta={mismatch_link_without_meta}")
        print(f"  mismatch_ids_disagree={mismatch_ids_disagree}")
        if args.apply:
            print(f"  backfilled={backfilled}")
        else:
            print("  backfilled=0 (dry-run; use --apply)")


if __name__ == "__main__":
    main()
