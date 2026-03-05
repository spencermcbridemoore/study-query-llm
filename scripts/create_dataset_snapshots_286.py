#!/usr/bin/env python3
"""Create idempotent 286-entry dataset snapshots for dbpedia and estela.

Creates two `dataset_snapshot` groups and stores exact sample manifests:
- dbpedia_286_seed<seed>_labeled
- estela_286_seed<seed>_unlabeled

Idempotency:
- If a snapshot with same name exists and manifest hash matches, reuse it.
- If same name exists with a different hash, fail with a clear error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.provenance_service import (
    LABEL_MODE_LABELED,
    LABEL_MODE_UNLABELED,
    GROUP_TYPE_DATASET_SNAPSHOT,
    ProvenanceService,
)
from study_query_llm.utils.estela_loader import load_estela_dict
from study_query_llm.utils.text_utils import flatten_prompt_dict


def _safe_text(text: str) -> bool:
    return isinstance(text, str) and 10 < len(text.strip().replace("\x00", "")) <= 1000


def _manifest_hash(snapshot_name: str, entries: List[Dict]) -> str:
    payload = {"snapshot_name": snapshot_name, "entries": entries}
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _sample_indices(total: int, sample_size: int, seed: int) -> np.ndarray:
    if total < sample_size:
        raise ValueError(f"Not enough rows to sample: total={total}, requested={sample_size}")
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(np.arange(total), size=sample_size, replace=False))


def _build_dbpedia_entries(sample_size: int, seed: int) -> List[Dict]:
    ds = load_dataset("dbpedia_14", split="train")
    filtered: List[Tuple[int, str, int]] = []
    for i, item in enumerate(ds):
        text = item.get("content", "")
        label = item.get("label", None)
        if _safe_text(text) and label is not None:
            filtered.append((i, text, int(label)))
    sample_idx = _sample_indices(len(filtered), sample_size, seed)
    entries: List[Dict] = []
    for pos, idx in enumerate(sample_idx.tolist()):
        src_id, text, label = filtered[idx]
        entries.append(
            {
                "position": pos,
                "source_id": int(src_id),
                "text": text,
                "label": int(label),
            }
        )
    return entries


def _build_estela_entries(sample_size: int, seed: int, pkl_path: Path) -> List[Dict]:
    data = load_estela_dict(pkl_path=str(pkl_path))
    flat = flatten_prompt_dict(data)
    rows: List[Tuple[str, str]] = []
    for key_tuple, text in flat.items():
        if _safe_text(text):
            source_id = "/".join(str(k) for k in key_tuple)
            rows.append((source_id, text))
    sample_idx = _sample_indices(len(rows), sample_size, seed)
    entries: List[Dict] = []
    for pos, idx in enumerate(sample_idx.tolist()):
        source_id, text = rows[idx]
        entries.append(
            {
                "position": pos,
                "source_id": source_id,
                "text": text,
                "label": None,
            }
        )
    return entries


def _existing_snapshot_with_hash(
    session,
    snapshot_name: str,
    expected_hash: str,
) -> int | None:
    group = (
        session.query(Group)
        .filter(
            Group.group_type == GROUP_TYPE_DATASET_SNAPSHOT,
            Group.name == snapshot_name,
        )
        .first()
    )
    if not group:
        return None

    manifest = (
        session.query(CallArtifact)
        .filter(
            CallArtifact.artifact_type == "dataset_snapshot_manifest",
            CallArtifact.metadata_json["group_id"].astext == str(group.id),
        )
        .first()
    )
    if not manifest:
        raise ValueError(
            f"Snapshot '{snapshot_name}' exists (id={group.id}) but has no manifest artifact"
        )

    manifest_hash = (manifest.metadata_json or {}).get("manifest_hash")
    if manifest_hash != expected_hash:
        raise ValueError(
            f"Snapshot '{snapshot_name}' exists (id={group.id}) with different manifest hash "
            f"(expected={expected_hash}, found={manifest_hash})"
        )
    return int(group.id)


def _create_or_reuse_snapshot(
    session,
    *,
    snapshot_name: str,
    source_dataset: str,
    label_mode: str,
    sample_size: int,
    seed: int,
    entries: List[Dict],
) -> int:
    repo = RawCallRepository(session)
    provenance = ProvenanceService(repo)
    artifacts = ArtifactService(repository=repo)

    hash_value = _manifest_hash(snapshot_name, entries)
    existing_id = _existing_snapshot_with_hash(session, snapshot_name, hash_value)
    if existing_id is not None:
        print(f"[SKIP] Reusing snapshot '{snapshot_name}' (id={existing_id})")
        return existing_id

    labels = [e["label"] for e in entries if e.get("label") is not None]
    label_stats = None
    if labels:
        unique, counts = np.unique(np.array(labels), return_counts=True)
        label_stats = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    metadata = {
        "has_ground_truth": label_mode == LABEL_MODE_LABELED,
        "manifest_hash": hash_value,
    }
    if label_stats:
        metadata["label_stats"] = label_stats

    snapshot_id = provenance.create_dataset_snapshot_group(
        snapshot_name=snapshot_name,
        source_dataset=source_dataset,
        sample_size=sample_size,
        label_mode=label_mode,
        sampling_method="seeded_random_filtered_10_1000_chars",
        sampling_seed=seed,
        metadata=metadata,
    )
    artifacts.store_dataset_snapshot_manifest(
        snapshot_group_id=snapshot_id,
        snapshot_name=snapshot_name,
        entries=entries,
        metadata={
            "source_dataset": source_dataset,
            "label_mode": label_mode,
            "sampling_seed": seed,
        },
    )
    print(f"[OK] Created snapshot '{snapshot_name}' (id={snapshot_id})")
    return snapshot_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/reuse 286-entry dataset snapshots")
    parser.add_argument("--sample-size", type=int, default=286)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--estela-pkl",
        type=str,
        default=str(Path("notebooks") / "estela_prompt_data.pkl"),
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    dbpedia_name = f"dbpedia_{args.sample_size}_seed{args.seed}_labeled"
    estela_name = f"estela_{args.sample_size}_seed{args.seed}_unlabeled"

    dbpedia_entries = _build_dbpedia_entries(args.sample_size, args.seed)
    estela_entries = _build_estela_entries(
        args.sample_size,
        args.seed,
        pkl_path=Path(args.estela_pkl),
    )

    with db.session_scope() as session:
        dbpedia_id = _create_or_reuse_snapshot(
            session,
            snapshot_name=dbpedia_name,
            source_dataset="dbpedia",
            label_mode=LABEL_MODE_LABELED,
            sample_size=args.sample_size,
            seed=args.seed,
            entries=dbpedia_entries,
        )
        estela_id = _create_or_reuse_snapshot(
            session,
            snapshot_name=estela_name,
            source_dataset="estela",
            label_mode=LABEL_MODE_UNLABELED,
            sample_size=args.sample_size,
            seed=args.seed,
            entries=estela_entries,
        )

    print("\nSnapshot IDs:")
    print(f"  dbpedia_snapshot_id={dbpedia_id}")
    print(f"  estela_snapshot_id={estela_id}")


if __name__ == "__main__":
    main()
