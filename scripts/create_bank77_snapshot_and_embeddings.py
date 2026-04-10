#!/usr/bin/env python3
"""Bootstrap BANK77 snapshot + full embeddings + intent means artifacts.

This script implements a lightweight, provenance-first bootstrap using existing
v2 entities (`Group`, `GroupLink`, `CallArtifact`) and service primitives:
- `ProvenanceService` for `dataset_snapshot` and `embedding_batch` groups
- `ArtifactService` for `dataset_snapshot_manifest`, `embedding_matrix`, and `metrics`
- `fetch_embeddings_async` helper for provider-backed embedding generation

Workflow:
1) Load `mteb/banking77` train+test deterministically.
2) Create/reuse one `dataset_snapshot` + manifest.
3) Create/reuse one full embedding matrix artifact (`N x d`).
4) Create/reuse one intent-mean matrix artifact (`77 x d`) + deterministic mapping metadata.
5) Verify readback, integrity metadata, and lineage links.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group, GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.embeddings.constants import CACHE_KEY_VERSION
from study_query_llm.services.embeddings.helpers import fetch_embeddings_async
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_DATASET_SNAPSHOT,
    LABEL_MODE_LABELED,
    ProvenanceService,
)

BANK77_DATASET_SOURCE = "mteb/banking77"
BANK77_EXPECTED_LABEL_COUNT = 77
BANK77_BOOTSTRAP_VERSION = "bank77_bootstrap_v1"
MAPPING_STEP_NAME = "bank77_intent_label_mapping"
MAPPING_TYPE = "bank77_intent_means_label_mapping"
SAMPLING_METHOD = "full_train_test_concat_deterministic"


@dataclass(frozen=True)
class MeansArtifactResult:
    means_batch_group_id: int
    means_artifact_id: int
    mapping_artifact_id: int
    means_shape: Tuple[int, int]


def _safe_text(value: Any) -> str:
    """Normalize text from dataset rows; raise when unusable."""
    if not isinstance(value, str):
        raise ValueError(f"Expected text string; got {type(value).__name__}")
    cleaned = value.replace("\x00", "").strip()
    if not cleaned:
        raise ValueError("Encountered empty text row in BANK77 source")
    return cleaned


def _slug(value: str) -> str:
    """Create a stable slug for group names."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    out = out.strip("-")
    return out or "unknown"


def _parse_chunk_sizes(raw: str) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid chunk size value: '{token}'") from exc
        if value <= 0:
            raise ValueError(f"Chunk size must be > 0, got {value}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError("At least one valid chunk size is required")
    return out


def _manifest_hash(snapshot_name: str, entries: List[Dict[str, Any]]) -> str:
    payload = {"snapshot_name": snapshot_name, "entries": entries}
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _coerce_group_id(metadata_json: Any) -> Optional[int]:
    if not isinstance(metadata_json, dict):
        return None
    raw = metadata_json.get("group_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _expected_byte_size(artifact: CallArtifact) -> Optional[int]:
    metadata = artifact.metadata_json or {}
    raw = metadata.get("byte_size")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    if artifact.byte_size is not None:
        try:
            return int(artifact.byte_size)
        except (TypeError, ValueError):
            return None
    return None


def _load_bank77_rows(
    dataset_source: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[int, str]]:
    """Load BANK77 rows with deterministic `split:index` source IDs."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - operational dependency
        raise RuntimeError(
            "datasets package is required. Install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_source)
    required_splits = ("train", "test")
    for split_name in required_splits:
        if split_name not in dataset:
            raise ValueError(
                f"Dataset '{dataset_source}' missing required split '{split_name}'"
            )

    label_id_to_intent: Dict[int, str] = {}
    rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}

    for split_name in required_splits:
        split = dataset[split_name]
        split_counts[split_name] = int(len(split))

        for idx, item in enumerate(split):
            text = _safe_text(item.get("text"))
            label_raw = item.get("label")
            intent_raw = item.get("label_text")
            if label_raw is None:
                raise ValueError(f"Dataset row missing label: split={split_name} idx={idx}")
            if intent_raw is None:
                raise ValueError(
                    f"Dataset row missing label_text: split={split_name} idx={idx}"
                )

            label_id = int(label_raw)
            intent = str(intent_raw).strip()
            if not intent:
                raise ValueError(
                    f"Dataset row has empty label_text: split={split_name} idx={idx}"
                )

            existing_intent = label_id_to_intent.get(label_id)
            if existing_intent is not None and existing_intent != intent:
                raise ValueError(
                    f"Label id {label_id} maps to multiple intents: "
                    f"'{existing_intent}' vs '{intent}'"
                )
            label_id_to_intent[label_id] = intent

            rows.append(
                {
                    "split": split_name,
                    "source_id": f"{split_name}:{idx}",
                    "text": text,
                    "label": label_id,
                    "intent": intent,
                }
            )

    train_label_names: Optional[List[str]] = None
    try:
        label_feature = dataset["train"].features["label"]
        names = getattr(label_feature, "names", None)
        if isinstance(names, list) and names:
            train_label_names = [str(name) for name in names]
    except Exception:
        train_label_names = None

    if train_label_names:
        for label_id, intent in enumerate(train_label_names):
            existing_intent = label_id_to_intent.get(int(label_id))
            if existing_intent is None:
                label_id_to_intent[int(label_id)] = intent
                continue
            if existing_intent != intent:
                raise ValueError(
                    f"Feature label name mismatch for id {label_id}: "
                    f"rows='{existing_intent}' features='{intent}'"
                )

    return rows, split_counts, label_id_to_intent


async def _run_ramp_probe_async(
    *,
    probe_texts: List[str],
    deployment: str,
    provider_name: str,
    chunk_sizes: List[int],
) -> List[Dict[str, Any]]:
    factory = ProviderFactory()
    embedding_provider = factory.create_embedding_provider(provider_name)
    results: List[Dict[str, Any]] = []
    async with embedding_provider:
        for chunk_size in chunk_sizes:
            start = time.perf_counter()
            call_count = 0
            error_message: Optional[str] = None
            try:
                for idx in range(0, len(probe_texts), int(chunk_size)):
                    batch = probe_texts[idx : idx + int(chunk_size)]
                    if not batch:
                        continue
                    await embedding_provider.create_embeddings(
                        batch,
                        deployment,
                    )
                    call_count += 1
            except Exception as exc:
                error_message = str(exc)
            elapsed = max(time.perf_counter() - start, 1e-9)
            rows_per_second = (
                float(len(probe_texts)) / elapsed if error_message is None else 0.0
            )
            results.append(
                {
                    "chunk_size": int(chunk_size),
                    "probe_rows": int(len(probe_texts)),
                    "api_calls": int(call_count),
                    "elapsed_seconds": float(elapsed),
                    "rows_per_second": float(rows_per_second),
                    "ok": error_message is None,
                    "error": error_message,
                }
            )
    return results


def _run_ramp_probe(
    *,
    probe_texts: List[str],
    deployment: str,
    provider_name: str,
    chunk_sizes: List[int],
) -> Tuple[int, List[Dict[str, Any]]]:
    results = asyncio.run(
        _run_ramp_probe_async(
            probe_texts=probe_texts,
            deployment=deployment,
            provider_name=provider_name,
            chunk_sizes=chunk_sizes,
        )
    )
    successful = [row for row in results if bool(row.get("ok"))]
    if not successful:
        raise RuntimeError(
            "Ramp probe failed for all chunk sizes; no valid chunk size selected."
        )
    best = max(successful, key=lambda row: float(row.get("rows_per_second") or 0.0))
    return int(best["chunk_size"]), results


def _validate_label_contract(
    label_id_to_intent: Dict[int, str],
    expected_label_count: int = BANK77_EXPECTED_LABEL_COUNT,
) -> List[int]:
    """Validate label cardinality and contiguous label IDs."""
    if len(label_id_to_intent) != expected_label_count:
        raise ValueError(
            f"BANK77 label cardinality changed: expected={expected_label_count}, "
            f"found={len(label_id_to_intent)}"
        )
    ordered_ids = sorted(int(k) for k in label_id_to_intent.keys())
    expected_ids = list(range(expected_label_count))
    if ordered_ids != expected_ids:
        raise ValueError(
            f"BANK77 label IDs are not contiguous {expected_ids[0]}..{expected_ids[-1]}: "
            f"found={ordered_ids}"
        )
    return ordered_ids


def _build_manifest_entries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        entries.append(
            {
                "position": int(position),
                "source_id": str(row["source_id"]),
                "split": str(row["split"]),
                "text": str(row["text"]),
                "label": int(row["label"]),
                "intent": str(row["intent"]),
            }
        )
    return entries


def _find_group_by_name(
    session,
    *,
    group_type: str,
    name: str,
) -> Optional[Group]:
    return (
        session.query(Group)
        .filter(
            Group.group_type == group_type,
            Group.name == name,
        )
        .first()
    )


def _find_artifacts_for_group(
    session,
    *,
    group_id: int,
    artifact_type: Optional[str] = None,
) -> List[CallArtifact]:
    query = session.query(CallArtifact)
    if artifact_type:
        query = query.filter(CallArtifact.artifact_type == artifact_type)
    artifacts = query.order_by(CallArtifact.id.asc()).all()
    out: List[CallArtifact] = []
    for artifact in artifacts:
        meta_group_id = _coerce_group_id(artifact.metadata_json)
        if meta_group_id == int(group_id):
            out.append(artifact)
    return out


def _find_single_artifact_for_group(
    session,
    *,
    group_id: int,
    artifact_type: str,
) -> Optional[CallArtifact]:
    artifacts = _find_artifacts_for_group(
        session,
        group_id=int(group_id),
        artifact_type=artifact_type,
    )
    if not artifacts:
        return None
    if len(artifacts) == 1:
        return artifacts[0]

    # Prefer latest artifact as effective head for idempotent reuse.
    return artifacts[-1]


def _load_embedding_matrix_artifact(
    artifact_service: ArtifactService,
    artifact: CallArtifact,
) -> np.ndarray:
    metadata = artifact.metadata_json or {}
    attempts = 6
    delay_seconds = 0.5
    for attempt in range(1, attempts + 1):
        try:
            matrix = artifact_service.load_artifact(
                str(artifact.uri),
                "embedding_matrix",
                expected_sha256=metadata.get("sha256"),
                expected_byte_size=_expected_byte_size(artifact),
            )
            return np.asarray(matrix, dtype=np.float64)
        except FileNotFoundError:
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds)
            delay_seconds *= 2.0
    raise RuntimeError("Unreachable: embedding matrix artifact retry loop exited unexpectedly")


def _load_json_artifact(
    artifact_service: ArtifactService,
    artifact: CallArtifact,
    *,
    artifact_type: str,
) -> Dict[str, Any]:
    metadata = artifact.metadata_json or {}
    loaded = artifact_service.load_artifact(
        str(artifact.uri),
        artifact_type,
        expected_sha256=metadata.get("sha256"),
        expected_byte_size=_expected_byte_size(artifact),
    )
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected JSON object for artifact id={artifact.id}; got {type(loaded).__name__}"
        )
    return loaded


def _ensure_storage_backend(
    artifact_service: ArtifactService,
    *,
    require_azure_blob: bool,
) -> None:
    backend = str(getattr(artifact_service.storage, "backend_type", "unknown"))
    if require_azure_blob and backend != "azure_blob":
        raise RuntimeError(
            "Storage backend is not azure_blob while --require-azure-blob is set. "
            f"Resolved backend='{backend}'."
        )


def _ensure_snapshot_group(
    session,
    *,
    snapshot_name: str,
    source_dataset: str,
    entries: List[Dict[str, Any]],
    split_counts: Dict[str, int],
    label_id_to_intent: Dict[int, str],
) -> Tuple[int, int, str]:
    """Create/reuse deterministic BANK77 dataset snapshot + manifest artifact."""
    repo = RawCallRepository(session)
    provenance = ProvenanceService(repo)
    artifacts = ArtifactService(repository=repo)

    manifest_hash = _manifest_hash(snapshot_name, entries)
    row_count = int(len(entries))
    ordered_label_ids = _validate_label_contract(label_id_to_intent)
    label_count = len(ordered_label_ids)

    existing_group = _find_group_by_name(
        session,
        group_type=GROUP_TYPE_DATASET_SNAPSHOT,
        name=snapshot_name,
    )
    if existing_group is not None:
        manifest_artifact = _find_single_artifact_for_group(
            session,
            group_id=int(existing_group.id),
            artifact_type="dataset_snapshot_manifest",
        )
        if manifest_artifact is None:
            raise ValueError(
                f"Snapshot '{snapshot_name}' exists (id={existing_group.id}) "
                "but has no dataset_snapshot_manifest artifact"
            )
        existing_hash = str((manifest_artifact.metadata_json or {}).get("manifest_hash") or "")
        if existing_hash != manifest_hash:
            raise ValueError(
                f"Snapshot '{snapshot_name}' exists (id={existing_group.id}) with different "
                f"manifest hash (expected={manifest_hash}, found={existing_hash})"
            )
        return int(existing_group.id), int(manifest_artifact.id), manifest_hash

    label_counts_by_id = Counter(int(entry["label"]) for entry in entries)
    metadata = {
        "has_ground_truth": True,
        "manifest_hash": manifest_hash,
        "dataset_source": source_dataset,
        "bootstrap_version": BANK77_BOOTSTRAP_VERSION,
        "row_count": row_count,
        "label_count": int(label_count),
        "split_counts": dict(split_counts),
        "label_ids": ordered_label_ids,
        "label_id_to_intent": {
            str(label_id): str(label_id_to_intent[label_id]) for label_id in ordered_label_ids
        },
        "label_counts_by_id": {
            str(label_id): int(label_counts_by_id.get(label_id, 0))
            for label_id in ordered_label_ids
        },
    }
    snapshot_group_id = provenance.create_dataset_snapshot_group(
        snapshot_name=snapshot_name,
        source_dataset=source_dataset,
        sample_size=row_count,
        label_mode=LABEL_MODE_LABELED,
        sampling_method=SAMPLING_METHOD,
        metadata=metadata,
    )
    manifest_artifact_id = artifacts.store_dataset_snapshot_manifest(
        snapshot_group_id=snapshot_group_id,
        snapshot_name=snapshot_name,
        entries=entries,
        metadata={
            "source_dataset": source_dataset,
            "label_mode": LABEL_MODE_LABELED,
            "split_counts": dict(split_counts),
            "label_count": int(label_count),
            "bootstrap_version": BANK77_BOOTSTRAP_VERSION,
        },
    )
    return int(snapshot_group_id), int(manifest_artifact_id), manifest_hash


def _ensure_group_link(
    repo: RawCallRepository,
    *,
    parent_group_id: int,
    child_group_id: int,
    link_type: str,
    relation: str,
) -> int:
    return int(
        repo.create_group_link(
            parent_group_id=int(parent_group_id),
            child_group_id=int(child_group_id),
            link_type=link_type,
            metadata_json={"relation": relation},
        )
    )


def _ensure_full_embedding_artifact(
    db: DatabaseConnectionV2,
    *,
    texts: List[str],
    snapshot_group_id: int,
    dataset_key: str,
    provider: str,
    embedding_engine: str,
    timeout_seconds: float,
    chunk_size: Optional[int],
) -> np.ndarray:
    matrix = asyncio.run(
        fetch_embeddings_async(
            texts_list=texts,
            deployment=embedding_engine,
            db=db,
            timeout=float(timeout_seconds),
            chunk_size=chunk_size,
            provider_name=provider,
            l3_cache_key=dataset_key,
            l3_entry_max=len(texts),
            l3_snapshot_group_id=int(snapshot_group_id),
        )
    )
    return np.asarray(matrix, dtype=np.float64)


def _resolve_full_embedding_artifact(
    session,
    *,
    dataset_key: str,
    provider: str,
    embedding_engine: str,
    row_count: int,
) -> Tuple[int, int, np.ndarray]:
    repo = RawCallRepository(session)
    artifacts = ArtifactService(repository=repo)
    hit = artifacts.find_embedding_matrix_artifact(
        dataset_key=dataset_key,
        embedding_engine=embedding_engine,
        provider=provider,
        entry_max=row_count,
        key_version=CACHE_KEY_VERSION,
    )
    if not hit:
        raise ValueError(
            "Full embedding_matrix artifact not found after embedding pass "
            f"(dataset_key={dataset_key})"
        )
    group_id = hit.get("group_id")
    if group_id is None:
        raise ValueError("Resolved full embedding artifact missing metadata.group_id")

    artifact = session.query(CallArtifact).filter(CallArtifact.id == int(hit["artifact_id"])).first()
    if artifact is None:
        raise ValueError(f"Resolved full artifact id={hit['artifact_id']} not found")

    matrix = _load_embedding_matrix_artifact(artifacts, artifact)
    return int(group_id), int(artifact.id), matrix


def _compute_intent_means(
    full_matrix: np.ndarray,
    rows: List[Dict[str, Any]],
    label_id_to_intent: Dict[int, str],
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[int, int]]:
    if int(full_matrix.shape[0]) != int(len(rows)):
        raise ValueError(
            "Embedding matrix row count mismatch: "
            f"matrix_rows={full_matrix.shape[0]} rows={len(rows)}"
        )
    ordered_label_ids = _validate_label_contract(label_id_to_intent)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)

    means: List[np.ndarray] = []
    mapping_rows: List[Dict[str, Any]] = []
    counts_by_label: Dict[int, int] = {}

    for row_index, label_id in enumerate(ordered_label_ids):
        mask = labels == int(label_id)
        indices = np.nonzero(mask)[0]
        if int(indices.size) == 0:
            raise ValueError(f"No rows found for label_id={label_id}")
        counts_by_label[int(label_id)] = int(indices.size)
        mean_vec = np.mean(full_matrix[indices], axis=0)
        means.append(np.asarray(mean_vec, dtype=np.float64))
        mapping_rows.append(
            {
                "row_index": int(row_index),
                "label_id": int(label_id),
                "intent": str(label_id_to_intent[label_id]),
                "count": int(indices.size),
            }
        )

    means_matrix = np.vstack(means).astype(np.float64, copy=False)
    return means_matrix, mapping_rows, counts_by_label


def _build_mapping_payload(
    *,
    source_dataset: str,
    snapshot_group_id: int,
    full_embedding_batch_group_id: int,
    means_embedding_batch_group_id: int,
    mapping_rows: List[Dict[str, Any]],
    counts_by_label: Dict[int, int],
) -> Dict[str, Any]:
    ordered_label_ids = [int(row["label_id"]) for row in mapping_rows]
    ordered_intents = [str(row["intent"]) for row in mapping_rows]
    return {
        "schema_version": "bank77.intent_mapping.v1",
        "mapping_type": MAPPING_TYPE,
        "source_dataset": source_dataset,
        "snapshot_group_id": int(snapshot_group_id),
        "full_embedding_batch_group_id": int(full_embedding_batch_group_id),
        "means_embedding_batch_group_id": int(means_embedding_batch_group_id),
        "row_count": int(len(mapping_rows)),
        "ordered_label_ids": ordered_label_ids,
        "ordered_intents": ordered_intents,
        "rows": mapping_rows,
        "counts_by_label_id": {str(k): int(v) for k, v in sorted(counts_by_label.items())},
    }


def _find_mapping_artifact_for_group(
    session,
    *,
    group_id: int,
) -> Optional[CallArtifact]:
    metrics_artifacts = _find_artifacts_for_group(
        session,
        group_id=int(group_id),
        artifact_type="metrics",
    )
    for artifact in metrics_artifacts:
        metadata = artifact.metadata_json or {}
        if str(metadata.get("step_name") or "") != MAPPING_STEP_NAME:
            continue
        if str(metadata.get("mapping_type") or "") != MAPPING_TYPE:
            continue
        return artifact
    return None


def _ensure_means_artifacts(
    session,
    *,
    source_dataset: str,
    snapshot_group_id: int,
    full_embedding_batch_group_id: int,
    full_matrix: np.ndarray,
    rows: List[Dict[str, Any]],
    label_id_to_intent: Dict[int, str],
    provider: str,
    embedding_engine: str,
    means_dataset_key: str,
) -> MeansArtifactResult:
    repo = RawCallRepository(session)
    provenance = ProvenanceService(repo)
    artifacts = ArtifactService(repository=repo)

    expected_label_count = len(_validate_label_contract(label_id_to_intent))
    means_hit = artifacts.find_embedding_matrix_artifact(
        dataset_key=means_dataset_key,
        embedding_engine=embedding_engine,
        provider=provider,
        entry_max=expected_label_count,
        key_version=CACHE_KEY_VERSION,
    )

    # Always prepare canonical mapping rows for deterministic validation/fallback creation.
    _, mapping_rows, counts_by_label = _compute_intent_means(
        full_matrix=full_matrix,
        rows=rows,
        label_id_to_intent=label_id_to_intent,
    )

    if means_hit:
        means_group_id_raw = means_hit.get("group_id")
        if means_group_id_raw is None:
            raise ValueError("Means embedding artifact missing metadata.group_id")
        means_group_id = int(means_group_id_raw)
        means_artifact_id = int(means_hit["artifact_id"])
        means_artifact = (
            session.query(CallArtifact).filter(CallArtifact.id == means_artifact_id).first()
        )
        if means_artifact is None:
            raise ValueError(f"Means artifact id={means_artifact_id} not found")

        means_matrix = _load_embedding_matrix_artifact(artifacts, means_artifact)
        if int(means_matrix.shape[0]) != int(expected_label_count):
            raise ValueError(
                f"Means artifact row mismatch: expected={expected_label_count}, "
                f"found={means_matrix.shape[0]}"
            )

        _ensure_group_link(
            repo,
            parent_group_id=means_group_id,
            child_group_id=int(snapshot_group_id),
            link_type="depends_on",
            relation="embedding_source_snapshot",
        )
        _ensure_group_link(
            repo,
            parent_group_id=means_group_id,
            child_group_id=int(full_embedding_batch_group_id),
            link_type="depends_on",
            relation="derived_from_full_embedding_matrix",
        )

        mapping_artifact = _find_mapping_artifact_for_group(session, group_id=means_group_id)
        if mapping_artifact is None:
            mapping_payload = _build_mapping_payload(
                source_dataset=source_dataset,
                snapshot_group_id=snapshot_group_id,
                full_embedding_batch_group_id=full_embedding_batch_group_id,
                means_embedding_batch_group_id=means_group_id,
                mapping_rows=mapping_rows,
                counts_by_label=counts_by_label,
            )
            mapping_artifact_id = artifacts.store_metrics(
                run_id=means_group_id,
                metrics=mapping_payload,
                step_name=MAPPING_STEP_NAME,
                metadata={
                    "mapping_type": MAPPING_TYPE,
                    "bootstrap_version": BANK77_BOOTSTRAP_VERSION,
                },
            )
        else:
            mapping_payload = _load_json_artifact(
                artifacts, mapping_artifact, artifact_type="metrics"
            )
            mapping_ids = [
                int(v) for v in list(mapping_payload.get("ordered_label_ids") or [])
            ]
            expected_ids = [int(row["label_id"]) for row in mapping_rows]
            if mapping_ids != expected_ids:
                raise ValueError(
                    "Existing mapping artifact label order differs from deterministic contract"
                )
            mapping_artifact_id = int(mapping_artifact.id)

        return MeansArtifactResult(
            means_batch_group_id=means_group_id,
            means_artifact_id=means_artifact_id,
            mapping_artifact_id=int(mapping_artifact_id),
            means_shape=(int(means_matrix.shape[0]), int(means_matrix.shape[1])),
        )

    means_matrix, mapping_rows, counts_by_label = _compute_intent_means(
        full_matrix=full_matrix,
        rows=rows,
        label_id_to_intent=label_id_to_intent,
    )
    means_batch_name = (
        f"bank77_means_{_slug(provider)}_{_slug(embedding_engine)}_{_slug(snapshot_group_id.__str__())}"
    )
    means_group_id = provenance.create_embedding_batch_group(
        name=means_batch_name,
        deployment=embedding_engine,
        metadata={
            "dataset_key": means_dataset_key,
            "provider": provider,
            "entry_max": int(means_matrix.shape[0]),
            "key_version": CACHE_KEY_VERSION,
            "representation": "intent_means",
            "source_snapshot_group_id": int(snapshot_group_id),
            "source_embedding_batch_group_id": int(full_embedding_batch_group_id),
            "bootstrap_version": BANK77_BOOTSTRAP_VERSION,
        },
    )
    means_artifact_id = artifacts.store_embedding_matrix(
        means_group_id,
        means_matrix,
        dataset_key=means_dataset_key,
        embedding_engine=embedding_engine,
        provider=provider,
        entry_max=int(means_matrix.shape[0]),
        key_version=CACHE_KEY_VERSION,
        metadata={
            "representation": "intent_means",
            "label_count": int(means_matrix.shape[0]),
            "source_snapshot_group_id": int(snapshot_group_id),
            "source_embedding_batch_group_id": int(full_embedding_batch_group_id),
            "bootstrap_version": BANK77_BOOTSTRAP_VERSION,
        },
    )

    mapping_payload = _build_mapping_payload(
        source_dataset=source_dataset,
        snapshot_group_id=snapshot_group_id,
        full_embedding_batch_group_id=full_embedding_batch_group_id,
        means_embedding_batch_group_id=means_group_id,
        mapping_rows=mapping_rows,
        counts_by_label=counts_by_label,
    )
    mapping_artifact_id = artifacts.store_metrics(
        run_id=means_group_id,
        metrics=mapping_payload,
        step_name=MAPPING_STEP_NAME,
        metadata={
            "mapping_type": MAPPING_TYPE,
            "bootstrap_version": BANK77_BOOTSTRAP_VERSION,
        },
    )

    _ensure_group_link(
        repo,
        parent_group_id=int(means_group_id),
        child_group_id=int(snapshot_group_id),
        link_type="depends_on",
        relation="embedding_source_snapshot",
    )
    _ensure_group_link(
        repo,
        parent_group_id=int(means_group_id),
        child_group_id=int(full_embedding_batch_group_id),
        link_type="depends_on",
        relation="derived_from_full_embedding_matrix",
    )

    return MeansArtifactResult(
        means_batch_group_id=int(means_group_id),
        means_artifact_id=int(means_artifact_id),
        mapping_artifact_id=int(mapping_artifact_id),
        means_shape=(int(means_matrix.shape[0]), int(means_matrix.shape[1])),
    )


def read_means_payload(
    db: DatabaseConnectionV2,
    *,
    means_dataset_key: str,
    provider: str,
    embedding_engine: str,
    expected_label_count: int,
) -> Tuple[np.ndarray, Dict[str, Any], int, int, int]:
    """Read means + mapping only (no full-matrix load) for immediate downstream use."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifacts = ArtifactService(repository=repo)

        means_hit = artifacts.find_embedding_matrix_artifact(
            dataset_key=means_dataset_key,
            embedding_engine=embedding_engine,
            provider=provider,
            entry_max=int(expected_label_count),
            key_version=CACHE_KEY_VERSION,
        )
        if not means_hit:
            raise ValueError(f"Means artifact not found for dataset_key={means_dataset_key}")

        means_group_id = means_hit.get("group_id")
        if means_group_id is None:
            raise ValueError("Means artifact missing metadata.group_id")

        means_artifact = (
            session.query(CallArtifact).filter(CallArtifact.id == int(means_hit["artifact_id"])).first()
        )
        if means_artifact is None:
            raise ValueError(f"Means artifact id={means_hit['artifact_id']} not found")
        means_matrix = _load_embedding_matrix_artifact(artifacts, means_artifact)

        mapping_artifact = _find_mapping_artifact_for_group(
            session,
            group_id=int(means_group_id),
        )
        if mapping_artifact is None:
            raise ValueError(
                f"Mapping artifact not found for means group_id={int(means_group_id)}"
            )
        mapping_payload = _load_json_artifact(
            artifacts,
            mapping_artifact,
            artifact_type="metrics",
        )
        return (
            means_matrix,
            mapping_payload,
            int(means_group_id),
            int(means_artifact.id),
            int(mapping_artifact.id),
        )


def _verify_lineage_link(
    session,
    *,
    parent_group_id: int,
    child_group_id: int,
    link_type: str,
) -> None:
    link = (
        session.query(GroupLink)
        .filter(
            GroupLink.parent_group_id == int(parent_group_id),
            GroupLink.child_group_id == int(child_group_id),
            GroupLink.link_type == link_type,
        )
        .first()
    )
    if link is None:
        raise ValueError(
            "Missing expected group link: "
            f"parent={parent_group_id}, child={child_group_id}, type={link_type}"
        )


def verify_bootstrap(
    db: DatabaseConnectionV2,
    *,
    snapshot_name: str,
    source_dataset: str,
    full_dataset_key: str,
    means_dataset_key: str,
    provider: str,
    embedding_engine: str,
    expected_row_count: int,
    expected_label_count: int,
) -> Dict[str, Any]:
    """Checkpoint verification across snapshot, full matrix, means, and linkage."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifacts = ArtifactService(repository=repo)

        snapshot_group = _find_group_by_name(
            session,
            group_type=GROUP_TYPE_DATASET_SNAPSHOT,
            name=snapshot_name,
        )
        if snapshot_group is None:
            raise ValueError(f"Snapshot not found: name='{snapshot_name}'")
        snapshot_group_id = int(snapshot_group.id)

        manifest_artifact = _find_single_artifact_for_group(
            session,
            group_id=snapshot_group_id,
            artifact_type="dataset_snapshot_manifest",
        )
        if manifest_artifact is None:
            raise ValueError(
                f"Snapshot group id={snapshot_group_id} missing dataset_snapshot_manifest artifact"
            )
        manifest_payload = _load_json_artifact(
            artifacts,
            manifest_artifact,
            artifact_type="dataset_snapshot_manifest",
        )
        entries = list(manifest_payload.get("entries") or [])
        if int(len(entries)) != int(expected_row_count):
            raise ValueError(
                f"Snapshot manifest count mismatch: expected={expected_row_count}, "
                f"found={len(entries)}"
            )
        labels = {int(entry["label"]) for entry in entries}
        intents = {str(entry.get("intent") or "") for entry in entries}
        if int(len(labels)) != int(expected_label_count):
            raise ValueError(
                f"Snapshot label cardinality mismatch: expected={expected_label_count}, "
                f"found={len(labels)}"
            )
        if int(len(intents)) != int(expected_label_count):
            raise ValueError(
                f"Snapshot intent cardinality mismatch: expected={expected_label_count}, "
                f"found={len(intents)}"
            )

        full_hit = artifacts.find_embedding_matrix_artifact(
            dataset_key=full_dataset_key,
            embedding_engine=embedding_engine,
            provider=provider,
            entry_max=int(expected_row_count),
            key_version=CACHE_KEY_VERSION,
        )
        if not full_hit:
            raise ValueError("Full embedding matrix artifact not found during verification")
        full_group_id_raw = full_hit.get("group_id")
        if full_group_id_raw is None:
            raise ValueError("Full embedding artifact missing metadata.group_id")
        full_group_id = int(full_group_id_raw)
        full_artifact = (
            session.query(CallArtifact).filter(CallArtifact.id == int(full_hit["artifact_id"])).first()
        )
        if full_artifact is None:
            raise ValueError(f"Full artifact id={full_hit['artifact_id']} not found")
        full_matrix = _load_embedding_matrix_artifact(artifacts, full_artifact)
        if int(full_matrix.shape[0]) != int(expected_row_count):
            raise ValueError(
                f"Full matrix row mismatch: expected={expected_row_count}, "
                f"found={full_matrix.shape[0]}"
            )

        means_matrix, mapping_payload, means_group_id, means_artifact_id, mapping_artifact_id = (
            read_means_payload(
                db,
                means_dataset_key=means_dataset_key,
                provider=provider,
                embedding_engine=embedding_engine,
                expected_label_count=expected_label_count,
            )
        )
        if int(means_matrix.shape[0]) != int(expected_label_count):
            raise ValueError(
                f"Means matrix row mismatch: expected={expected_label_count}, "
                f"found={means_matrix.shape[0]}"
            )
        if int(means_matrix.shape[1]) != int(full_matrix.shape[1]):
            raise ValueError(
                "Means/full dimension mismatch: "
                f"means_dim={means_matrix.shape[1]}, full_dim={full_matrix.shape[1]}"
            )

        ordered_ids = [int(v) for v in list(mapping_payload.get("ordered_label_ids") or [])]
        if len(ordered_ids) != int(expected_label_count):
            raise ValueError(
                f"Mapping ordered_label_ids mismatch: expected={expected_label_count}, "
                f"found={len(ordered_ids)}"
            )
        rows = list(mapping_payload.get("rows") or [])
        if len(rows) != int(expected_label_count):
            raise ValueError(
                f"Mapping rows mismatch: expected={expected_label_count}, found={len(rows)}"
            )
        for idx, row in enumerate(rows):
            if int(row.get("row_index", -1)) != int(idx):
                raise ValueError(f"Mapping row_index mismatch at idx={idx}")
            if int(row.get("label_id", -1)) != int(ordered_ids[idx]):
                raise ValueError(f"Mapping label_id mismatch at idx={idx}")

        _verify_lineage_link(
            session,
            parent_group_id=full_group_id,
            child_group_id=snapshot_group_id,
            link_type="depends_on",
        )
        _verify_lineage_link(
            session,
            parent_group_id=means_group_id,
            child_group_id=snapshot_group_id,
            link_type="depends_on",
        )
        _verify_lineage_link(
            session,
            parent_group_id=means_group_id,
            child_group_id=full_group_id,
            link_type="depends_on",
        )

        return {
            "snapshot_group_id": snapshot_group_id,
            "manifest_artifact_id": int(manifest_artifact.id),
            "full_embedding_batch_group_id": full_group_id,
            "full_embedding_artifact_id": int(full_artifact.id),
            "full_embedding_shape": [int(full_matrix.shape[0]), int(full_matrix.shape[1])],
            "means_embedding_batch_group_id": int(means_group_id),
            "means_embedding_artifact_id": int(means_artifact_id),
            "means_shape": [int(means_matrix.shape[0]), int(means_matrix.shape[1])],
            "mapping_artifact_id": int(mapping_artifact_id),
            "expected_row_count": int(expected_row_count),
            "expected_label_count": int(expected_label_count),
            "source_dataset": source_dataset,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/reuse BANK77 dataset snapshot + full/means embedding artifacts",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        default=BANK77_DATASET_SOURCE,
        help="HuggingFace dataset source (default: mteb/banking77)",
    )
    parser.add_argument(
        "--snapshot-name",
        type=str,
        default="bank77_mteb_full_labeled",
        help="Dataset snapshot group name",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="azure",
        help="Embedding provider name for ProviderFactory (default: azure)",
    )
    parser.add_argument(
        "--embedding-engine",
        type=str,
        default="text-embedding-3-large",
        help="Embedding deployment/model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional async batch chunk size for embedding calls",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="Timeout for full embedding fetch (default: 1800s)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional DATABASE_URL override (default: env DATABASE_URL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and print derived contract values without writing DB/artifacts",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Create/reuse dataset snapshot + manifest only (skip embeddings)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run readback + lineage verification only (requires existing artifacts)",
    )
    parser.add_argument(
        "--require-azure-blob",
        action="store_true",
        help="Fail if storage backend is not azure_blob",
    )
    parser.add_argument(
        "--ramp-probe-chunk-sizes",
        type=str,
        default="16,32,64,128",
        help="Comma-separated chunk sizes for embedding ramp probe",
    )
    parser.add_argument(
        "--ramp-probe-rows",
        type=int,
        default=128,
        help="Number of BANK77 rows used for ramp probe",
    )
    parser.add_argument(
        "--ramp-only",
        action="store_true",
        help="Run chunk-size ramp probe only; do not write snapshot/artifacts",
    )
    parser.add_argument(
        "--auto-ramp",
        action="store_true",
        help="Run ramp probe and use best chunk size for full execution",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_only and args.dry_run:
        raise ValueError("--verify-only cannot be combined with --dry-run")
    if args.verify_only and args.snapshot_only:
        raise ValueError("--verify-only cannot be combined with --snapshot-only")
    if args.ramp_only and (args.verify_only or args.snapshot_only):
        raise ValueError("--ramp-only cannot be combined with --verify-only/--snapshot-only")

    db_url = args.database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "DATABASE_URL is required (env DATABASE_URL or --database-url flag)"
        )

    rows, split_counts, label_id_to_intent = _load_bank77_rows(args.dataset_source)
    ordered_label_ids = _validate_label_contract(label_id_to_intent)
    manifest_entries = _build_manifest_entries(rows)
    manifest_hash = _manifest_hash(args.snapshot_name, manifest_entries)
    full_dataset_key = (
        f"bank77:{args.dataset_source}:manifest={manifest_hash}:full_embeddings"
    )
    means_dataset_key = f"bank77:{args.dataset_source}:manifest={manifest_hash}:intent_means"

    row_count = int(len(rows))
    label_count = int(len(ordered_label_ids))

    if args.ramp_only or args.auto_ramp:
        chunk_sizes = _parse_chunk_sizes(args.ramp_probe_chunk_sizes)
        probe_rows = int(max(1, min(int(args.ramp_probe_rows), row_count)))
        probe_texts = [str(row["text"]) for row in rows[:probe_rows]]
        selected_chunk_size, probe_results = _run_ramp_probe(
            probe_texts=probe_texts,
            deployment=args.embedding_engine,
            provider_name=args.provider,
            chunk_sizes=chunk_sizes,
        )
        print("[RAMP] Probe results")
        print(json.dumps(probe_results, indent=2, sort_keys=True))
        print(f"[RAMP] Selected chunk_size={selected_chunk_size}")
        if args.ramp_only:
            return
        if args.auto_ramp:
            args.chunk_size = int(selected_chunk_size)

    if args.dry_run:
        print("[DRY-RUN] BANK77 bootstrap contract")
        print(f"  dataset_source={args.dataset_source}")
        print(f"  snapshot_name={args.snapshot_name}")
        print(f"  row_count={row_count}")
        print(f"  label_count={label_count}")
        print(f"  split_counts={json.dumps(split_counts, sort_keys=True)}")
        print(f"  manifest_hash={manifest_hash}")
        print(f"  full_dataset_key={full_dataset_key}")
        print(f"  means_dataset_key={means_dataset_key}")
        return

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifacts = ArtifactService(repository=repo)
        _ensure_storage_backend(
            artifacts,
            require_azure_blob=bool(args.require_azure_blob),
        )
        snapshot_group_id, manifest_artifact_id, _ = _ensure_snapshot_group(
            session,
            snapshot_name=args.snapshot_name,
            source_dataset=args.dataset_source,
            entries=manifest_entries,
            split_counts=split_counts,
            label_id_to_intent=label_id_to_intent,
        )

    if args.verify_only:
        summary = verify_bootstrap(
            db,
            snapshot_name=args.snapshot_name,
            source_dataset=args.dataset_source,
            full_dataset_key=full_dataset_key,
            means_dataset_key=means_dataset_key,
            provider=args.provider,
            embedding_engine=args.embedding_engine,
            expected_row_count=row_count,
            expected_label_count=label_count,
        )
        print("[OK] Verification passed")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.snapshot_only:
        print("[OK] Snapshot contract ready")
        print(f"  snapshot_group_id={snapshot_group_id}")
        print(f"  manifest_artifact_id={manifest_artifact_id}")
        print(f"  row_count={row_count}")
        print(f"  label_count={label_count}")
        return

    texts = [str(row["text"]) for row in rows]
    _ensure_full_embedding_artifact(
        db,
        texts=texts,
        snapshot_group_id=snapshot_group_id,
        dataset_key=full_dataset_key,
        provider=args.provider,
        embedding_engine=args.embedding_engine,
        timeout_seconds=float(args.timeout_seconds),
        chunk_size=args.chunk_size,
    )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        full_batch_group_id, full_artifact_id, full_matrix = _resolve_full_embedding_artifact(
            session,
            dataset_key=full_dataset_key,
            provider=args.provider,
            embedding_engine=args.embedding_engine,
            row_count=row_count,
        )
        _ensure_group_link(
            repo,
            parent_group_id=full_batch_group_id,
            child_group_id=snapshot_group_id,
            link_type="depends_on",
            relation="embedding_source_snapshot",
        )

        means_result = _ensure_means_artifacts(
            session,
            source_dataset=args.dataset_source,
            snapshot_group_id=snapshot_group_id,
            full_embedding_batch_group_id=full_batch_group_id,
            full_matrix=full_matrix,
            rows=rows,
            label_id_to_intent=label_id_to_intent,
            provider=args.provider,
            embedding_engine=args.embedding_engine,
            means_dataset_key=means_dataset_key,
        )

    verification_summary = verify_bootstrap(
        db,
        snapshot_name=args.snapshot_name,
        source_dataset=args.dataset_source,
        full_dataset_key=full_dataset_key,
        means_dataset_key=means_dataset_key,
        provider=args.provider,
        embedding_engine=args.embedding_engine,
        expected_row_count=row_count,
        expected_label_count=label_count,
    )

    print("[OK] BANK77 bootstrap complete")
    print(f"  snapshot_group_id={snapshot_group_id}")
    print(f"  manifest_artifact_id={manifest_artifact_id}")
    print(f"  full_embedding_batch_group_id={full_batch_group_id}")
    print(f"  full_embedding_artifact_id={full_artifact_id}")
    print(f"  means_embedding_batch_group_id={means_result.means_batch_group_id}")
    print(f"  means_embedding_artifact_id={means_result.means_artifact_id}")
    print(f"  mapping_artifact_id={means_result.mapping_artifact_id}")
    print(f"  full_dataset_key={full_dataset_key}")
    print(f"  means_dataset_key={means_dataset_key}")
    print(f"  means_shape={list(means_result.means_shape)}")
    print("[OK] Verification summary")
    print(json.dumps(verification_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
