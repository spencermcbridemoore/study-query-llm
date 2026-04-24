"""Shared utilities for canonical clustering runners."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from study_query_llm.algorithms.dimensionality_reduction import pca_svd_project


def to_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize payload to UTF-8 JSON bytes."""
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize matrix rows."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def preprocess_for_effective_pipeline(
    *,
    embeddings: np.ndarray,
    pipeline_resolved: list[dict[str, Any]],
    pipeline_effective: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply preprocessing stages implied by effective pipeline."""
    resolved_map = {
        str(entry.get("stage") or ""): dict(entry.get("params") or {})
        for entry in pipeline_resolved
    }
    matrix = np.asarray(embeddings, dtype=np.float64)
    meta: dict[str, Any] = {"preprocess_applied": []}
    for stage in pipeline_effective:
        if stage == "normalize":
            matrix = normalize_rows(matrix)
            meta["preprocess_applied"].append("normalize")
            continue
        if stage == "pca":
            params = resolved_map.get("pca") or {}
            n_components = params.get("n_components")
            if n_components is not None:
                matrix, pca_meta = pca_svd_project(matrix, int(n_components))
                meta["preprocess_applied"].append("pca")
                meta["pca"] = {
                    "n_components": int(pca_meta.get("pca_dim_used") or int(n_components)),
                }
            continue
        if stage in {"kmeans", "gmm", "hdbscan"}:
            break
    return matrix, meta


def cluster_size_summary(labels: np.ndarray) -> tuple[list[int], dict[str, int], int, float]:
    """Return cluster-id list, cluster size map, noise count, and noise fraction."""
    labels_int = np.asarray(labels, dtype=np.int64)
    n_samples = int(labels_int.shape[0])
    cluster_ids = sorted(int(v) for v in np.unique(labels_int) if int(v) >= 0)
    cluster_sizes = {
        str(cluster_id): int(np.sum(labels_int == cluster_id))
        for cluster_id in cluster_ids
    }
    noise_count = int(np.sum(labels_int < 0))
    noise_fraction = float(noise_count / n_samples) if n_samples > 0 else 0.0
    return cluster_ids, cluster_sizes, noise_count, noise_fraction
