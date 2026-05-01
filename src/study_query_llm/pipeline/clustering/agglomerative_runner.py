"""Canonical runner: agglomerative clustering with fixed k."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from study_query_llm.pipeline.runner import allow_no_run_stage

from .runner_common import cluster_size_summary, to_json_bytes


def _resolve_k(parameters: Mapping[str, Any], *, n_samples: int) -> int:
    raw = parameters.get("k")
    if raw is None:
        raw = parameters.get("agglomerative_k")
    if raw is None:
        raise ValueError("agglomerative+fixed-k requires integer parameter k")
    k = int(raw)
    if k < 2:
        raise ValueError(f"agglomerative+fixed-k requires k >= 2, got {k}")
    if k >= n_samples:
        raise ValueError(
            "agglomerative+fixed-k requires k < n_samples "
            f"(k={k}, n_samples={n_samples})"
        )
    return k


@allow_no_run_stage
def run_agglomerative_fixed_k_analysis(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: Mapping[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Run deterministic agglomerative clustering with a fixed number of clusters."""
    if embeddings is None:
        raise ValueError("agglomerative+fixed-k requires embedding rows")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("agglomerative+fixed-k requires non-empty 2D embeddings")

    n_samples = int(matrix.shape[0])
    n_features = int(matrix.shape[1])
    k = _resolve_k(parameters, n_samples=n_samples)

    linkage = str(
        parameters.get("agglomerative_linkage")
        or parameters.get("linkage")
        or "ward"
    ).strip().lower()
    metric = str(
        parameters.get("agglomerative_metric")
        or parameters.get("metric")
        or "euclidean"
    ).strip().lower()
    if linkage == "ward":
        metric = "euclidean"

    try:
        model = AgglomerativeClustering(
            n_clusters=int(k),
            linkage=linkage,
            metric=metric,
        )
    except TypeError:
        # Backward compatibility for older sklearn releases.
        model = AgglomerativeClustering(
            n_clusters=int(k),
            linkage=linkage,
            affinity=metric,  # type: ignore[call-arg]
        )

    labels = np.asarray(model.fit_predict(matrix), dtype=np.int64)
    cluster_ids, cluster_sizes, noise_count, noise_fraction = cluster_size_summary(labels)

    summary = {
        "method_name": str(method_name),
        "base_algorithm": "agglomerative",
        "input_group_id": int(input_group_id),
        "input_group_type": str(input_group_type),
        "input_group_metadata": dict(input_group_metadata or {}),
        "text_count": int(len(texts)),
        "n_samples": n_samples,
        "n_features": n_features,
        "cluster_count": int(len(cluster_ids)),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "noise_count": int(noise_count),
        "noise_fraction": float(noise_fraction),
        "parameters": {
            "k": int(k),
            "linkage": linkage,
            "metric": metric,
        },
    }
    labels_payload = {
        "cluster_labels": labels.tolist(),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "parameters": dict(summary["parameters"]),
    }

    return {
        "scalar_results": {
            "n_samples": float(n_samples),
            "n_features": float(n_features),
            "cluster_count": float(len(cluster_ids)),
            "k": float(k),
        },
        "structured_results": {
            "clustering_summary": summary,
            "clustering_labels": labels_payload,
        },
        "artifacts": {
            "agglomerative_summary.json": to_json_bytes(summary),
            "agglomerative_labels.json": to_json_bytes(labels_payload),
        },
        "result_ref": "agglomerative_summary.json",
    }
