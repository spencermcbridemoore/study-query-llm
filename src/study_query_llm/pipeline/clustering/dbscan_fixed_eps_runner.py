"""Grammar-bound DBSCAN fixed-eps clustering."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.cluster import DBSCAN

from study_query_llm.pipeline.runner import allow_no_run_stage

from .grammar import parse_method_name
from .runner_common import (
    assert_no_chain_conflicts,
    cluster_size_summary,
    preprocess_for_effective_pipeline,
    resolve_pipeline_or_synthesize,
    to_json_bytes,
)


def _resolve_eps_min_samples(parameters: Mapping[str, Any]) -> tuple[float, int, str]:
    raw_eps = parameters.get("eps")
    if raw_eps is None:
        raise ValueError("dbscan fixed-eps requires parameter eps")
    eps = float(raw_eps)
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    raw_ms = parameters.get("min_samples")
    if raw_ms is None:
        raise ValueError("dbscan fixed-eps requires integer parameter min_samples")
    min_samples = int(raw_ms)
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")

    metric = str(parameters.get("metric", "euclidean")).strip().lower()
    return eps, min_samples, metric


@allow_no_run_stage
def run_dbscan_fixed_eps_analysis(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: Mapping[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Run sklearn DBSCAN with fixed ``eps``; grammar-bound preprocessing chain."""
    if embeddings is None:
        raise ValueError(f"{method_name} requires embedding rows")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{method_name} requires non-empty 2D embeddings")

    _, chain, _fit_tok = parse_method_name(method_name)
    assert_no_chain_conflicts(method_name=method_name, parameters=parameters, chain=chain)

    eps, min_samples, metric = _resolve_eps_min_samples(parameters)

    pipeline_resolved, pipeline_effective = resolve_pipeline_or_synthesize(
        method_name=method_name,
        parameters=parameters,
        embeddings=matrix,
    )
    processed, preprocess_meta = preprocess_for_effective_pipeline(
        embeddings=matrix,
        pipeline_resolved=pipeline_resolved,
        pipeline_effective=pipeline_effective,
    )

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = np.asarray(model.fit_predict(processed), dtype=np.int64)
    cluster_ids, cluster_sizes, noise_count, noise_fraction = cluster_size_summary(labels)

    summary = {
        "method_name": str(method_name),
        "base_algorithm": "dbscan",
        "input_group_id": int(input_group_id),
        "input_group_type": str(input_group_type),
        "input_group_metadata": dict(input_group_metadata or {}),
        "text_count": int(len(texts)),
        "n_samples": int(processed.shape[0]),
        "n_features": int(processed.shape[1]),
        "cluster_count": int(len(cluster_ids)),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "noise_count": int(noise_count),
        "noise_fraction": float(noise_fraction),
        "parameters": {
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": metric,
            "preprocess_applied": list(preprocess_meta.get("preprocess_applied") or []),
        },
        "preprocess": preprocess_meta,
    }
    labels_payload = {
        "cluster_labels": labels.tolist(),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "parameters": dict(summary["parameters"]),
    }

    return {
        "scalar_results": {
            "n_samples": float(processed.shape[0]),
            "n_features": float(processed.shape[1]),
            "cluster_count": float(len(cluster_ids)),
            "noise_count": float(noise_count),
            "noise_fraction": float(noise_fraction),
        },
        "structured_results": {
            "clustering_summary": summary,
            "clustering_labels": labels_payload,
        },
        "artifacts": {
            "dbscan_summary.json": to_json_bytes(summary),
            "dbscan_labels.json": to_json_bytes(labels_payload),
        },
        "result_ref": "dbscan_summary.json",
    }
