"""Grammar-bound fixed-k GaussianMixture runners."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.mixture import GaussianMixture

from study_query_llm.pipeline.runner import allow_no_run_stage

from .grammar import parse_method_name
from .runner_common import (
    assert_no_chain_conflicts,
    cluster_size_summary,
    preprocess_for_effective_pipeline,
    resolve_pipeline_or_synthesize,
    to_json_bytes,
)


def _resolve_k(parameters: Mapping[str, Any], *, n_samples: int) -> int:
    raw = parameters.get("k")
    if raw is None:
        raise ValueError("fixed-k GMM requires integer parameter k")
    k = int(raw)
    if k < 2:
        raise ValueError(f"fixed-k GMM requires k >= 2, got {k}")
    if k >= n_samples:
        raise ValueError(
            f"fixed-k GMM requires k < n_samples (k={k}, n_samples={n_samples})"
        )
    return k


@allow_no_run_stage
def run_gmm_fixed_k_analysis(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: Mapping[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit GaussianMixture with fixed ``k``; grammar-bound preprocessing chain."""
    if embeddings is None:
        raise ValueError(f"{method_name} requires embedding rows")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{method_name} requires non-empty 2D embeddings")

    _, chain, _fit_tok = parse_method_name(method_name)
    assert_no_chain_conflicts(method_name=method_name, parameters=parameters, chain=chain)

    n_samples = int(matrix.shape[0])
    k = _resolve_k(parameters, n_samples=n_samples)

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

    random_state = int(parameters.get("random_state", 42))
    n_init = int(parameters.get("n_init", 1))
    covariance_type = str(parameters.get("covariance_type", "full"))

    model = GaussianMixture(
        n_components=int(k),
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
    )
    labels = np.asarray(model.fit_predict(processed), dtype=np.int64)
    cluster_ids, cluster_sizes, noise_count, noise_fraction = cluster_size_summary(labels)

    summary = {
        "method_name": str(method_name),
        "base_algorithm": "gmm",
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
            "k": int(k),
            "random_state": random_state,
            "n_init": n_init,
            "covariance_type": covariance_type,
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
            "k": float(k),
        },
        "structured_results": {
            "clustering_summary": summary,
            "clustering_labels": labels_payload,
        },
        "artifacts": {
            "gmm_summary.json": to_json_bytes(summary),
            "gmm_labels.json": to_json_bytes(labels_payload),
        },
        "result_ref": "gmm_summary.json",
    }
