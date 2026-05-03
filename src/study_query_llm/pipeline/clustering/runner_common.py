"""Shared utilities for canonical clustering runners."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from study_query_llm.algorithms.dimensionality_reduction import pca_svd_project

from .grammar import parse_method_name


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
        if stage in {
            "kmeans",
            "gmm",
            "hdbscan",
            "dbscan",
            "agglomerative",
            "spherical-kmeans",
        }:
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


def build_parameters_schema(
    *,
    fixed_keys: dict[str, dict[str, Any]],
    chain: tuple[str, ...],
    fixed_required: list[str],
) -> dict[str, Any]:
    """Build a JSON-schema dict for bundled clustering parameters."""
    properties = dict(fixed_keys)
    required = list(fixed_required)
    if "pca" in chain:
        properties["pca_n_components"] = {"type": "integer", "minimum": 1}
        required.append("pca_n_components")
    return {"type": "object", "properties": properties, "required": required}


def assert_no_chain_conflicts(
    *,
    method_name: str,
    parameters: Mapping[str, Any],
    chain: tuple[str, ...],
) -> None:
    """Reject user parameters that conflict with the grammar-encoded chain."""
    params = dict(parameters or {})
    if "pca" not in chain and params.get("pca_n_components") is not None:
        raise ValueError(
            f"{method_name} does not include a PCA chain stage; "
            "remove pca_n_components from parameters"
        )
    # Legacy booleans that imply out-of-band preprocessing
    if "normalize" not in chain:
        for key in (
            "normalize_embeddings",
            "kmeans_normalize_embeddings",
            "gmm_normalize_embeddings",
            "hdbscan_normalize_embeddings",
            "dbscan_normalize_embeddings",
        ):
            raw = params.get(key)
            if raw is True:
                raise ValueError(
                    f"{method_name} does not include normalize in its preprocessing chain; "
                    f"do not set {key}=True"
                )


def synthesize_fixed_bundled_payload(
    *,
    method_name: str,
    parameters: Mapping[str, Any],
    embedding_dim: int,
    n_samples: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build ``_v1_pipeline_{resolved,effective}`` for fixed bundled methods from grammar."""
    base_algorithm, chain, _fit_tok = parse_method_name(method_name)
    emb_dim = int(embedding_dim)
    n_s = int(n_samples)
    max_pca = max(1, min(emb_dim, max(1, n_s - 1)))

    if "pca" in chain:
        raw = parameters.get("pca_n_components")
        if raw is None:
            raise ValueError(
                f"{method_name} requires integer parameter pca_n_components "
                "(PCA chain stage present)"
            )
        nc = int(raw)
        if nc < 1 or nc > max_pca:
            raise ValueError(
                f"pca_n_components={nc} exceeds max(1, min(embedding_dim={emb_dim}, "
                f"n_samples-1={max(1, n_s - 1)}))={max_pca}; "
                f"choose a value in [1, {max_pca}]"
            )

    pipeline_resolved: list[dict[str, Any]] = [{"stage": "embed", "params": {}}]
    for stage in chain:
        if stage == "normalize":
            pipeline_resolved.append({"stage": "normalize", "params": {}})
        elif stage == "pca":
            nc = int(parameters["pca_n_components"])
            pipeline_resolved.append(
                {
                    "stage": "pca",
                    "params": {"random_state": 42, "n_components": int(nc)},
                }
            )

    pipeline_resolved.append({"stage": base_algorithm, "params": {}})
    pipeline_effective = ["embed", *list(chain), base_algorithm]
    return pipeline_resolved, pipeline_effective


def resolve_pipeline_or_synthesize(
    *,
    method_name: str,
    parameters: Mapping[str, Any],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return resolved/effective pipelines, synthesizing from grammar when absent."""
    resolved_in = list(parameters.get("_v1_pipeline_resolved") or [])
    effective_in = list(parameters.get("_v1_pipeline_effective") or [])
    base_algorithm, chain, _fit_tok = parse_method_name(method_name)
    expected_effective = ["embed", *list(chain), base_algorithm]

    matrix = np.asarray(embeddings, dtype=np.float64)
    n_samples = int(matrix.shape[0])
    embedding_dim = int(matrix.shape[1])

    if resolved_in and effective_in:
        actual = [str(s) for s in effective_in]
        if actual != expected_effective:
            raise ValueError(
                f"{method_name}: pipeline_effective mismatch: expected {expected_effective}, "
                f"got {actual}"
            )
        return resolved_in, actual

    synth_resolved, synth_eff = synthesize_fixed_bundled_payload(
        method_name=method_name,
        parameters=parameters,
        embedding_dim=embedding_dim,
        n_samples=n_samples,
    )
    if [str(x) for x in synth_eff] != expected_effective:
        raise ValueError(
            f"{method_name}: synthesized pipeline_effective mismatch: "
            f"expected {expected_effective}, got {synth_eff}"
        )
    return synth_resolved, [str(x) for x in synth_eff]


# --- JSON-schema bundles for ``AlgorithmSpec.parameters_schema`` (Wave 1) ---

KMEANS_FAMILY_FIXED_KEYS: dict[str, dict[str, Any]] = {
    "k": {"type": "integer", "minimum": 2},
    "random_state": {"type": "integer"},
    "n_init": {"type": "integer", "minimum": 1},
    "init": {"type": "string", "enum": ["k-means++", "random"]},
}

GMM_FAMILY_FIXED_KEYS: dict[str, dict[str, Any]] = {
    "k": {"type": "integer", "minimum": 2},
    "random_state": {"type": "integer"},
    "n_init": {"type": "integer", "minimum": 1},
    "covariance_type": {"type": "string"},
}

DBSCAN_FAMILY_FIXED_KEYS: dict[str, dict[str, Any]] = {
    "eps": {"type": "number", "exclusiveMinimum": 0.0},
    "min_samples": {"type": "integer", "minimum": 1},
    "metric": {"type": "string"},
}

AGGLOMERATIVE_FAMILY_FIXED_KEYS: dict[str, dict[str, Any]] = {
    "k": {"type": "integer", "minimum": 2},
    "linkage": {"type": "string", "enum": ["ward", "complete", "average", "single"]},
    "metric": {"type": "string"},
}

HDBSCAN_FAMILY_FIXED_KEYS: dict[str, dict[str, Any]] = {
    "min_cluster_size": {"type": "integer", "minimum": 2},
    "min_samples": {"type": "integer", "minimum": 1},
    "metric": {"type": "string"},
    "cluster_selection_method": {"type": "string", "enum": ["eom", "leaf"]},
    "cluster_selection_epsilon": {"type": "number", "minimum": 0.0},
    "alpha": {"type": "number", "exclusiveMinimum": 0.0},
    "allow_single_cluster": {"type": "boolean"},
    "random_state": {"type": "integer"},
    "core_dist_n_jobs": {"type": "integer"},
    "approx_min_span_tree": {"type": "boolean"},
}

SWEEP_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "k_range": {"type": "array", "items": {"type": "integer"}},
        "selection_metric": {"type": "string"},
        "selection_rule": {"type": "string"},
        "pca_n_components": {"type": "integer"},
        "kmeans_distance_metric": {"type": "string"},
        "gmm_covariance_type": {"type": "string"},
    },
}

HDBSCAN_BASELINE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hdbscan_min_cluster_size": {"type": "integer"},
        "min_cluster_size": {"type": "integer"},
        "hdbscan_min_samples": {"type": "integer"},
        "min_samples": {"type": "integer"},
        "hdbscan_metric": {"type": "string"},
        "metric": {"type": "string"},
        "hdbscan_cluster_selection_method": {"type": "string"},
        "cluster_selection_method": {"type": "string"},
        "hdbscan_cluster_selection_epsilon": {"type": "number"},
        "cluster_selection_epsilon": {"type": "number"},
        "hdbscan_alpha": {"type": "number"},
        "alpha": {"type": "number"},
        "hdbscan_allow_single_cluster": {"type": "boolean"},
        "allow_single_cluster": {"type": "boolean"},
        "hdbscan_normalize_embeddings": {"type": "boolean"},
        "normalize_embeddings": {"type": "boolean"},
        "hdbscan_random_state": {"type": "integer"},
        "random_state": {"type": "integer"},
        "hdbscan_core_dist_n_jobs": {"type": "integer"},
        "core_dist_n_jobs": {"type": "integer"},
        "hdbscan_approx_min_span_tree": {"type": "boolean"},
        "approx_min_span_tree": {"type": "boolean"},
    },
}
