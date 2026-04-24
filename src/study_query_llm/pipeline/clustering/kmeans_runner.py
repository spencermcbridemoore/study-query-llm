"""Canonical runner: kmeans + silhouette + kneedle."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from study_query_llm.pipeline.runner import allow_no_run_stage

from .runner_common import (
    cluster_size_summary,
    preprocess_for_effective_pipeline,
    to_json_bytes,
)
from .selection import build_selection_evidence, kneedle_choice


def _resolved_pipeline_from_params(
    parameters: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = list(parameters.get("_v1_pipeline_resolved") or [])
    effective = list(parameters.get("_v1_pipeline_effective") or [])
    if resolved and effective:
        return resolved, [str(stage) for stage in effective]
    return (
        [{"stage": "embed", "params": {}}, {"stage": "kmeans", "params": {}}],
        ["embed", "kmeans"],
    )


def _terminal_params(pipeline_resolved: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in pipeline_resolved:
        if str(entry.get("stage") or "") == "kmeans":
            return dict(entry.get("params") or {})
    return {}


@allow_no_run_stage
def run_kmeans_silhouette_kneedle_analysis(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: Mapping[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Run deterministic KMeans sweep with silhouette + kneedle selection."""
    if embeddings is None:
        raise ValueError("kmeans+silhouette+kneedle requires embedding rows")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("kmeans+silhouette+kneedle requires non-empty 2D embeddings")

    pipeline_resolved, pipeline_effective = _resolved_pipeline_from_params(parameters)
    processed, preprocess_meta = preprocess_for_effective_pipeline(
        embeddings=matrix,
        pipeline_resolved=pipeline_resolved,
        pipeline_effective=pipeline_effective,
    )
    terminal = _terminal_params(pipeline_resolved)

    raw_range = list(terminal.get("k_range") or parameters.get("k_range") or [2, 3, 5, 8, 10, 15, 20, 30, 50])
    k_range = sorted(set(int(v) for v in raw_range if int(v) >= 2 and int(v) < int(processed.shape[0])))
    if not k_range:
        raise ValueError("kmeans sweep requires at least one valid k in [2, n_samples)")

    random_state = int(terminal.get("random_state", parameters.get("random_state", 42)))
    n_init = int(terminal.get("n_init", parameters.get("kmeans_n_init", 20)))
    max_iter = int(terminal.get("max_iter", parameters.get("kmeans_max_iter", 300)))
    distance_metric = str(terminal.get("distance_metric", "cosine")).strip().lower()
    sklearn_metric = "cosine" if distance_metric == "cosine" else "euclidean"

    curve: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    inertia_by_k: dict[int, float] = {}
    for k in k_range:
        model = KMeans(
            n_clusters=int(k),
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            algorithm="lloyd",
        )
        labels = np.asarray(model.fit_predict(processed), dtype=np.int64)
        labels_by_k[int(k)] = labels
        inertia_by_k[int(k)] = float(model.inertia_)
        unique_count = int(np.unique(labels).shape[0])
        if unique_count <= 1:
            silhouette = -1.0
        else:
            silhouette = float(silhouette_score(processed, labels, metric=sklearn_metric))
        curve.append(
            {
                "k": int(k),
                "silhouette": float(silhouette),
                "inertia": float(model.inertia_),
                "cluster_count": unique_count,
            }
        )

    silhouettes = [float(item["silhouette"]) for item in curve]
    chosen_k = kneedle_choice(k_range, silhouettes)
    chosen_labels = labels_by_k[int(chosen_k)]
    chosen_inertia = float(inertia_by_k[int(chosen_k)])
    cluster_ids, cluster_sizes, noise_count, noise_fraction = cluster_size_summary(chosen_labels)

    selection_curve_name = "kmeans_selection_curve.json"
    selection_evidence = build_selection_evidence(
        sweep_range=k_range,
        selection_metric="silhouette",
        selection_rule="kneedle",
        chosen_value=int(chosen_k),
        selection_curve_artifact_ref=selection_curve_name,
        chosen_label="chosen_k",
        selection_rule_params={"curve": "concave", "direction": "increasing"},
        rationale=f"kneedle chose k={chosen_k} on silhouette curve",
    )
    summary = {
        "method_name": str(method_name),
        "base_algorithm": "kmeans",
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
            "random_state": random_state,
            "n_init": n_init,
            "max_iter": max_iter,
            "distance_metric": distance_metric,
            "k_range": k_range,
            "chosen_k": int(chosen_k),
            "selection_metric": "silhouette",
            "selection_rule": "kneedle",
        },
        "selection_evidence": selection_evidence,
        "preprocess": preprocess_meta,
        "objective": {"inertia": chosen_inertia},
    }
    labels_payload = {
        "cluster_labels": chosen_labels.tolist(),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "parameters": summary["parameters"],
    }
    selection_curve_payload = {
        "metric": "silhouette",
        "selection_rule": "kneedle",
        "points": curve,
        "chosen_k": int(chosen_k),
    }

    return {
        "scalar_results": {
            "n_samples": float(processed.shape[0]),
            "n_features": float(processed.shape[1]),
            "cluster_count": float(len(cluster_ids)),
            "chosen_k": float(chosen_k),
            "objective_inertia": float(chosen_inertia),
        },
        "structured_results": {
            "clustering_summary": summary,
            "clustering_labels": labels_payload,
            "selection_evidence": selection_evidence,
        },
        "artifacts": {
            "kmeans_summary.json": to_json_bytes(summary),
            "kmeans_labels.json": to_json_bytes(labels_payload),
            selection_curve_name: to_json_bytes(selection_curve_payload),
        },
        "result_ref": "kmeans_summary.json",
    }
