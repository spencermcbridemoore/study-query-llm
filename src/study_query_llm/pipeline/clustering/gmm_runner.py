"""Canonical runner: gmm + bic + argmin."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.mixture import GaussianMixture

from study_query_llm.pipeline.runner import allow_no_run_stage

from .runner_common import (
    cluster_size_summary,
    preprocess_for_effective_pipeline,
    to_json_bytes,
)
from .selection import argmin_choice, build_selection_evidence


def _resolved_pipeline_from_params(
    parameters: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = list(parameters.get("_v1_pipeline_resolved") or [])
    effective = list(parameters.get("_v1_pipeline_effective") or [])
    if resolved and effective:
        return resolved, [str(stage) for stage in effective]
    return (
        [{"stage": "embed", "params": {}}, {"stage": "gmm", "params": {}}],
        ["embed", "gmm"],
    )


def _terminal_params(pipeline_resolved: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in pipeline_resolved:
        if str(entry.get("stage") or "") == "gmm":
            return dict(entry.get("params") or {})
    return {}


@allow_no_run_stage
def run_gmm_bic_argmin_analysis(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: Mapping[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Run GaussianMixture sweep with BIC + argmin selection."""
    if embeddings is None:
        raise ValueError("gmm+bic+argmin requires embedding rows")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("gmm+bic+argmin requires non-empty 2D embeddings")

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
        raise ValueError("gmm sweep requires at least one valid k in [2, n_samples)")

    random_state = int(terminal.get("random_state", parameters.get("random_state", 42)))
    n_init = int(terminal.get("n_init", parameters.get("gmm_n_init", 10)))
    max_iter = int(terminal.get("max_iter", parameters.get("gmm_max_iter", 200)))
    reg_covar = float(terminal.get("reg_covar", parameters.get("gmm_reg_covar", 1.0e-6)))
    covariance_type = str(terminal.get("covariance_type", "full"))

    curve: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    bic_by_k: dict[int, float] = {}
    loglike_by_k: dict[int, float] = {}
    for k in k_range:
        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
        )
        gmm.fit(processed)
        labels = np.asarray(gmm.predict(processed), dtype=np.int64)
        bic = float(gmm.bic(processed))
        avg_loglike = float(gmm.score(processed))
        labels_by_k[int(k)] = labels
        bic_by_k[int(k)] = bic
        loglike_by_k[int(k)] = avg_loglike
        curve.append(
            {
                "k": int(k),
                "bic": bic,
                "avg_log_likelihood": avg_loglike,
                "converged": bool(getattr(gmm, "converged_", False)),
            }
        )

    bics = [float(item["bic"]) for item in curve]
    chosen_k = argmin_choice(k_range, bics)
    chosen_labels = labels_by_k[int(chosen_k)]
    chosen_bic = float(bic_by_k[int(chosen_k)])
    chosen_loglike = float(loglike_by_k[int(chosen_k)])
    cluster_ids, cluster_sizes, noise_count, noise_fraction = cluster_size_summary(chosen_labels)

    selection_curve_name = "gmm_selection_curve.json"
    selection_evidence = build_selection_evidence(
        sweep_range=k_range,
        selection_metric="bic",
        selection_rule="argmin",
        chosen_value=int(chosen_k),
        selection_curve_artifact_ref=selection_curve_name,
        chosen_label="chosen_k",
        selection_rule_params={},
        rationale=f"argmin chose k={chosen_k} with minimum BIC",
    )
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
            "random_state": random_state,
            "n_init": n_init,
            "max_iter": max_iter,
            "reg_covar": reg_covar,
            "covariance_type": covariance_type,
            "k_range": k_range,
            "chosen_k": int(chosen_k),
            "selection_metric": "bic",
            "selection_rule": "argmin",
        },
        "selection_evidence": selection_evidence,
        "preprocess": preprocess_meta,
        "objective": {
            "bic": chosen_bic,
            "avg_log_likelihood": chosen_loglike,
        },
    }
    labels_payload = {
        "cluster_labels": chosen_labels.tolist(),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "parameters": summary["parameters"],
    }
    selection_curve_payload = {
        "metric": "bic",
        "selection_rule": "argmin",
        "points": curve,
        "chosen_k": int(chosen_k),
    }

    return {
        "scalar_results": {
            "n_samples": float(processed.shape[0]),
            "n_features": float(processed.shape[1]),
            "cluster_count": float(len(cluster_ids)),
            "chosen_k": float(chosen_k),
            "objective_bic": chosen_bic,
            "avg_log_likelihood": chosen_loglike,
        },
        "structured_results": {
            "clustering_summary": summary,
            "clustering_labels": labels_payload,
            "selection_evidence": selection_evidence,
        },
        "artifacts": {
            "gmm_summary.json": to_json_bytes(summary),
            "gmm_labels.json": to_json_bytes(labels_payload),
            selection_curve_name: to_json_bytes(selection_curve_payload),
        },
        "result_ref": "gmm_summary.json",
    }
