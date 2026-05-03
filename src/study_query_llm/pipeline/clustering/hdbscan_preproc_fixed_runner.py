"""Grammar-bound HDBSCAN with normalize/pca preprocessing (sibling to ``hdbscan_runner``)."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from study_query_llm.pipeline.runner import allow_no_run_stage

from .grammar import parse_method_name
from .runner_common import (
    assert_no_chain_conflicts,
    preprocess_for_effective_pipeline,
    resolve_pipeline_or_synthesize,
)

HDBSCAN_PREPROC_DEFAULT_METRIC = "euclidean"
HDBSCAN_DEFAULT_RANDOM_STATE = 0
HDBSCAN_DEFAULT_CORE_DIST_N_JOBS = 1
HDBSCAN_DEFAULT_APPROX_MIN_SPAN_TREE = False


def _bool_param(params: Mapping[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key in params:
            value = params[key]
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off"}:
                    return False
            return bool(value)
    return default


def _optional_int_param(
    params: Mapping[str, Any],
    *keys: str,
) -> int | None:
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if value in (None, ""):
            return None
        return int(value)
    return None


def _required_int_param(
    params: Mapping[str, Any],
    *keys: str,
    default: int,
) -> int:
    for key in keys:
        if key in params:
            return int(params[key])
    return int(default)


def _float_param(
    params: Mapping[str, Any],
    *keys: str,
    default: float,
) -> float:
    for key in keys:
        if key in params:
            return float(params[key])
    return float(default)


def _str_param(
    params: Mapping[str, Any],
    *keys: str,
    default: str,
) -> str:
    for key in keys:
        if key in params and params[key] is not None:
            return str(params[key]).strip()
    return str(default)


def _to_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


@allow_no_run_stage
def run_hdbscan_preproc_fixed_analysis(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: Mapping[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """HDBSCAN on preprocess-transformed embeddings (grammar-bound chain)."""
    params = dict(parameters or {})
    resolved_representation = str(
        input_group_metadata.get("representation")
        or params.get("embedding_representation")
        or params.get("representation_type")
        or "full"
    ).strip().lower()
    if resolved_representation != "full":
        raise ValueError(
            "HDBSCAN analysis requires embedding representation 'full' "
            f"(one vector per text row), got {resolved_representation!r}"
        )

    if embeddings is None:
        raise ValueError("HDBSCAN preprocess variants require embedding_matrix on the input group")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"HDBSCAN analysis expected 2D embeddings, got shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("HDBSCAN analysis requires at least one embedding row")

    _, chain, _fit_tok = parse_method_name(method_name)
    assert_no_chain_conflicts(method_name=method_name, parameters=params, chain=chain)

    if _bool_param(params, "hdbscan_normalize_embeddings", "normalize_embeddings", default=False):
        raise ValueError(
            f"{method_name} encodes normalize via the method name; "
            "do not set hdbscan_normalize_embeddings or normalize_embeddings"
        )

    pipeline_resolved, pipeline_effective = resolve_pipeline_or_synthesize(
        method_name=method_name,
        parameters=params,
        embeddings=matrix,
    )
    processed, _pre_meta = preprocess_for_effective_pipeline(
        embeddings=matrix,
        pipeline_resolved=pipeline_resolved,
        pipeline_effective=pipeline_effective,
    )

    min_cluster_size: int | None = None
    for key in ("min_cluster_size", "hdbscan_min_cluster_size"):
        if key in params and params[key] is not None:
            min_cluster_size = int(params[key])
            break
    if min_cluster_size is None:
        raise ValueError(f"{method_name} requires integer parameter min_cluster_size")
    if min_cluster_size < 2:
        raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")
    min_samples = _optional_int_param(params, "hdbscan_min_samples", "min_samples")
    metric = _str_param(
        params,
        "hdbscan_metric",
        "metric",
        default=HDBSCAN_PREPROC_DEFAULT_METRIC,
    ).lower()
    cluster_selection_method = _str_param(
        params,
        "hdbscan_cluster_selection_method",
        "cluster_selection_method",
        default="eom",
    ).lower()
    if cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError(
            "hdbscan_cluster_selection_method must be 'eom' or 'leaf', "
            f"got {cluster_selection_method!r}"
        )
    cluster_selection_epsilon = _float_param(
        params,
        "hdbscan_cluster_selection_epsilon",
        "cluster_selection_epsilon",
        default=0.0,
    )
    alpha = _float_param(params, "hdbscan_alpha", "alpha", default=1.0)
    allow_single_cluster = _bool_param(
        params,
        "hdbscan_allow_single_cluster",
        "allow_single_cluster",
        default=False,
    )
    random_state = _required_int_param(
        params,
        "hdbscan_random_state",
        "random_state",
        default=HDBSCAN_DEFAULT_RANDOM_STATE,
    )
    core_dist_n_jobs = _required_int_param(
        params,
        "hdbscan_core_dist_n_jobs",
        "core_dist_n_jobs",
        default=HDBSCAN_DEFAULT_CORE_DIST_N_JOBS,
    )
    if core_dist_n_jobs == 0:
        raise ValueError("hdbscan_core_dist_n_jobs must be != 0")
    approx_min_span_tree = _bool_param(
        params,
        "hdbscan_approx_min_span_tree",
        "approx_min_span_tree",
        default=HDBSCAN_DEFAULT_APPROX_MIN_SPAN_TREE,
    )

    try:
        import hdbscan  # type: ignore
    except Exception as exc:  # pragma: no cover - import path is env-dependent
        raise RuntimeError(
            "HDBSCAN analysis requires the optional dependency 'hdbscan'. "
            "Install it via pip or conda before using bundled HDBSCAN methods."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "alpha": alpha,
        "allow_single_cluster": allow_single_cluster,
        "random_state": random_state,
        "core_dist_n_jobs": core_dist_n_jobs,
        "approx_min_span_tree": approx_min_span_tree,
    }
    model = hdbscan.HDBSCAN(**model_kwargs)
    random_state_applied = True
    try:
        labels = np.asarray(model.fit_predict(processed), dtype=np.int64)
    except TypeError as exc:
        if "random_state" not in str(exc):
            raise
        model_kwargs.pop("random_state", None)
        model = hdbscan.HDBSCAN(**model_kwargs)
        labels = np.asarray(model.fit_predict(processed), dtype=np.int64)
        random_state_applied = False

    n_samples = int(labels.shape[0])
    cluster_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    cluster_sizes = {
        str(cluster_id): int(np.sum(labels == cluster_id))
        for cluster_id in cluster_ids
    }
    noise_count = int(np.sum(labels < 0))
    noise_fraction = float(noise_count / n_samples) if n_samples > 0 else 0.0
    largest_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0
    probabilities = np.asarray(getattr(model, "probabilities_", np.array([])), dtype=np.float64)
    mean_membership_probability = (
        float(np.mean(probabilities))
        if probabilities.shape[0] == n_samples and n_samples > 0
        else 0.0
    )
    outlier_scores = np.asarray(getattr(model, "outlier_scores_", np.array([])), dtype=np.float64)
    mean_outlier_score = (
        float(np.mean(outlier_scores))
        if outlier_scores.shape[0] == n_samples and n_samples > 0
        else 0.0
    )

    used_parameters: dict[str, Any] = {
        "hdbscan_min_cluster_size": int(min_cluster_size),
        "hdbscan_min_samples": int(min_samples) if min_samples is not None else None,
        "hdbscan_metric": metric,
        "hdbscan_cluster_selection_method": cluster_selection_method,
        "hdbscan_cluster_selection_epsilon": float(cluster_selection_epsilon),
        "hdbscan_alpha": float(alpha),
        "hdbscan_allow_single_cluster": bool(allow_single_cluster),
        "hdbscan_normalize_embeddings": False,
        "hdbscan_random_state": int(random_state),
        "hdbscan_random_state_applied": bool(random_state_applied),
        "hdbscan_core_dist_n_jobs": int(core_dist_n_jobs),
        "hdbscan_approx_min_span_tree": bool(approx_min_span_tree),
    }

    summary: dict[str, Any] = {
        "method_name": str(method_name),
        "base_algorithm": "hdbscan",
        "input_group_id": int(input_group_id),
        "input_group_type": str(input_group_type),
        "input_group_metadata": dict(input_group_metadata or {}),
        "text_count": int(len(texts)),
        "n_samples": int(n_samples),
        "n_features": int(processed.shape[1]),
        "cluster_count": int(len(cluster_ids)),
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "largest_cluster_size": int(largest_cluster_size),
        "noise_count": int(noise_count),
        "noise_fraction": float(noise_fraction),
        "mean_membership_probability": float(mean_membership_probability),
        "mean_outlier_score": float(mean_outlier_score),
        "parameters": used_parameters,
    }
    labels_payload: dict[str, Any] = {
        "cluster_labels": labels.tolist(),
        "noise_label": -1,
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "parameters": used_parameters,
    }

    return {
        "scalar_results": {
            "n_samples": float(n_samples),
            "n_features": float(processed.shape[1]),
            "cluster_count": float(len(cluster_ids)),
            "noise_count": float(noise_count),
            "noise_fraction": float(noise_fraction),
            "largest_cluster_size": float(largest_cluster_size),
            "mean_membership_probability": float(mean_membership_probability),
            "mean_outlier_score": float(mean_outlier_score),
        },
        "structured_results": {
            "hdbscan_summary": summary,
            "hdbscan_cluster_labels": labels_payload,
        },
        "artifacts": {
            "hdbscan_summary.json": _to_json_bytes(summary),
            "hdbscan_labels.json": _to_json_bytes(labels_payload),
        },
        "result_ref": "hdbscan_summary.json",
    }
