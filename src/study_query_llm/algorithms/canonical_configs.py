"""
Canonical config builders for registered methods.

**Unused.** Provides a single future-source-of-truth for the shape of
``config_json`` per ``(method_name, method_version)``. Each builder is a
**pure function** that takes a loose ``params`` dict and returns a
normalized config dict (sorted keys, canonical key names, types coerced,
nullables defaulted) ready to be hashed into a run fingerprint.

Adopting per-method requires a deliberate decision because adoption changes
the ``fingerprint_hash`` for that method going forward (the fingerprint
hashes ``config_json`` minus scheduling-only keys, so any change to the
canonical shape is observable downstream). This module is therefore created
in advance but not yet imported by any writer.

Coverage (Option B from ``harmless-now-prelim-prep``):

* Clustering composites + components in
  :mod:`study_query_llm.algorithms.recipes`:
  ``cosine_kllmeans_no_pca@1.0``, ``mean_pool_tokens@1.0``,
  ``pca_svd_project@1.0``, ``kmeanspp_init@1.0``, ``k_llmmeans@1.0``,
  ``umap_project@1.0``.
* MCQ probe + analyses:
  ``mcq_answer_position_probe@1.0``, ``mcq_compliance_metrics@1.0``,
  ``mcq_answer_position_distribution@1.0``,
  ``mcq_answer_position_chi_square@1.0``.
* Register-only text-classification methods:
  ``knn_prototype_classifier@0.1``, ``linear_probe_logreg@0.1``,
  ``label_embedding_zero_shot@0.1``, ``prompted_llm_classifier@0.1``,
  ``mixture_of_experts_classifier@0.1``.

Critical invariants:

* Builders MUST NOT import anything from
  :mod:`study_query_llm.services` or :mod:`study_query_llm.experiments`.
* Builders MUST NOT mutate their input ``params`` dict.
* Returned dicts MUST be JSON-serialisable and key-sorted-stable (callers
  may json.dumps with ``sort_keys=True``).
* No writer in :mod:`study_query_llm.services` or
  :mod:`study_query_llm.experiments` is allowed to import this module yet;
  see ``tests/test_services/test_canonical_configs.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple


CanonicalConfigBuilder = Callable[[Dict[str, Any]], Dict[str, Any]]


def _opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _opt_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _opt_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    return list(value)


def _sorted(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with keys in sorted order (top level only)."""
    return {k: d[k] for k in sorted(d.keys())}


# ---------------------------------------------------------------------------
# Clustering: components
# ---------------------------------------------------------------------------


def _build_mean_pool_tokens_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "normalize": _opt_bool(p.get("normalize")),
    })


def _build_pca_svd_project_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "pca_dim": _opt_int(p.get("pca_dim")),
    })


def _build_kmeanspp_init_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "seed_space": _opt_str(p.get("seed_space")),
    })


def _build_k_llmmeans_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "max_iter": _opt_int(p.get("max_iter")),
        "distance_metric": _opt_str(p.get("distance_metric")),
        "normalize_vectors": _opt_bool(p.get("normalize_vectors")),
        "llm_interval": _opt_int(p.get("llm_interval")),
        "max_samples": _opt_int(p.get("max_samples")),
        "coverage_threshold": _opt_float(p.get("coverage_threshold")),
    })


def _build_umap_project_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "n_neighbors": _opt_int(p.get("n_neighbors")),
        "min_dist": _opt_float(p.get("min_dist")),
        "n_components": _opt_int(p.get("n_components")),
        "metric": _opt_str(p.get("metric")),
        "random_state": _opt_int(p.get("random_state")),
    })


# ---------------------------------------------------------------------------
# Clustering: composite
# ---------------------------------------------------------------------------


def _build_cosine_kllmeans_no_pca_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "k_min": _opt_int(p.get("k_min")),
        "k_max": _opt_int(p.get("k_max")),
        "n_restarts": _opt_int(p.get("n_restarts")),
    })


# ---------------------------------------------------------------------------
# MCQ
# ---------------------------------------------------------------------------


def _build_mcq_answer_position_probe_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "deployment": _opt_str(p.get("deployment")),
        "level": _opt_str(p.get("level")),
        "subject": _opt_str(p.get("subject")),
        "options_per_question": _opt_int(p.get("options_per_question")),
        "questions_per_test": _opt_int(p.get("questions_per_test")),
        "label_style": _opt_str(p.get("label_style")),
        "spread_correct_answer_uniformly": _opt_bool(
            p.get("spread_correct_answer_uniformly")
        ),
        "samples_per_combo": _opt_int(p.get("samples_per_combo")),
        "template_version": _opt_str(p.get("template_version")),
    })


def _build_mcq_compliance_metrics_v1_0(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "request_id": _opt_int(p.get("request_id")),
        "analysis_key": _opt_str(p.get("analysis_key")),
    })


def _build_mcq_answer_position_distribution_v1_0(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "request_id": _opt_int(p.get("request_id")),
        "analysis_key": _opt_str(p.get("analysis_key")),
    })


def _build_mcq_answer_position_chi_square_v1_0(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "request_id": _opt_int(p.get("request_id")),
        "analysis_key": _opt_str(p.get("analysis_key")),
    })


# ---------------------------------------------------------------------------
# Text classification (register-only)
# ---------------------------------------------------------------------------


def _build_knn_prototype_classifier_v0_1(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "k": _opt_int(p.get("k")),
        "metric": _opt_str(p.get("metric")),
        "weighting": _opt_str(p.get("weighting")),
        "normalize_embeddings": _opt_bool(p.get("normalize_embeddings")),
        "tie_break": _opt_str(p.get("tie_break")),
    })


def _build_linear_probe_logreg_v0_1(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "regularization": _opt_str(p.get("regularization")),
        "C": _opt_float(p.get("C")),
        "max_iter": _opt_int(p.get("max_iter")),
        "class_weight": _opt_str(p.get("class_weight")),
        "solver": _opt_str(p.get("solver")),
        "normalize_embeddings": _opt_bool(p.get("normalize_embeddings")),
        "random_state": _opt_int(p.get("random_state")),
    })


def _build_label_embedding_zero_shot_v0_1(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "label_template": _opt_str(p.get("label_template")),
        "metric": _opt_str(p.get("metric")),
        "normalize_embeddings": _opt_bool(p.get("normalize_embeddings")),
        "temperature": _opt_float(p.get("temperature")),
    })


def _build_prompted_llm_classifier_v0_1(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "prompt_template_id": _opt_str(p.get("prompt_template_id")),
        "n_shot": _opt_int(p.get("n_shot")),
        "deployment": _opt_str(p.get("deployment")),
        "temperature": _opt_float(p.get("temperature")),
        "max_tokens": _opt_int(p.get("max_tokens")),
        "label_parser": _opt_str(p.get("label_parser")),
        "label_set": _opt_list(p.get("label_set")),
    })


def _build_mixture_of_experts_classifier_v0_1(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    p = dict(params or {})
    return _sorted({
        "experts": _opt_list(p.get("experts")),
        "gating_strategy": _opt_str(p.get("gating_strategy")),
        "gating_temperature": _opt_float(p.get("gating_temperature")),
        "combine": _opt_str(p.get("combine")),
        "normalize_embeddings": _opt_bool(p.get("normalize_embeddings")),
    })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CANONICAL_CONFIG_BUILDERS: Dict[Tuple[str, str], CanonicalConfigBuilder] = {
    # Clustering components
    ("mean_pool_tokens", "1.0"): _build_mean_pool_tokens_v1_0,
    ("pca_svd_project", "1.0"): _build_pca_svd_project_v1_0,
    ("kmeanspp_init", "1.0"): _build_kmeanspp_init_v1_0,
    ("k_llmmeans", "1.0"): _build_k_llmmeans_v1_0,
    ("umap_project", "1.0"): _build_umap_project_v1_0,
    # Clustering composite
    ("cosine_kllmeans_no_pca", "1.0"): _build_cosine_kllmeans_no_pca_v1_0,
    # MCQ probe + analyses
    ("mcq_answer_position_probe", "1.0"): _build_mcq_answer_position_probe_v1_0,
    ("mcq_compliance_metrics", "1.0"): _build_mcq_compliance_metrics_v1_0,
    (
        "mcq_answer_position_distribution",
        "1.0",
    ): _build_mcq_answer_position_distribution_v1_0,
    (
        "mcq_answer_position_chi_square",
        "1.0",
    ): _build_mcq_answer_position_chi_square_v1_0,
    # Text classification (register-only)
    ("knn_prototype_classifier", "0.1"): _build_knn_prototype_classifier_v0_1,
    ("linear_probe_logreg", "0.1"): _build_linear_probe_logreg_v0_1,
    ("label_embedding_zero_shot", "0.1"): _build_label_embedding_zero_shot_v0_1,
    ("prompted_llm_classifier", "0.1"): _build_prompted_llm_classifier_v0_1,
    (
        "mixture_of_experts_classifier",
        "0.1",
    ): _build_mixture_of_experts_classifier_v0_1,
}


def canonical_config_for(
    method_name: str,
    method_version: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the canonical ``config_json`` for the given method.

    Pure: ``params`` is not mutated; the returned dict is freshly constructed.

    Args:
        method_name: Registered method name (e.g. ``"k_llmmeans"``).
        method_version: Registered method version (e.g. ``"1.0"``).
        params: Loose params dict from any caller.

    Returns:
        A new dict with canonical keys/types/order suitable for hashing into
        a run fingerprint.

    Raises:
        KeyError: When no builder is registered for ``(method_name,
            method_version)``. Callers SHOULD treat this as a hard error
            rather than silently fall back to ``params`` -- a missing
            builder means provenance shape for that method is undefined.
    """
    key = (method_name, method_version)
    builder = CANONICAL_CONFIG_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"No canonical_config builder registered for "
            f"{method_name}@{method_version}. Known builders: "
            f"{sorted(CANONICAL_CONFIG_BUILDERS.keys())}"
        )
    return builder(copy.deepcopy(dict(params or {})))
