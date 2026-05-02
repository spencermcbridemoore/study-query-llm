"""Tests for canonical clustering runners and registry envelope invariants.

Slice 1.5 retired the ``clustering_v1`` provenance envelope and renamed the
three sweep methods under the bundled-grammar (`<algorithm>+<chain>`). The
v1 resolver/validators/hashing/schema modules and ``rules-v1.0.0.yaml`` were
deleted, so this file no longer exercises any v1 envelope paths. Reproducibility
of the sweep runners is now verified by feeding the runner a literal
``_v1_pipeline_{resolved,effective}`` payload (the same shape the analyze
dispatcher injects via ``_synthesize_v1_pipeline_for_bundled_method``).
"""

from __future__ import annotations

import numpy as np

from study_query_llm.pipeline.clustering import (
    iter_algorithm_specs,
    run_gmm_bic_argmin_analysis,
    run_kmeans_silhouette_kneedle_analysis,
)


def _toy_embeddings() -> np.ndarray:
    cluster_a = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, -0.1],
            [0.3, 0.0],
            [0.15, 0.2],
            [0.05, -0.2],
        ],
        dtype=np.float64,
    )
    cluster_b = np.asarray(
        [
            [5.0, 5.1],
            [4.9, 4.8],
            [5.2, 5.0],
            [4.8, 5.2],
            [5.1, 4.9],
            [5.3, 5.1],
        ],
        dtype=np.float64,
    )
    return np.vstack([cluster_a, cluster_b])


def _kmeans_pipeline_payload(matrix: np.ndarray) -> dict[str, object]:
    """Mirror the synthesizer payload used by the bundled-grammar dispatcher."""
    pca_components = max(1, min(100, int(matrix.shape[1]), max(1, int(matrix.shape[0]) - 1)))
    pipeline_resolved: list[dict[str, object]] = [
        {"stage": "embed", "params": {}},
        {"stage": "normalize", "params": {}},
        {
            "stage": "pca",
            "params": {"random_state": 42, "n_components": pca_components},
        },
        {
            "stage": "kmeans",
            "params": {
                "n_init": 20,
                "max_iter": 300,
                "init": "k-means++",
                "random_state": 42,
                "distance_metric": "cosine",
                "k_range": [2, 3, 5, 8, 10, 15, 20, 30, 50],
                "selection_metric": "silhouette",
                "selection_rule": "kneedle",
            },
        },
    ]
    return {
        "_v1_pipeline_resolved": pipeline_resolved,
        "_v1_pipeline_effective": ["embed", "normalize", "pca", "kmeans"],
    }


def _gmm_pipeline_payload(matrix: np.ndarray) -> dict[str, object]:
    pca_components = max(1, min(100, int(matrix.shape[1]), max(1, int(matrix.shape[0]) - 1)))
    pipeline_resolved: list[dict[str, object]] = [
        {"stage": "embed", "params": {}},
        {"stage": "normalize", "params": {}},
        {
            "stage": "pca",
            "params": {"random_state": 42, "n_components": pca_components},
        },
        {
            "stage": "gmm",
            "params": {
                "reg_covar": 1.0e-6,
                "n_init": 10,
                "max_iter": 200,
                "random_state": 42,
                "covariance_type": "full",
                "k_range": [2, 3, 5, 8, 10, 15, 20, 30, 50],
                "selection_metric": "bic",
                "selection_rule": "argmin",
            },
        },
    ]
    return {
        "_v1_pipeline_resolved": pipeline_resolved,
        "_v1_pipeline_effective": ["embed", "normalize", "pca", "gmm"],
    }


def test_all_registry_methods_are_envelope_none() -> None:
    """Slice 1.5 invariant: every spec in the runtime registry ships with
    ``provenance_envelope == "none"``. The ``clustering_v1`` envelope value
    was removed alongside the v1 resolver/validators; only ``"none"`` remains
    in the ``ProvenanceEnvelope`` literal.
    """
    specs = iter_algorithm_specs()
    assert specs, "registry should expose at least one algorithm spec"
    for spec in specs:
        assert spec.provenance_envelope == "none", (
            f"{spec.method_name!r} carried envelope={spec.provenance_envelope!r}; "
            "all registry methods must be envelope=none after Slice 1.5."
        )


def test_kmeans_runner_reproducible_selection() -> None:
    """Two identical calls to the kmeans runner with a literal
    ``_v1_pipeline_{resolved,effective}`` payload must produce the same
    ``chosen_k`` and the same selection curve artifact."""
    matrix = _toy_embeddings()
    params = _kmeans_pipeline_payload(matrix)
    first = run_kmeans_silhouette_kneedle_analysis(
        method_name="kmeans+normalize+pca+sweep",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    second = run_kmeans_silhouette_kneedle_analysis(
        method_name="kmeans+normalize+pca+sweep",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    first_sel = first["structured_results"]["selection_evidence"]
    second_sel = second["structured_results"]["selection_evidence"]
    assert first_sel["chosen_k"] == second_sel["chosen_k"]
    assert first_sel["chosen_k"] in first_sel["sweep_range"]
    assert "kmeans_selection_curve.json" in first["artifacts"]


def test_gmm_runner_reproducible_selection() -> None:
    """GMM counterpart of ``test_kmeans_runner_reproducible_selection``."""
    matrix = _toy_embeddings()
    params = _gmm_pipeline_payload(matrix)
    first = run_gmm_bic_argmin_analysis(
        method_name="gmm+normalize+pca+sweep",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    second = run_gmm_bic_argmin_analysis(
        method_name="gmm+normalize+pca+sweep",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    first_sel = first["structured_results"]["selection_evidence"]
    second_sel = second["structured_results"]["selection_evidence"]
    assert first_sel["chosen_k"] == second_sel["chosen_k"]
    assert first_sel["chosen_k"] in first_sel["sweep_range"]
    assert "gmm_selection_curve.json" in first["artifacts"]
