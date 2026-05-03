"""Per-family tests for ``run_agglomerative_preproc_fixed_k_analysis`` (Wave 1 PR2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.agglomerative_preproc_runner import (
    run_agglomerative_preproc_fixed_k_analysis,
)


def _toy_embeddings(n_samples: int = 12, n_features: int = 8, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


@pytest.mark.parametrize(
    "method_name",
    ["agglomerative+normalize+fixed-k", "agglomerative+pca+fixed-k"],
)
def test_runner_is_deterministic_without_random_state(method_name: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 3}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    a = run_agglomerative_preproc_fixed_k_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    b = run_agglomerative_preproc_fixed_k_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    assert (
        a["structured_results"]["clustering_labels"]["cluster_labels"]
        == b["structured_results"]["clustering_labels"]["cluster_labels"]
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"k": 1},
        {"k": 12},
    ],
)
def test_runner_validates_k_bounds(params: dict[str, int]) -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError):
        run_agglomerative_preproc_fixed_k_analysis(
            method_name="agglomerative+normalize+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters=params,
        )


@pytest.mark.parametrize(
    ("method_name", "bad_param"),
    [
        ("agglomerative+normalize+fixed-k", "pca_n_components"),
        ("agglomerative+pca+fixed-k", "normalize_embeddings"),
    ],
)
def test_runner_rejects_chain_conflicting_parameters(method_name: str, bad_param: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    params[bad_param] = True if bad_param.endswith("embeddings") else 4
    with pytest.raises(ValueError):
        run_agglomerative_preproc_fixed_k_analysis(
            method_name=method_name,
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters=params,
        )


@pytest.mark.parametrize(
    ("method_name", "expected_chain"),
    [
        ("agglomerative+normalize+fixed-k", ["normalize"]),
        ("agglomerative+pca+fixed-k", ["pca"]),
    ],
)
def test_summary_records_chain_actually_applied(
    method_name: str, expected_chain: list[str]
) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    result = run_agglomerative_preproc_fixed_k_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    summary = result["structured_results"]["clustering_summary"]
    assert summary["parameters"]["preprocess_applied"] == expected_chain


def test_pca_n_components_is_required_on_pca_methods() -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="pca_n_components"):
        run_agglomerative_preproc_fixed_k_analysis(
            method_name="agglomerative+pca+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"k": 2},
        )


def test_pca_n_components_over_cap_raises() -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="exceeds max"):
        run_agglomerative_preproc_fixed_k_analysis(
            method_name="agglomerative+pca+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"k": 2, "pca_n_components": 999},
        )


def test_artifact_shape_parity_with_baseline() -> None:
    matrix = _toy_embeddings()
    result = run_agglomerative_preproc_fixed_k_analysis(
        method_name="agglomerative+normalize+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters={"k": 2},
    )
    assert set(result["artifacts"]) == {"agglomerative_summary.json", "agglomerative_labels.json"}
    assert result["result_ref"] == "agglomerative_summary.json"
    summary = result["structured_results"]["clustering_summary"]
    assert summary["base_algorithm"] == "agglomerative"
    for key in ("method_name", "n_samples", "n_features", "cluster_ids", "parameters", "preprocess"):
        assert key in summary


def test_ward_linkage_forces_euclidean_metric() -> None:
    matrix = _toy_embeddings()
    result = run_agglomerative_preproc_fixed_k_analysis(
        method_name="agglomerative+normalize+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters={"k": 2, "linkage": "ward", "metric": "cosine"},
    )
    used = result["structured_results"]["clustering_summary"]["parameters"]
    assert used["linkage"] == "ward"
    assert used["metric"] == "euclidean"
