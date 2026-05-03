"""Per-family tests for ``run_kmeans_fixed_k_analysis`` (Wave 1 PR1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.kmeans_fixed_k_runner import (
    run_kmeans_fixed_k_analysis,
)
from study_query_llm.pipeline.clustering.runner_common import (
    synthesize_fixed_bundled_payload,
)


def _toy_embeddings(n_samples: int = 12, n_features: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


def _attach_payload(method_name: str, params: dict[str, Any], matrix: np.ndarray) -> dict[str, Any]:
    resolved, effective = synthesize_fixed_bundled_payload(
        method_name=method_name,
        parameters=params,
        embedding_dim=int(matrix.shape[1]),
        n_samples=int(matrix.shape[0]),
    )
    out = dict(params)
    out["_v1_pipeline_resolved"] = resolved
    out["_v1_pipeline_effective"] = effective
    return out


@pytest.mark.parametrize(
    "method_name",
    [
        "kmeans+fixed-k",
        "kmeans+normalize+fixed-k",
        "kmeans+pca+fixed-k",
        "kmeans+normalize+pca+fixed-k",
        "spherical-kmeans+approx+fixed-k",
        "spherical-kmeans+approx+pca+fixed-k",
    ],
)
def test_runner_is_deterministic_with_fixed_random_state(method_name: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 3, "random_state": 7, "n_init": 5}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    a = run_kmeans_fixed_k_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    b = run_kmeans_fixed_k_analysis(
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
        run_kmeans_fixed_k_analysis(
            method_name="kmeans+fixed-k",
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
        ("kmeans+fixed-k", "pca_n_components"),
        ("kmeans+fixed-k", "normalize_embeddings"),
        ("kmeans+normalize+fixed-k", "pca_n_components"),
    ],
)
def test_runner_rejects_chain_conflicting_parameters(method_name: str, bad_param: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    params[bad_param] = True if bad_param.endswith("embeddings") else 4
    with pytest.raises(ValueError):
        run_kmeans_fixed_k_analysis(
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
        ("kmeans+fixed-k", []),
        ("kmeans+normalize+fixed-k", ["normalize"]),
        ("kmeans+pca+fixed-k", ["pca"]),
        ("kmeans+normalize+pca+fixed-k", ["normalize", "pca"]),
        ("spherical-kmeans+approx+fixed-k", ["normalize"]),
        ("spherical-kmeans+approx+pca+fixed-k", ["normalize", "pca"]),
    ],
)
def test_summary_records_chain_actually_applied(
    method_name: str, expected_chain: list[str]
) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    result = run_kmeans_fixed_k_analysis(
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
    assert summary["preprocess"]["preprocess_applied"] == expected_chain


@pytest.mark.parametrize(
    "method_name",
    ["spherical-kmeans+approx+fixed-k", "spherical-kmeans+approx+pca+fixed-k"],
)
def test_spherical_kmeans_summary_carries_correct_base_algorithm(method_name: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    result = run_kmeans_fixed_k_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    assert result["structured_results"]["clustering_summary"]["base_algorithm"] == "spherical-kmeans"


def test_pca_n_components_is_required_on_pca_methods() -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="pca_n_components"):
        run_kmeans_fixed_k_analysis(
            method_name="kmeans+pca+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"k": 2},
        )


def test_pca_n_components_over_cap_raises() -> None:
    matrix = _toy_embeddings(n_samples=12, n_features=8)
    with pytest.raises(ValueError, match="exceeds max"):
        run_kmeans_fixed_k_analysis(
            method_name="kmeans+pca+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"k": 2, "pca_n_components": 999},
        )


def test_artifact_and_summary_shape() -> None:
    matrix = _toy_embeddings()
    params = {"k": 2, "pca_n_components": 4}
    result = run_kmeans_fixed_k_analysis(
        method_name="kmeans+normalize+pca+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    assert set(result["artifacts"]) == {"kmeans_summary.json", "kmeans_labels.json"}
    assert result["result_ref"] == "kmeans_summary.json"
    summary = result["structured_results"]["clustering_summary"]
    for key in ("method_name", "base_algorithm", "n_samples", "cluster_ids", "parameters", "preprocess"):
        assert key in summary


def test_direct_invocation_without_payload_uses_grammar_synthesis_and_matches_dispatcher_path() -> None:
    matrix = _toy_embeddings()
    base_params = {"k": 2, "pca_n_components": 4, "random_state": 0, "n_init": 5}

    direct = run_kmeans_fixed_k_analysis(
        method_name="kmeans+normalize+pca+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=base_params,
    )
    via_dispatcher = run_kmeans_fixed_k_analysis(
        method_name="kmeans+normalize+pca+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=_attach_payload("kmeans+normalize+pca+fixed-k", base_params, matrix),
    )
    assert (
        direct["structured_results"]["clustering_labels"]["cluster_labels"]
        == via_dispatcher["structured_results"]["clustering_labels"]["cluster_labels"]
    )
    assert (
        direct["structured_results"]["clustering_summary"]["preprocess"]["preprocess_applied"]
        == ["normalize", "pca"]
    )
