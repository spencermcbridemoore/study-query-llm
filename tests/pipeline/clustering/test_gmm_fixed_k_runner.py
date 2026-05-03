"""Per-family tests for ``run_gmm_fixed_k_analysis`` (Wave 1 PR1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.gmm_fixed_k_runner import (
    run_gmm_fixed_k_analysis,
)
from study_query_llm.pipeline.clustering.runner_common import (
    synthesize_fixed_bundled_payload,
)


def _toy_embeddings(n_samples: int = 12, n_features: int = 8, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


@pytest.mark.parametrize(
    "method_name",
    [
        "gmm+fixed-k",
        "gmm+normalize+fixed-k",
        "gmm+pca+fixed-k",
        "gmm+normalize+pca+fixed-k",
    ],
)
def test_runner_is_deterministic_with_fixed_random_state(method_name: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2, "random_state": 0, "n_init": 1, "covariance_type": "diag"}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    a = run_gmm_fixed_k_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    b = run_gmm_fixed_k_analysis(
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
        run_gmm_fixed_k_analysis(
            method_name="gmm+fixed-k",
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
        ("gmm+fixed-k", "pca_n_components"),
        ("gmm+fixed-k", "gmm_normalize_embeddings"),
        ("gmm+normalize+fixed-k", "pca_n_components"),
    ],
)
def test_runner_rejects_chain_conflicting_parameters(method_name: str, bad_param: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    params[bad_param] = True if bad_param.endswith("embeddings") else 4
    with pytest.raises(ValueError):
        run_gmm_fixed_k_analysis(
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
        ("gmm+fixed-k", []),
        ("gmm+normalize+fixed-k", ["normalize"]),
        ("gmm+pca+fixed-k", ["pca"]),
        ("gmm+normalize+pca+fixed-k", ["normalize", "pca"]),
    ],
)
def test_summary_records_chain_actually_applied(
    method_name: str, expected_chain: list[str]
) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"k": 2}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    result = run_gmm_fixed_k_analysis(
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
        run_gmm_fixed_k_analysis(
            method_name="gmm+pca+fixed-k",
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
        run_gmm_fixed_k_analysis(
            method_name="gmm+pca+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"k": 2, "pca_n_components": 999},
        )


def test_artifact_and_summary_shape() -> None:
    matrix = _toy_embeddings()
    result = run_gmm_fixed_k_analysis(
        method_name="gmm+normalize+pca+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters={"k": 2, "pca_n_components": 4},
    )
    assert set(result["artifacts"]) == {"gmm_summary.json", "gmm_labels.json"}
    assert result["result_ref"] == "gmm_summary.json"
    summary = result["structured_results"]["clustering_summary"]
    assert summary["base_algorithm"] == "gmm"


def test_direct_invocation_without_payload_matches_dispatcher_path() -> None:
    matrix = _toy_embeddings()
    base_params = {"k": 2, "pca_n_components": 4, "random_state": 0}
    direct = run_gmm_fixed_k_analysis(
        method_name="gmm+normalize+pca+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=base_params,
    )
    resolved, effective = synthesize_fixed_bundled_payload(
        method_name="gmm+normalize+pca+fixed-k",
        parameters=base_params,
        embedding_dim=int(matrix.shape[1]),
        n_samples=int(matrix.shape[0]),
    )
    via_dispatcher = run_gmm_fixed_k_analysis(
        method_name="gmm+normalize+pca+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters={
            **base_params,
            "_v1_pipeline_resolved": resolved,
            "_v1_pipeline_effective": effective,
        },
    )
    assert (
        direct["structured_results"]["clustering_labels"]["cluster_labels"]
        == via_dispatcher["structured_results"]["clustering_labels"]["cluster_labels"]
    )
