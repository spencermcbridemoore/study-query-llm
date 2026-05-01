"""Tests for agglomerative fixed-k clustering runner."""

from __future__ import annotations

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.agglomerative_runner import (
    run_agglomerative_fixed_k_analysis,
)


def _toy_embeddings() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ],
        dtype=np.float64,
    )


def test_agglomerative_runner_is_deterministic_for_same_input() -> None:
    matrix = _toy_embeddings()
    params = {"k": 2}
    first = run_agglomerative_fixed_k_analysis(
        method_name="agglomerative+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    second = run_agglomerative_fixed_k_analysis(
        method_name="agglomerative+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    assert first["structured_results"]["clustering_labels"]["cluster_labels"] == second[
        "structured_results"
    ]["clustering_labels"]["cluster_labels"]


@pytest.mark.parametrize("params", [{}, {"k": 1}, {"k": 6}, {"k": 10}])
def test_agglomerative_runner_validates_k_bounds(params: dict[str, int]) -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="k"):
        run_agglomerative_fixed_k_analysis(
            method_name="agglomerative+fixed-k",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
            parameters=params,
        )


def test_agglomerative_runner_artifact_and_summary_shape() -> None:
    matrix = _toy_embeddings()
    result = run_agglomerative_fixed_k_analysis(
        method_name="agglomerative+fixed-k",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters={"k": 2},
    )

    summary = result["structured_results"]["clustering_summary"]
    assert summary["base_algorithm"] == "agglomerative"
    assert summary["parameters"]["k"] == 2
    assert result["result_ref"] == "agglomerative_summary.json"
    assert "agglomerative_summary.json" in result["artifacts"]
    assert "agglomerative_labels.json" in result["artifacts"]
