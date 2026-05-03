"""Tests for sweep kmeans runner (bundled grammar)."""

from __future__ import annotations

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.kmeans_runner import (
    run_kmeans_silhouette_kneedle_analysis,
)


def test_kmeans_sweep_raises_without_synthesized_pipeline() -> None:
    matrix = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Slice 2 invariant lock"):
        run_kmeans_silhouette_kneedle_analysis(
            method_name="kmeans+normalize+pca+sweep",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=["a", "b", "c"],
            parameters={},
        )
