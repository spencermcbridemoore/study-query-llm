"""Tests for sweep serialization helpers."""

from __future__ import annotations

from study_query_llm.experiments.sweep_io import serialize_sweep_result


class _SweepLike:
    """Minimal object matching SweepResult shape for serialization."""

    def __init__(self, by_k):
        self.pca = {}
        self.by_k = by_k
        self.Z = None
        self.Z_norm = None
        self.dist = None


def test_serialize_sweep_result_preserves_tries_under_by_k() -> None:
    tries = [
        {"try_idx": 0, "seed_value": 1, "objective": 0.5},
        {"try_idx": 1, "seed_value": 2, "objective": 0.4},
    ]
    res = _SweepLike(
        {
            "3": {
                "labels": [0, 1, 0],
                "objective": 0.4,
                "objectives": [0.5, 0.4],
                "labels_all": [[0, 1, 0], [0, 1, 1]],
                "tries": tries,
            }
        }
    )
    payload = serialize_sweep_result(res)
    assert payload["by_k"]["3"]["tries"] == tries
