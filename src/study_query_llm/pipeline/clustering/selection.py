"""Selection helpers for sweep-and-select clustering methods."""

from __future__ import annotations

from typing import Any

import numpy as np


def kneedle_choice(
    x_values: list[int],
    y_values: list[float],
) -> int:
    """Return knee location using a deterministic normalized-distance heuristic.

    This is intentionally dependency-free to keep selection deterministic across
    environments (no optional third-party runtime package required).
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if not x_values:
        raise ValueError("kneedle_choice requires at least one candidate")
    if len(x_values) == 1:
        return int(x_values[0])

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.allclose(y, y[0]):
        return int(x_values[int(np.argmax(y))])

    # Normalize x/y into [0, 1] then pick max distance from diagonal.
    x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1e-12)
    y_norm = (y - y.min()) / max(float(y.max() - y.min()), 1e-12)
    distances = y_norm - x_norm
    best_idx = int(np.argmax(distances))
    return int(x_values[best_idx])


def argmin_choice(
    x_values: list[int],
    y_values: list[float],
) -> int:
    """Return x at minimum y (stable first-min tie break)."""
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if not x_values:
        raise ValueError("argmin_choice requires at least one candidate")
    best_idx = int(np.argmin(np.asarray(y_values, dtype=np.float64)))
    return int(x_values[best_idx])


def build_selection_evidence(
    *,
    sweep_range: list[int],
    selection_metric: str,
    selection_rule: str,
    chosen_value: int,
    selection_curve_artifact_ref: str,
    chosen_label: str = "chosen_k",
    selection_rule_params: dict[str, Any] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Build standard selection-evidence summary payload."""
    payload: dict[str, Any] = {
        "sweep_range": [int(v) for v in sweep_range],
        "selection_metric": str(selection_metric),
        "selection_rule": str(selection_rule),
        "selection_rule_params": dict(selection_rule_params or {}),
        "selection_curve_artifact_ref": str(selection_curve_artifact_ref),
        str(chosen_label): int(chosen_value),
    }
    if rationale:
        payload[f"{chosen_label}_rationale"] = str(rationale)
    return payload
