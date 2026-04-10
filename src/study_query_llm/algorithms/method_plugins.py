"""
Method plugin entrypoints for fixed-K and unknown-K clustering workflows.

The unknown-K path currently supports:
- fixed_k_selector (default): run fixed-K sweep and select the best objective
- hdbscan: optional dependency path; falls back to fixed_k_selector if unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from .sweep import SweepConfig, SweepResult, run_sweep

UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR = "fixed_k_selector"
UNKNOWN_K_STRATEGY_HDBSCAN = "hdbscan"


@dataclass(frozen=True)
class MethodPlugin:
    key: str
    description: str
    default_determinism_class: str


def _best_k_from_sweep_result(result: SweepResult) -> Optional[int]:
    best_k: Optional[int] = None
    best_objective: Optional[float] = None
    for k_str, payload in (result.by_k or {}).items():
        try:
            k = int(k_str)
        except (TypeError, ValueError):
            continue
        obj = payload.objective if payload is not None else None
        if obj is None:
            continue
        if best_objective is None or float(obj) < float(best_objective):
            best_objective = float(obj)
            best_k = int(k)
    return best_k


def run_fixed_k_plugin(
    texts: list[str],
    embeddings: np.ndarray,
    config: SweepConfig,
    *,
    paraphraser: Optional[Callable[[str], str]] = None,
    embedder: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> SweepResult:
    """Fixed-K method plugin delegates directly to run_sweep."""
    return run_sweep(
        texts,
        embeddings,
        config,
        paraphraser=paraphraser,
        embedder=embedder,
    )


def run_unknown_k_plugin(
    texts: list[str],
    embeddings: np.ndarray,
    config: SweepConfig,
    *,
    strategy: str = UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR,
    paraphraser: Optional[Callable[[str], str]] = None,
    embedder: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Run unknown-K plugin strategy and return a normalized result envelope.
    """
    strategy_key = str(strategy or UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR).lower()
    if strategy_key == UNKNOWN_K_STRATEGY_HDBSCAN:
        try:
            import hdbscan  # type: ignore

            model = hdbscan.HDBSCAN(metric="euclidean")
            labels = model.fit_predict(embeddings)
            return {
                "strategy": UNKNOWN_K_STRATEGY_HDBSCAN,
                "cluster_labels": labels.tolist(),
                "cluster_count": int(len(set(int(x) for x in labels if int(x) >= 0))),
                "noise_count": int(sum(1 for x in labels if int(x) < 0)),
                "fallback_used": False,
            }
        except Exception as exc:
            # Fallback to fixed-K selector when hdbscan path is unavailable.
            sweep_result = run_fixed_k_plugin(
                texts,
                embeddings,
                config,
                paraphraser=paraphraser,
                embedder=embedder,
            )
            return {
                "strategy": UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR,
                "best_k": _best_k_from_sweep_result(sweep_result),
                "sweep_result": sweep_result,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }

    sweep_result = run_fixed_k_plugin(
        texts,
        embeddings,
        config,
        paraphraser=paraphraser,
        embedder=embedder,
    )
    return {
        "strategy": UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR,
        "best_k": _best_k_from_sweep_result(sweep_result),
        "sweep_result": sweep_result,
        "fallback_used": False,
    }


def available_method_plugins() -> Dict[str, MethodPlugin]:
    """Return plugin metadata keyed by method path."""
    return {
        "fixed_k": MethodPlugin(
            key="fixed_k",
            description="Fixed-K sweep and reducer workflow",
            default_determinism_class="pseudo_deterministic",
        ),
        "unknown_k": MethodPlugin(
            key="unknown_k",
            description="Unknown-K path (fixed-k selector baseline, hdbscan optional)",
            default_determinism_class="best_effort",
        ),
    }
