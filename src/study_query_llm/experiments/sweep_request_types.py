"""
Deterministic request/target schema for clustering_sweep_request lifecycle.

Provides helpers to expand parameter axes into run targets and build run_key
strings that match the convention used by ingestion and run_300_bigrun_sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Schema version for request metadata (bump when breaking changes)
REQUEST_SCHEMA_VERSION = 1

# Valid request_status values
REQUEST_STATUS_REQUESTED = "requested"
REQUEST_STATUS_RUNNING = "running"
REQUEST_STATUS_FULFILLED = "fulfilled"
REQUEST_STATUS_CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunTarget:
    """Single run target from expanded parameter axes."""

    dataset: str
    embedding_engine: str
    summarizer: str
    entry_max: int
    n_restarts_suffix: str  # e.g. "50runs"


def _safe_name(s: str) -> str:
    """Normalize string for use in run_key (matches run_300_bigrun_sweep._safe_name)."""
    return s.replace("-", "_").replace("/", "_")


def normalize_summarizer(value: Any) -> str:
    """Convert summarizer value to canonical string for run_key.

    None -> "None", otherwise str(value).
    """
    if value is None:
        return "None"
    return str(value)


def build_run_key(
    dataset: str,
    embedding_engine: str,
    summarizer: str,
    entry_max: int,
    n_restarts_suffix: str = "50runs",
) -> str:
    """Build deterministic run_key matching current ingestion convention.

    Format: {dataset}_{engine_safe}_{sum_safe}_{entry_max}_{n_restarts_suffix}

    Must match the logic in run_300_bigrun_sweep.py:
        run_key = f"{dataset_name}_{engine_safe}_{sum_safe}_{ENTRY_MAX}_50runs"
    """
    engine_safe = _safe_name(embedding_engine)
    sum_safe = _safe_name(normalize_summarizer(summarizer))
    return f"{dataset}_{engine_safe}_{sum_safe}_{entry_max}_{n_restarts_suffix}"


def expand_parameter_axes(
    parameter_axes: Dict[str, List[Any]],
    entry_max: int,
    n_restarts_suffix: str = "50runs",
) -> List[RunTarget]:
    """Expand parameter axes into deterministic list of RunTargets.

    Expected axes:
        - datasets: list of dataset names
        - embedding_engines: list of embedding engine names
        - summarizers: list of summarizer values (None or str)

    Ordering: datasets (outer) -> embedding_engines -> summarizers (inner).
    """
    datasets = parameter_axes.get("datasets", [])
    engines = parameter_axes.get("embedding_engines", [])
    summarizers = parameter_axes.get("summarizers", [])

    targets: List[RunTarget] = []
    for dataset in datasets:
        for engine in engines:
            for summ in summarizers:
                targets.append(
                    RunTarget(
                        dataset=str(dataset),
                        embedding_engine=str(engine),
                        summarizer=normalize_summarizer(summ),
                        entry_max=entry_max,
                        n_restarts_suffix=n_restarts_suffix,
                    )
                )
    return targets


def targets_to_run_keys(targets: List[RunTarget]) -> List[str]:
    """Convert RunTargets to run_key strings."""
    return [
        build_run_key(
            t.dataset,
            t.embedding_engine,
            t.summarizer,
            t.entry_max,
            t.n_restarts_suffix,
        )
        for t in targets
    ]
