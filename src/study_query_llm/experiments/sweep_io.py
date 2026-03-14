"""Sweep result serialization and file I/O."""

import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def get_output_dir(output_dir: Optional[str] = None) -> Path:
    """Resolve the sweep output directory.

    Resolution order:
    1. Explicit ``output_dir`` argument
    2. ``SWEEP_OUTPUT_DIR`` environment variable
    3. ``<repo_root>/experimental_results`` (original default)

    The directory is created if it does not exist.
    """
    if output_dir:
        p = Path(output_dir)
    elif env := os.environ.get("SWEEP_OUTPUT_DIR"):
        p = Path(env)
    else:
        p = Path(__file__).resolve().parent.parent.parent.parent / "experimental_results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def serialize_sweep_result(result: Any) -> Dict[str, Any]:
    """Convert a single ``SweepResult`` to a JSON-safe dictionary."""
    data: Dict[str, Any] = {"pca": result.pca, "by_k": {}}

    if result.Z is not None:
        data["Z"] = result.Z.tolist()
    if result.Z_norm is not None:
        data["Z_norm"] = result.Z_norm.tolist()
    if result.dist is not None:
        data["dist"] = result.dist.tolist()

    for k, k_data in result.by_k.items():
        labels_raw = k_data.get("labels", [])
        labels_all_raw = k_data.get("labels_all")
        data["by_k"][k] = {
            "representatives": k_data.get("representatives", []),
            "labels": (
                labels_raw.tolist()
                if hasattr(labels_raw, "tolist")
                else labels_raw
            ),
            "labels_all": (
                [
                    la.tolist() if hasattr(la, "tolist") else la
                    for la in labels_all_raw
                ]
                if labels_all_raw is not None
                else None
            ),
            "objective": k_data.get("objective", {}),
            "objectives": k_data.get("objectives", []),
            "n_iter": k_data.get("n_iter", 0),
            "timing": k_data.get("timing", {}),
            "stability": k_data.get("stability"),
        }

    return data


def save_single_sweep_result(
    result: Any,
    output_file: str,
    ground_truth_labels: Optional[np.ndarray] = None,
    dataset_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a single ``SweepResult`` with metadata to a pickle file.

    .. deprecated::
        Use ``ingest_result_to_db`` for artifact-backed persistence (blob-first pipeline).
        This function is for transitional backfill only.
    """
    warnings.warn(
        "save_single_sweep_result is deprecated; use ingest_result_to_db for "
        "artifact-backed persistence (blob-first pipeline).",
        DeprecationWarning,
        stacklevel=2,
    )
    final = {
        "result": serialize_sweep_result(result),
        "ground_truth_labels": (
            ground_truth_labels.tolist()
            if ground_truth_labels is not None
            else None
        ),
        "dataset_name": dataset_name,
        "metadata": metadata or {},
    }
    with open(output_file, "wb") as f:
        pickle.dump(final, f)
    return output_file


def save_batch_sweep_results(
    all_results: Dict[str, Any],
    output_file: Optional[str] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    dataset_name: str = "unknown",
    output_dir: Optional[str] = None,
) -> str:
    """Save multiple ``SweepResult`` objects keyed by summarizer name.

    .. deprecated::
        Use ``ingest_result_to_db`` for each result for artifact-backed persistence.
        This function is for transitional backfill only.
    """
    warnings.warn(
        "save_batch_sweep_results is deprecated; use ingest_result_to_db for "
        "each result (artifact-backed persistence).",
        DeprecationWarning,
        stacklevel=2,
    )
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = get_output_dir(output_dir)
        output_file = str(out / f"pca_kllmeans_sweep_results_{ts}.pkl")

    serialized = {
        name: serialize_sweep_result(res)
        for name, res in all_results.items()
    }

    final = {
        "summarizers": serialized,
        "ground_truth_labels": (
            ground_truth_labels.tolist()
            if ground_truth_labels is not None
            else None
        ),
        "dataset_name": dataset_name,
    }
    with open(output_file, "wb") as f:
        pickle.dump(final, f)
    return output_file
