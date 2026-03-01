"""Canonical sweep result metric helpers.

Consolidates duplicated metric extraction logic previously scattered across
``panel_app/views/sweep_explorer.py``, ``scripts/common/sweep_utils.py``,
and several plotting scripts.
"""

from typing import Any, Dict, List, Optional

import numpy as np

METRICS = [
    "objective",
    "dispersion",
    "silhouette",
    "ari",
    "cosine_sim",
    "cosine_sim_norm",
]


# ---------------------------------------------------------------------------
# Primitive metric helpers
# ---------------------------------------------------------------------------

def dist_from_result(result_dict: Dict[str, Any]) -> Optional[np.ndarray]:
    """Compute or retrieve a cosine-distance matrix from a serialized result dict.

    Checks for a pre-computed ``"dist"`` key first; falls back to computing
    the cosine distance matrix from ``"Z"`` if present.
    """
    dist = result_dict.get("dist")
    if dist is not None:
        return np.asarray(dist)
    Z = result_dict.get("Z")
    if Z is None:
        return None
    Z = np.asarray(Z)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = Z / np.maximum(norms, 1e-12)
    return np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, 2.0)


def try_ari(
    gt: Optional[np.ndarray], labels: Optional[np.ndarray]
) -> Optional[float]:
    """Compute Adjusted Rand Index, returning ``None`` on any failure."""
    if gt is None or labels is None or len(labels) != len(gt):
        return None
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(gt, labels))
    except Exception:
        return None


def try_silhouette(
    dist: Optional[np.ndarray], labels: Optional[np.ndarray]
) -> Optional[float]:
    """Compute silhouette score from a precomputed distance matrix."""
    if dist is None or labels is None:
        return None
    try:
        from sklearn.metrics import silhouette_score

        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Row extractors -- flatten pickle results into list[dict]
# ---------------------------------------------------------------------------

def rows_from_50runs(item: dict) -> List[dict]:
    """Flatten one ``no_pca_50runs_*.pkl`` item into a list of row dicts.

    Each row represents a single (k, run_idx) observation with all computed
    metrics.
    """
    data = item["data"]
    result = data.get("result") or {}
    by_k = result.get("by_k") or {}
    meta = data.get("metadata") or {}
    dataset = data.get("dataset_name", "unknown")
    summarizer = str(meta.get("summarizer", "None"))
    engine = str(meta.get("embedding_engine", "?"))
    gt = data.get("ground_truth_labels")
    if gt is not None:
        gt = np.asarray(gt)
    n_samples = len(gt) if gt is not None else 0

    dist = dist_from_result(result)

    rows: List[dict] = []
    for k_str, entry in by_k.items():
        try:
            k = int(k_str)
        except ValueError:
            continue
        objectives = entry.get("objectives") or []
        labels_all = entry.get("labels_all") or []
        n_restarts = max(len(objectives), len(labels_all))
        if n_restarts == 0:
            continue

        for i in range(n_restarts):
            ob = objectives[i] if i < len(objectives) else None
            lab = (
                np.asarray(labels_all[i])
                if (labels_all and i < len(labels_all))
                else None
            )

            n = n_samples
            if n == 0 and lab is not None:
                n = len(lab)

            dispersion = (float(ob) / n) if (ob is not None and n > 0) else None
            cosine_sim = (1.0 - dispersion) if dispersion is not None else None
            cosine_sim_norm = (
                (cosine_sim + 1.0) / 2.0 if cosine_sim is not None else None
            )
            sil = try_silhouette(dist, lab)
            ari = try_ari(gt, lab)

            rows.append(
                {
                    "dataset": dataset,
                    "engine": engine,
                    "summarizer": summarizer,
                    "data_type": "50runs",
                    "k": k,
                    "run_idx": i,
                    "n_samples": n,
                    "objective": float(ob) if ob is not None else None,
                    "dispersion": dispersion,
                    "silhouette": sil,
                    "ari": ari,
                    "cosine_sim": cosine_sim,
                    "cosine_sim_norm": cosine_sim_norm,
                }
            )
    return rows


def rows_from_sweep(item: dict) -> List[dict]:
    """Flatten one ``experimental_sweep_*.pkl`` item into a list of row dicts.

    Each row represents a single k observation with the best-of-restarts
    labels and derived metrics.
    """
    data = item["data"]
    result = data.get("result") or {}
    by_k = result.get("by_k") or {}
    meta = data.get("metadata") or {}
    dataset = data.get("dataset_name", "unknown")
    summarizer = str(meta.get("summarizer", "None"))
    engine = str(meta.get("embedding_engine", "?"))
    gt = data.get("ground_truth_labels")
    if gt is not None:
        gt = np.asarray(gt)

    dist_arr = result.get("dist")
    if dist_arr is not None:
        dist_arr = np.asarray(dist_arr)
    else:
        dist_arr = dist_from_result(result)

    rows: List[dict] = []
    for k_str, entry in by_k.items():
        try:
            k = int(k_str)
        except ValueError:
            continue
        labels = entry.get("labels")
        if labels is not None:
            labels = np.asarray(labels)
        n = (
            len(labels)
            if labels is not None
            else (len(gt) if gt is not None else 0)
        )

        ob_raw = entry.get("objective")
        if isinstance(ob_raw, dict):
            ob = ob_raw.get("value")
        elif isinstance(ob_raw, (int, float)):
            ob = float(ob_raw)
        else:
            ob = None

        dispersion = (float(ob) / n) if (ob is not None and n > 0) else None
        cosine_sim = (1.0 - dispersion) if dispersion is not None else None
        cosine_sim_norm = (
            (cosine_sim + 1.0) / 2.0 if cosine_sim is not None else None
        )

        stab = entry.get("stability")
        sil = None
        if isinstance(stab, dict):
            sil = stab.get("silhouette", {}).get("mean")
        elif (
            dist_arr is not None
            and labels is not None
            and len(labels) == dist_arr.shape[0]
        ):
            sil = try_silhouette(dist_arr, labels)

        ari = try_ari(gt, labels)

        rows.append(
            {
                "dataset": dataset,
                "engine": engine,
                "summarizer": summarizer,
                "data_type": "sweep",
                "k": k,
                "run_idx": 0,
                "n_samples": n,
                "objective": float(ob) if ob is not None else None,
                "dispersion": dispersion,
                "silhouette": sil,
                "ari": ari,
                "cosine_sim": cosine_sim,
                "cosine_sim_norm": cosine_sim_norm,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-k metric extraction (used by sweep_utils.ingest_result_to_db)
# ---------------------------------------------------------------------------

def extract_by_k_metrics(
    result_dict: Dict[str, Any],
    ground_truth_labels: Optional[np.ndarray],
) -> Dict[int, Dict[str, List]]:
    """Extract per-k metric arrays from an in-memory serialized sweep result.

    Returns ``{k: {metric_name: [values...]}}`` suitable for DB ingestion.
    """
    from collections import defaultdict

    by_k_raw = result_dict.get("by_k") or {}
    gt = ground_truth_labels
    if gt is not None:
        gt = np.asarray(gt)
    n_samples = len(gt) if gt is not None else 0

    dist = dist_from_result(result_dict)

    by_k: Dict[int, Dict[str, List]] = defaultdict(
        lambda: {m: [] for m in METRICS}
    )

    for k_str, entry in by_k_raw.items():
        try:
            k = int(k_str)
        except (ValueError, TypeError):
            continue
        objectives = entry.get("objectives") or []
        labels_all = entry.get("labels_all") or []
        n_restarts = max(len(objectives), len(labels_all))
        if n_restarts == 0:
            continue

        for i in range(n_restarts):
            ob = objectives[i] if i < len(objectives) else None
            lab = (
                np.asarray(labels_all[i])
                if (labels_all and i < len(labels_all))
                else None
            )

            n = n_samples or (len(lab) if lab is not None else 0)
            dispersion = (float(ob) / n) if (ob is not None and n > 0) else None
            cosine_sim = (1.0 - dispersion) if dispersion is not None else None
            cosine_sim_norm = (
                (cosine_sim + 1.0) / 2.0 if cosine_sim is not None else None
            )
            sil = try_silhouette(dist, lab)
            ari = try_ari(gt, lab)

            bucket = by_k[k]
            bucket["objective"].append(float(ob) if ob is not None else None)
            bucket["dispersion"].append(dispersion)
            bucket["silhouette"].append(sil)
            bucket["ari"].append(ari)
            bucket["cosine_sim"].append(cosine_sim)
            bucket["cosine_sim_norm"].append(cosine_sim_norm)

    return dict(by_k)
