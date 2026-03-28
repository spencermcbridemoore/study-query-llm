"""
Shared sweep data loading for Sweep Explorer and Perspective tabs.

Keeps file vs database paths in one place so behavior does not drift.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from study_query_llm.experiments.result_metrics import (
    METRICS,
    rows_from_50runs as _rows_from_50runs,
    rows_from_sweep as _rows_from_sweep,
)
from study_query_llm.services.sweep_query_service import mcq_explorer_metric_columns
from study_query_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# Must match CATEGORICAL_DIMS in sweep_explorer (PKL row schema).
_CATEGORICAL_DIMS_FOR_PKL = ("dataset", "engine", "summarizer", "data_type")


def load_sweep_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan data_dir for both pkl patterns and flatten into a tidy DataFrame.

    Returns an empty DataFrame with the correct schema if nothing is found.
    """
    data_dir = Path(data_dir)
    all_rows: list[dict] = []

    # --- 50-runs files (original and bigrun_300 series) ---
    for pattern in ("no_pca_50runs_*.pkl", "bigrun_300_*.pkl"):
        for p in sorted(data_dir.glob(pattern)):
            try:
                if p.stat().st_size == 0:
                    continue
                with open(p, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                logger.warning("Skip %s: %s", p.name, e)
                continue
            if not isinstance(data, dict) or "result" not in data:
                continue
            try:
                all_rows.extend(_rows_from_50runs({"data": data}))
            except Exception as e:
                logger.warning("Error parsing %s: %s", p.name, e)

    # --- multi-embedding sweep files ---
    for p in sorted(data_dir.glob("experimental_sweep_*.pkl")):
        try:
            if p.stat().st_size == 0:
                continue
            with open(p, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning("Skip %s: %s", p.name, e)
            continue
        if not isinstance(data, dict):
            continue
        meta = data.get("metadata") or {}
        sweep = meta.get("sweep_config") or {}
        # Only load no-PCA multi-embedding sweep files (matching original script filter)
        if "embedding_engine" not in meta:
            continue
        try:
            all_rows.extend(_rows_from_sweep({"data": data}))
        except Exception as e:
            logger.warning("Error parsing %s: %s", p.name, e)

    if not all_rows:
        schema = {
            "dataset": pd.Series([], dtype="str"),
            "engine": pd.Series([], dtype="str"),
            "summarizer": pd.Series([], dtype="str"),
            "data_type": pd.Series([], dtype="str"),
            "k": pd.Series([], dtype="int64"),
            "run_idx": pd.Series([], dtype="int64"),
            "n_samples": pd.Series([], dtype="int64"),
            **{m: pd.Series([], dtype="float64") for m in METRICS},
        }
        return pd.DataFrame(schema)

    df = pd.DataFrame(all_rows)
    df["k"] = df["k"].astype(int)
    df["run_idx"] = df["run_idx"].astype(int)
    df["n_samples"] = df["n_samples"].fillna(0).astype(int)
    for cat in _CATEGORICAL_DIMS_FOR_PKL:
        if cat in df.columns:
            df[cat] = df[cat].astype(str)
    return df.sort_values(["dataset", "engine", "summarizer", "k", "run_idx"]).reset_index(
        drop=True
    )


def fetch_sweep_dataframe(
    *,
    is_mcq: bool,
    source_is_database: bool,
    data_dir: str,
    scope_id: Optional[int],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load sweep metrics from files or the database.

    Returns ``(dataframe, extra_pct_columns)``. For MCQ database loads, the second
    value lists dynamic ``pct_*`` columns not in the static explorer metric list;
    callers should merge these into metric dropdowns.
    """
    if not source_is_database:
        return load_sweep_data(data_dir), []

    from panel_app.helpers import get_db_connection
    from study_query_llm.db.raw_call_repository import RawCallRepository
    from study_query_llm.services.sweep_query_service import SweepQueryService

    static_mcq = set(mcq_explorer_metric_columns())
    db = get_db_connection()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepQueryService(repo)
        if is_mcq:
            df = svc.get_mcq_metrics_df(mcq_request_id=scope_id)
            extra_pct = sorted(
                c
                for c in df.columns
                if c.startswith("pct_") and c not in static_mcq
            )
            return df, extra_pct
        df = svc.get_sweep_metrics_df(clustering_sweep_id=scope_id)
        return df, []
