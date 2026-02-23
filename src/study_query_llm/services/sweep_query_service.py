"""
Sweep Query Service — reconstruct a flat sweep-metrics DataFrame from the DB.

This mirrors the shape produced by ``load_sweep_data()`` in
``panel_app/views/sweep_explorer.py`` but reads from Groups/GroupLinks
instead of pkl files.  Rows are one-per-restart-per-k, matching the
"Raw runs" layout.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import text as sa_text

from ..db.models_v2 import Group, GroupLink
from ..db.raw_call_repository import RawCallRepository
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "dataset", "engine", "summarizer", "data_type",
    "k", "run_idx", "n_samples",
    "objective", "dispersion", "silhouette", "ari",
    "cosine_sim", "cosine_sim_norm",
]


class SweepQueryService:
    """Read sweep experiment data back from Groups/GroupLinks in the DB."""

    def __init__(self, repository: RawCallRepository):
        self.repository = repository

    def get_sweep_metrics_df(
        self,
        dataset: Optional[str] = None,
        engine: Optional[str] = None,
        summarizer: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query run groups and their step children, expand metric arrays
        into individual rows, and return a tidy DataFrame.

        Optional filters narrow the result set before expansion.
        """
        session = self.repository.session

        query = session.query(Group).filter(
            Group.group_type == "run",
            sa_text("metadata_json->>'algorithm' LIKE :alg"),
        ).params(alg="cosine_kllmeans%")
        if dataset:
            query = query.filter(
                sa_text("metadata_json->>'dataset' = :ds"),
            ).params(ds=dataset)
        if engine:
            query = query.filter(
                sa_text("metadata_json->>'embedding_engine' = :eng"),
            ).params(eng=engine)
        if summarizer:
            query = query.filter(
                sa_text("metadata_json->>'summarizer' = :summ"),
            ).params(summ=summarizer)

        runs = query.all()
        if not runs:
            return _empty_df()

        all_rows: list[dict] = []

        for run in runs:
            meta = run.metadata_json or {}
            ds = meta.get("dataset", "unknown")
            eng = meta.get("embedding_engine", "?")
            summ = meta.get("summarizer", "None")
            data_type = meta.get("data_type", "unknown")

            step_links = (
                session.query(GroupLink)
                .filter(
                    GroupLink.parent_group_id == run.id,
                    GroupLink.link_type == "step",
                )
                .order_by(GroupLink.position)
                .all()
            )

            for link in step_links:
                step = session.query(Group).get(link.child_group_id)
                if step is None:
                    continue
                smeta = step.metadata_json or {}
                k = smeta.get("k", link.position)
                n_samples = smeta.get("n_samples", 0)

                objectives = smeta.get("objectives", [])
                dispersions = smeta.get("dispersions", [])
                aris = smeta.get("aris", [])
                silhouettes = smeta.get("silhouettes", [])
                cosine_sims = smeta.get("cosine_sims", [])
                cosine_sim_norms = smeta.get("cosine_sim_norms", [])

                n_restarts = max(
                    len(objectives), len(dispersions),
                    len(aris), len(silhouettes),
                    len(cosine_sims), len(cosine_sim_norms),
                    1,
                )

                for i in range(n_restarts):
                    all_rows.append({
                        "dataset": ds,
                        "engine": eng,
                        "summarizer": summ,
                        "data_type": data_type,
                        "k": k,
                        "run_idx": i,
                        "n_samples": n_samples,
                        "objective": _safe_idx(objectives, i),
                        "dispersion": _safe_idx(dispersions, i),
                        "silhouette": _safe_idx(silhouettes, i),
                        "ari": _safe_idx(aris, i),
                        "cosine_sim": _safe_idx(cosine_sims, i),
                        "cosine_sim_norm": _safe_idx(cosine_sim_norms, i),
                    })

        if not all_rows:
            return _empty_df()

        df = pd.DataFrame(all_rows)
        df["k"] = df["k"].astype(int)
        df["run_idx"] = df["run_idx"].astype(int)
        df["n_samples"] = df["n_samples"].fillna(0).astype(int)
        for cat in ("dataset", "engine", "summarizer", "data_type"):
            df[cat] = df[cat].astype(str)
        return df.sort_values(
            ["dataset", "engine", "summarizer", "k", "run_idx"],
        ).reset_index(drop=True)

    def count_runs(self) -> int:
        """Return the total number of sweep run groups in the DB."""
        session = self.repository.session
        return (
            session.query(Group)
            .filter(
                Group.group_type == "run",
                sa_text("metadata_json->>'algorithm' LIKE :alg"),
            )
            .params(alg="cosine_kllmeans%")
            .count()
        )


def _safe_idx(lst: list, idx: int):
    """Return lst[idx] if in range, else None."""
    if idx < len(lst):
        v = lst[idx]
        return float(v) if v is not None else None
    return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series([], dtype="float64" if col not in (
        "dataset", "engine", "summarizer", "data_type",
    ) else "str") for col in EXPECTED_COLUMNS})
