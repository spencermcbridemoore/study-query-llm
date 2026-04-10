"""
Sweep Query Service — reconstruct a flat sweep-metrics DataFrame from the DB.

This mirrors the shape produced by ``load_sweep_data()`` in
``panel_app/views/sweep_explorer.py`` but reads from Groups/GroupLinks
instead of pkl files.  Rows are one-per-restart-per-k, matching the
"Raw runs" layout.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text as sa_text, func
from sqlalchemy.orm import Session, aliased

from ..analysis.mcq_from_run import compliance_rates_from_probe
from ..db.models_v2 import Group, GroupLink
from ..db.raw_call_repository import RawCallRepository
from ..services.provenance_service import (
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
    GROUP_TYPE_MCQ_RUN,
    GROUP_TYPE_MCQ_SWEEP,
    GROUP_TYPE_MCQ_SWEEP_REQUEST,
)
from ..services.provenanced_run_service import ProvenancedRunService
from ..services.sweep_request_service import SweepRequestService
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "dataset", "engine", "summarizer", "data_type",
    "clustering_sweep_id",
    "primary_snapshot_id",
    "dataset_snapshot_ids",
    "k", "run_idx", "n_samples",
    "objective", "dispersion", "silhouette", "ari",
    "cosine_sim", "cosine_sim_norm",
]

_CLUSTERING_PARENT_TYPES = (
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
)
_CLUSTERING_PARENT_PREFERENCE = (
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
)

_MCQ_PARENT_TYPES = (
    GROUP_TYPE_MCQ_SWEEP_REQUEST,
    GROUP_TYPE_MCQ_SWEEP,
)
_MCQ_PARENT_PREFERENCE = (
    GROUP_TYPE_MCQ_SWEEP_REQUEST,
    GROUP_TYPE_MCQ_SWEEP,
)

_NO_PARENT_SCOPE_SENTINEL = ""


def _normalize_snapshot_ids(meta: Dict[str, Any]) -> List[int]:
    """Return sorted unique snapshot ids from run metadata."""
    raw_snapshot_ids = meta.get("dataset_snapshot_ids")
    if raw_snapshot_ids is None and meta.get("dataset_snapshot_id") is not None:
        raw_snapshot_ids = [meta.get("dataset_snapshot_id")]
    if raw_snapshot_ids is None:
        return []
    if not isinstance(raw_snapshot_ids, (list, tuple, set)):
        raw_snapshot_ids = [raw_snapshot_ids]
    snapshot_ids: List[int] = []
    for sid in raw_snapshot_ids:
        try:
            snapshot_ids.append(int(sid))
        except (TypeError, ValueError):
            continue
    return sorted(set(snapshot_ids))


def _parent_scope_id_by_child_run(
    session: Session,
    run_ids: List[int],
    *,
    allowed_parent_types: tuple[str, ...],
    preference_order: tuple[str, ...],
) -> Dict[int, int]:
    """
    Map each child run group id to a single parent scope id via ``contains`` links.

    When multiple parents match (e.g. both sweep and request), pick the type that
    appears earliest in *preference_order*; ties use ``min(parent_group_id)``.
    """
    if not run_ids:
        return {}
    type_rank = {t: i for i, t in enumerate(preference_order)}
    Parent = aliased(Group)
    rows = (
        session.query(
            GroupLink.child_group_id,
            GroupLink.parent_group_id,
            Parent.group_type,
        )
        .join(Parent, Parent.id == GroupLink.parent_group_id)
        .filter(
            GroupLink.child_group_id.in_(run_ids),
            GroupLink.link_type == "contains",
            Parent.group_type.in_(allowed_parent_types),
        )
        .all()
    )
    by_child: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for child_id, parent_id, gtype in rows:
        if gtype not in type_rank:
            continue
        by_child[child_id].append((parent_id, gtype))

    out: dict[int, int] = {}
    for child_id, candidates in by_child.items():
        best_rank = min(type_rank[gt] for _, gt in candidates)
        best = [(pid, gt) for pid, gt in candidates if type_rank[gt] == best_rank]
        out[child_id] = min(pid for pid, _ in best)
    return out


class SweepQueryService:
    """Read sweep experiment data back from Groups/GroupLinks in the DB."""

    def __init__(self, repository: RawCallRepository):
        self.repository = repository

    def list_clustering_sweeps(self) -> List[dict]:
        """
        Return all clustering_sweep groups as a list of summary dicts.

        Each dict has: id, name, created_at, algorithm, n_runs, parameter_axes.
        Suitable for populating a UI dropdown.
        """
        session = self.repository.session
        sweeps = (
            session.query(Group)
            .filter(
                Group.group_type == "clustering_sweep",
            )
            .order_by(Group.created_at.desc())
            .all()
        )
        if not sweeps:
            return []

        sweep_ids = [s.id for s in sweeps]

        # Bulk-count child links for all sweeps in one query
        count_rows = (
            session.query(
                GroupLink.parent_group_id,
                func.count(GroupLink.id).label("n_runs"),
            )
            .filter(
                GroupLink.parent_group_id.in_(sweep_ids),
                GroupLink.link_type == "contains",
            )
            .group_by(GroupLink.parent_group_id)
            .all()
        )
        counts = {row.parent_group_id: row.n_runs for row in count_rows}

        result = []
        for s in sweeps:
            meta = s.metadata_json or {}
            result.append({
                "id": s.id,
                "name": s.name,
                "created_at": s.created_at,
                "algorithm": meta.get("algorithm", "?"),
                "n_runs": counts.get(s.id, 0),
                "parameter_axes": meta.get("parameter_axes", {}),
            })
        return result

    def list_clustering_sweep_requests(
        self,
        status_filter: Optional[str] = None,
        include_fulfilled: bool = True,
    ) -> List[dict]:
        """
        Return clustering_sweep_request groups as list of summary dicts.

        Each dict has: id, name, request_status, expected_count, created_at.
        """
        svc = SweepRequestService(self.repository)
        return svc.list_requests(
            status=status_filter,
            include_fulfilled=include_fulfilled,
        )

    def get_request_progress_summary(self, request_id: int) -> Optional[dict]:
        """
        Return progress summary for a sweep request.

        Returns dict with expected_count, completed_count, missing_count,
        completed_run_keys, missing_run_keys (first 10), or None if not found.
        """
        svc = SweepRequestService(self.repository)
        req = svc.get_request(request_id)
        if not req:
            return None
        progress = svc.compute_progress(request_id)
        progress["request_name"] = req.get("name", "?")
        progress["request_status"] = req.get("request_status", "?")
        # Truncate long lists for summary
        if len(progress.get("missing_run_keys", [])) > 10:
            progress["missing_run_keys_preview"] = progress["missing_run_keys"][:10]
        return progress

    def get_unified_execution_runs(self, request_id: int) -> List[dict]:
        """
        Return unified execution-provenance rows for a request.

        Includes persisted provenanced_runs plus compatibility-mapped legacy
        method/analysis outputs when explicit rows do not yet exist.
        """
        service = ProvenancedRunService(self.repository)
        return service.list_unified_execution_view(request_id)

    def get_sweep_metrics_df(
        self,
        dataset: Optional[str] = None,
        engine: Optional[str] = None,
        summarizer: Optional[str] = None,
        clustering_sweep_id: Optional[int] = None,
        exclude_pre_fix: bool = False,
    ) -> pd.DataFrame:
        """
        Query run groups and their step children, expand metric arrays
        into individual rows, and return a tidy DataFrame.

        Optional filters narrow the result set before expansion.
        When ``clustering_sweep_id`` is provided only the clustering_run groups
        that are children of that sweep (via a "contains" GroupLink) are returned.

        When ``exclude_pre_fix`` is True, runs with
        ``metadata_json->>'centroid_fix_era' = 'pre_fix'`` are excluded.
        """
        session = self.repository.session

        if clustering_sweep_id is not None:
            # Filter to runs that are children of the given sweep
            child_links = (
                session.query(GroupLink)
                .filter(
                    GroupLink.parent_group_id == clustering_sweep_id,
                    GroupLink.link_type == "contains",
                )
                .all()
            )
            run_ids = [lnk.child_group_id for lnk in child_links]
            if not run_ids:
                return _empty_df()
            query = session.query(Group).filter(
                Group.group_type == "clustering_run",
                Group.id.in_(run_ids),
            )
        else:
            query = session.query(Group).filter(
                Group.group_type == "clustering_run",
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
        if exclude_pre_fix:
            query = query.filter(
                sa_text(
                    "(metadata_json->>'centroid_fix_era' IS NULL "
                    "OR metadata_json->>'centroid_fix_era' != 'pre_fix')"
                ),
            )

        runs = query.all()
        if not runs:
            return _empty_df()

        # Build run metadata lookup keyed by run ID
        run_meta: dict[int, dict] = {}
        for run in runs:
            meta = run.metadata_json or {}
            snapshot_ids = _normalize_snapshot_ids(meta)
            run_meta[run.id] = {
                "dataset": meta.get("dataset", "unknown"),
                "engine": meta.get("embedding_engine", "?"),
                "summarizer": meta.get("summarizer", "None"),
                "data_type": meta.get("data_type", "unknown"),
                "primary_snapshot_id": (
                    str(snapshot_ids[0]) if snapshot_ids else _NO_PARENT_SCOPE_SENTINEL
                ),
                "dataset_snapshot_ids": ",".join(str(sid) for sid in snapshot_ids),
            }

        run_ids = list(run_meta.keys())

        if clustering_sweep_id is not None:
            run_to_scope = {rid: str(clustering_sweep_id) for rid in run_ids}
        else:
            pmap = _parent_scope_id_by_child_run(
                session,
                run_ids,
                allowed_parent_types=_CLUSTERING_PARENT_TYPES,
                preference_order=_CLUSTERING_PARENT_PREFERENCE,
            )
            run_to_scope = {
                rid: str(pmap[rid]) if rid in pmap else _NO_PARENT_SCOPE_SENTINEL
                for rid in run_ids
            }

        # Bulk fetch all clustering_step links for every run in one query
        all_step_links = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id.in_(run_ids),
                GroupLink.link_type == "clustering_step",
            )
            .all()
        )

        # Map step_id -> (run_id, position) and collect step IDs
        step_id_to_run: dict[int, tuple[int, int]] = {
            lnk.child_group_id: (lnk.parent_group_id, lnk.position)
            for lnk in all_step_links
        }
        step_ids = list(step_id_to_run.keys())

        if not step_ids:
            return _empty_df()

        # Bulk fetch all step groups in one query — only metadata_json needed
        steps = (
            session.query(Group)
            .filter(
                Group.id.in_(step_ids),
            )
            .all()
        )

        # Expand each step's metric arrays into rows
        all_rows: list[dict] = []
        for step in steps:
            run_id, position = step_id_to_run[step.id]
            rmeta = run_meta[run_id]
            smeta = step.metadata_json or {}

            k = smeta.get("k", position)
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
                    "dataset": rmeta["dataset"],
                    "engine": rmeta["engine"],
                    "summarizer": rmeta["summarizer"],
                    "data_type": rmeta["data_type"],
                    "clustering_sweep_id": run_to_scope[run_id],
                    "primary_snapshot_id": rmeta["primary_snapshot_id"],
                    "dataset_snapshot_ids": rmeta["dataset_snapshot_ids"],
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
        for cat in (
            "dataset",
            "engine",
            "summarizer",
            "data_type",
            "clustering_sweep_id",
            "primary_snapshot_id",
            "dataset_snapshot_ids",
        ):
            df[cat] = df[cat].astype(str)
        return df.sort_values(
            [
                "clustering_sweep_id",
                "dataset",
                "engine",
                "summarizer",
                "k",
                "run_idx",
            ],
        ).reset_index(drop=True)

    def get_mcq_metrics_df(self, mcq_request_id: Optional[int] = None) -> pd.DataFrame:
        """
        Build a tidy DataFrame with one row per ``mcq_run`` group.

        When ``mcq_request_id`` is set, only runs linked to that request via
        ``GroupLink`` (``link_type='contains'``, parent = request) are included.
        When None, all ``mcq_run`` groups are returned (unbounded; use request
        filter in UI when possible).

        Columns include categoricals from run metadata, compliance metrics
        (via :func:`compliance_rates_from_probe`), ``chi_square_vs_uniform``,
        ``answer_count_total``, and per-label ``pct_{L}`` / ``count_{L}`` from
        ``pooled_distribution``.
        """
        session = self.repository.session

        if mcq_request_id is not None:
            child_links = (
                session.query(GroupLink)
                .filter(
                    GroupLink.parent_group_id == mcq_request_id,
                    GroupLink.link_type == "contains",
                )
                .all()
            )
            run_ids = list({lnk.child_group_id for lnk in child_links})
            if not run_ids:
                return _empty_mcq_df()
            runs = (
                session.query(Group)
                .filter(
                    Group.group_type == GROUP_TYPE_MCQ_RUN,
                    Group.id.in_(run_ids),
                )
                .all()
            )
        else:
            runs = (
                session.query(Group)
                .filter(
                    Group.group_type == GROUP_TYPE_MCQ_RUN,
                )
                .order_by(Group.id.asc())
                .all()
            )

        if not runs:
            return _empty_mcq_df()

        run_id_list = [r.id for r in runs]
        if mcq_request_id is not None:
            run_to_mcq_scope = {rid: str(mcq_request_id) for rid in run_id_list}
        else:
            pmap = _parent_scope_id_by_child_run(
                session,
                run_id_list,
                allowed_parent_types=_MCQ_PARENT_TYPES,
                preference_order=_MCQ_PARENT_PREFERENCE,
            )
            run_to_mcq_scope = {
                rid: str(pmap[rid]) if rid in pmap else _NO_PARENT_SCOPE_SENTINEL
                for rid in run_id_list
            }

        runs_sorted = sorted(runs, key=lambda g: (str((g.metadata_json or {}).get("run_key") or ""), g.id))
        rows: List[Dict[str, Any]] = []
        for i, run in enumerate(runs_sorted):
            rows.append(
                _mcq_row_from_group(
                    run,
                    run_idx=i,
                    parent_mcq_request_id=run_to_mcq_scope[run.id],
                )
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return _empty_mcq_df()

        df["k"] = df["k"].astype(int)
        df["run_idx"] = df["run_idx"].astype(int)
        df["n_samples"] = df["n_samples"].fillna(0).astype(int)
        for cat in _MCQ_CATEGORICAL_COLS:
            if cat in df.columns:
                df[cat] = df[cat].astype(str)
        return df.sort_values(
            ["deployment", "subject", "run_key"],
            kind="mergesort",
        ).reset_index(drop=True)

    def count_runs(self) -> int:
        """Return the total number of sweep run groups in the DB."""
        session = self.repository.session
        return (
            session.query(Group)
            .filter(
                Group.group_type == "clustering_run",
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
    str_cols = (
        "dataset",
        "engine",
        "summarizer",
        "data_type",
        "clustering_sweep_id",
        "primary_snapshot_id",
        "dataset_snapshot_ids",
    )
    return pd.DataFrame(
        {
            col: pd.Series([], dtype="float64" if col not in str_cols else "str")
            for col in EXPECTED_COLUMNS
        }
    )


_MCQ_CATEGORICAL_COLS = (
    "run_key",
    "mcq_request_id",
    "deployment",
    "subject",
    "level",
    "options_per_question",
    "questions_per_test",
    "label_style",
    "spread_correct_answer_uniformly",
    "samples_per_combo",
    "template_version",
)

_MCQ_STATIC_FLOAT_COLS = (
    "chi_square_vs_uniform",
    "answer_count_total",
    "format_compliance_rate",
    "question_count_compliance_rate",
    "answer_key_parse_rate",
)


def _mcq_row_from_group(
    run: Group,
    run_idx: int,
    *,
    parent_mcq_request_id: str = _NO_PARENT_SCOPE_SENTINEL,
) -> Dict[str, Any]:
    meta = dict(run.metadata_json or {})
    summary = dict(meta.get("result_summary") or {})
    probe_details: Dict[str, Any] = {
        "summary": summary,
        "call_errors": [],
        "parse_failures": [],
    }
    rates = compliance_rates_from_probe(probe_details)

    n_samples = int(
        summary.get("answer_count_total")
        or summary.get("samples_with_valid_answer_key")
        or 0
    )
    chi = summary.get("chi_square_vs_uniform")
    try:
        chi_f = float(chi)
    except (TypeError, ValueError):
        chi_f = float("nan")

    row: Dict[str, Any] = {
        "run_key": str(meta.get("run_key", "")),
        "mcq_request_id": str(parent_mcq_request_id),
        "deployment": str(meta.get("deployment", "") or summary.get("deployment", "") or ""),
        "subject": str(meta.get("subject", "") or summary.get("subject", "") or ""),
        "level": str(meta.get("level") or summary.get("level") or ""),
        "options_per_question": str(meta.get("options_per_question", "")),
        "questions_per_test": str(meta.get("questions_per_test", "")),
        "label_style": str(meta.get("label_style", "")),
        "spread_correct_answer_uniformly": str(meta.get("spread_correct_answer_uniformly", "")),
        "samples_per_combo": str(meta.get("samples_per_combo", "")),
        "template_version": str(meta.get("template_version", "")),
        "k": 1,
        "run_idx": int(run_idx),
        "n_samples": n_samples,
        "chi_square_vs_uniform": chi_f,
        "answer_count_total": float(summary.get("answer_count_total") or 0),
        **rates,
    }

    pooled = summary.get("pooled_distribution") or {}
    for lab, cell in pooled.items():
        if not lab:
            continue
        key = str(lab).upper()
        cell_d = cell or {}
        pct = cell_d.get("pct")
        cnt = cell_d.get("count")
        if isinstance(pct, (int, float)) and not (isinstance(pct, float) and math.isnan(float(pct))):
            row[f"pct_{key}"] = float(pct)
        else:
            row[f"pct_{key}"] = float("nan")
        try:
            row[f"count_{key}"] = int(cnt) if cnt is not None else 0
        except (TypeError, ValueError):
            row[f"count_{key}"] = 0

    return row


def _empty_mcq_df() -> pd.DataFrame:
    str_cols = list(_MCQ_CATEGORICAL_COLS)
    num_cols = ["k", "run_idx", "n_samples"] + list(_MCQ_STATIC_FLOAT_COLS)
    # Placeholder label columns so the Panel metric list can resolve when DB is empty
    for lab in ("A", "B", "C", "D", "E"):
        num_cols.append(f"pct_{lab}")
        num_cols.append(f"count_{lab}")
    cols: Dict[str, pd.Series] = {
        c: pd.Series([], dtype="str") for c in str_cols
    }
    cols.update({c: pd.Series([], dtype="float64") for c in num_cols})
    return pd.DataFrame(cols)


# Public list of MCQ metric column names for Panel (excludes ids / raw counts if desired)
def mcq_explorer_metric_columns() -> List[str]:
    """Scalar/plot Y-axis columns for MCQ sweep explorer (stable ordering)."""
    base = [
        "format_compliance_rate",
        "question_count_compliance_rate",
        "answer_key_parse_rate",
        "chi_square_vs_uniform",
        "answer_count_total",
    ]
    for lab in ("A", "B", "C", "D", "E"):
        base.append(f"pct_{lab}")
    return base
