"""
Sweep Explorer tab — interactive Plotly-based viewer for pkl experiment results.

Loads no_pca_50runs_*.pkl and experimental_sweep_*.pkl files, flattens them into
a tidy DataFrame, and renders configurable faceted line/box plots with full
dimensional control (rows, cols, color, x, y, aggregation, filters).
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

from study_query_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from study_query_llm.experiments.result_metrics import (
    METRICS,
    dist_from_result as _dist_from_z,
    try_ari as _try_ari,
    try_silhouette as _try_silhouette,
    rows_from_50runs as _rows_from_50runs,
    rows_from_sweep as _rows_from_sweep,
)
from study_query_llm.experiments.sweep_request_types import SWEEP_TYPE_MCQ
from study_query_llm.services.sweep_query_service import mcq_explorer_metric_columns

# Categorical dimensions available for faceting / color / filters
CATEGORICAL_DIMS = ["dataset", "engine", "summarizer", "data_type"]

# Numeric dimensions usable on X-axis
X_AXIS_OPTIONS = ["k", "run_idx", "n_samples"]

AGGREGATION_OPTIONS = ["Raw runs", "Mean", "Mean ± stdev", "Last k only"]



# _rows_from_50runs and _rows_from_sweep are imported from
# study_query_llm.experiments.result_metrics above.


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
    for cat in CATEGORICAL_DIMS:
        if cat in df.columns:
            df[cat] = df[cat].astype(str)
    return df.sort_values(["dataset", "engine", "summarizer", "k", "run_idx"]).reset_index(drop=True)


# ===========================================================================
# Aggregation
# ===========================================================================

def _group_cols_for_aggregate(
    df: pd.DataFrame,
    group_dims: list[str],
    x_axis: str | None,
) -> list[str]:
    """Dims to group by when aggregating; includes x_axis if present (for line plots)."""
    tail = ["k", "n_samples"]
    parts = list(group_dims)
    if x_axis and x_axis in df.columns and x_axis not in parts:
        parts.append(x_axis)
    parts.extend(tail)
    seen: set[str] = set()
    out: list[str] = []
    for c in parts:
        if c in df.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def aggregate_df(
    df: pd.DataFrame,
    y_metric: str,
    agg_mode: str,
    group_dims: list[str] | None = None,
    x_axis: str | None = None,
) -> pd.DataFrame:
    """
    Transform the flat DataFrame according to the aggregation mode.

    Args:
        group_dims: Which categorical dimensions to group by. Include dims with
            visual roles (Grid rows/cols/Color) and **Mean group** so that
            "Free" / "Filter" dims don't split aggregated series. Defaults to
            all CATEGORICAL_DIMS for backward compatibility.

    Returns a DataFrame ready for plotting, adding `y_err` column for Mean±stdev.
    """
    if df.empty:
        return df

    if group_dims is None:
        group_dims = CATEGORICAL_DIMS

    if agg_mode == "Raw runs":
        out = df.copy()
        out["y_err"] = np.nan
        return out

    group_cols = _group_cols_for_aggregate(df, group_dims, x_axis)

    if agg_mode == "Last k only":
        cat_cols = [c for c in group_dims if c in df.columns]
        idx = df.groupby(cat_cols)["k"].transform("max") if cat_cols else df["k"].max()
        filtered = df[df["k"] == idx].copy()
        out = filtered.groupby(group_cols, as_index=False)[y_metric].mean() if group_cols else filtered
        out["y_err"] = np.nan
        return out

    if agg_mode in ("Mean", "Mean ± stdev"):
        if not group_cols:
            # No facet/color dims — aggregate; keep k and/or x_axis progression
            gb_cols = [c for c in ["k"] if c in df.columns]
            if x_axis and x_axis in df.columns and x_axis not in gb_cols:
                gb_cols.append(x_axis)
            if not gb_cols:
                m = float(df[y_metric].mean())
                std_v = (
                    float(df[y_metric].std())
                    if agg_mode == "Mean ± stdev" and len(df) > 1
                    else np.nan
                )
                agg = pd.DataFrame({y_metric: [m]})
                if agg_mode == "Mean ± stdev":
                    agg["y_err"] = [0.0 if std_v != std_v else std_v]
                else:
                    agg["y_err"] = [np.nan]
                return agg
            agg = df.groupby(gb_cols, as_index=False).agg(
                y_mean=(y_metric, "mean"),
                y_std=(y_metric, "std"),
            )
            agg = agg.rename(columns={"y_mean": y_metric})
        else:
            agg = df.groupby(group_cols, as_index=False).agg(
                y_mean=(y_metric, "mean"),
                y_std=(y_metric, "std"),
            )
            agg = agg.rename(columns={"y_mean": y_metric})
        if agg_mode == "Mean ± stdev":
            agg["y_err"] = agg["y_std"].fillna(0)
        else:
            agg["y_err"] = np.nan
        agg = agg.drop(columns=["y_std"], errors="ignore")
        return agg

    return df.copy()


# ===========================================================================
# Figure builder
# ===========================================================================

def build_figure(
    df: pd.DataFrame,
    y_metric: str,
    x_axis: str,
    row_dim: Optional[str],
    col_dim: Optional[str],
    color_dim: Optional[str],
    agg_mode: str,
    plot_height: int = 600,
    *,
    connect_lines: bool = True,
) -> go.Figure:
    """
    Build a Plotly figure from the prepared DataFrame.

    Uses px.line for standard modes, px.scatter for Last k only,
    and adds error bands for Mean±stdev.

    If connect_lines is False, scatter traces use markers only; if True, lines
    connect points (scatter sorted by x for stable polylines).
    """
    if df.empty or y_metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data loaded. Set data directory and click Load.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(height=plot_height)
        return fig

    # Remove rows where the metric is NaN
    plot_df = df.dropna(subset=[y_metric]).copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data for metric '{y_metric}'.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(height=plot_height)
        return fig

    # Resolve None sentinel values from selects
    facet_row = row_dim if row_dim and row_dim != "None" else None
    facet_col = col_dim if col_dim and col_dim != "None" else None
    color = color_dim if color_dim and color_dim != "None" else None

    # Make all facet/color dims strings to avoid Plotly type issues
    for col in [c for c in [facet_row, facet_col, color] if c]:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].astype(str)

    xcol = x_axis if x_axis in plot_df.columns else "k"
    if agg_mode == "Last k only" and connect_lines:
        sort_cols = [c for c in [xcol, color, facet_row, facet_col] if c and c in plot_df.columns]
        if sort_cols:
            plot_df = plot_df.sort_values(sort_cols, kind="mergesort")

    common_kwargs = dict(
        x=xcol,
        y=y_metric,
        facet_row=facet_row,
        facet_col=facet_col,
        color=color,
        markers=True,
    )

    if agg_mode == "Last k only":
        fig = px.scatter(
            plot_df,
            **{k: v for k, v in common_kwargs.items() if k != "markers"},
            title=f"{y_metric} — last k per series",
        )
    elif agg_mode == "Mean ± stdev" and "y_err" in plot_df.columns:
        fig = px.line(
            plot_df,
            **common_kwargs,
            error_y="y_err",
            title=f"{y_metric} — mean ± stdev",
        )
    else:
        label = {"Raw runs": "all restarts", "Mean": "mean"}.get(agg_mode, agg_mode)
        fig = px.line(
            plot_df,
            **common_kwargs,
            title=f"{y_metric} — {label}",
        )

    # Layout polish
    fig.update_layout(
        height=plot_height,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text=y_metric)

    if connect_lines:
        fig.update_traces(mode="lines+markers", selector=dict(type="scatter"))
    else:
        fig.update_traces(mode="markers", selector=dict(type="scatter"))

    return fig


MAX_BAR_METRICS = 12


def aggregated_metrics_long(
    filtered: pd.DataFrame,
    metrics: list[str],
    agg_mode: str,
    group_dims: list[str],
    x_axis: str,
    facet_row: Optional[str],
    facet_col: Optional[str],
) -> pd.DataFrame:
    """One row per (facet cells, x bucket, metric) for grouped bar charts."""
    rows: list[dict] = []
    facet_keys = [c for c in [facet_row, facet_col] if c and c in filtered.columns]

    for m in metrics:
        if m not in filtered.columns:
            continue
        agg = aggregate_df(
            filtered, m, agg_mode, group_dims=group_dims, x_axis=x_axis
        )
        if agg.empty:
            continue
        xcol = x_axis if x_axis in agg.columns else ("k" if "k" in agg.columns else None)
        use = agg.dropna(subset=[m])
        if use.empty:
            continue
        for _, r in use.iterrows():
            extras = [
                d
                for d in group_dims
                if d in r.index
                and d != facet_row
                and d != facet_col
                and d != xcol
            ]
            if xcol is None:
                x_cat = (
                    " | ".join(str(r[d]) for d in extras) if extras else "(all)"
                )
            elif extras:
                x_cat = str(r[xcol]) + " | " + " | ".join(str(r[d]) for d in extras)
            else:
                x_cat = str(r[xcol])
            rec: dict = {
                "x_cat": x_cat,
                "metric": m,
                "value": float(r[m]),
            }
            if "y_err" in use.columns:
                ev = r["y_err"]
                rec["y_err"] = float(ev) if pd.notna(ev) else np.nan
            else:
                rec["y_err"] = np.nan
            for fk in facet_keys:
                if fk in r.index:
                    rec[fk] = r[fk]
            rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["x_cat", "metric", "value", "y_err", *facet_keys])

    long_df = pd.DataFrame(rows)
    for fk in facet_keys:
        if fk in long_df.columns:
            long_df[fk] = long_df[fk].astype(str)
    return long_df


def build_figure_bars(
    long_df: pd.DataFrame,
    x_axis: str,
    row_dim: Optional[str],
    col_dim: Optional[str],
    agg_mode: str,
    plot_height: int = 600,
) -> go.Figure:
    """Grouped bars: x = category, color = metric name."""
    if long_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No bar data (check metrics and filters).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(height=plot_height)
        return fig

    plot_df = long_df.copy()
    facet_row = row_dim if row_dim and row_dim != "None" else None
    facet_col = col_dim if col_dim and col_dim != "None" else None

    use_err = "y_err" in plot_df.columns and plot_df["y_err"].notna().any()
    if use_err:
        fig = px.bar(
            plot_df,
            x="x_cat",
            y="value",
            color="metric",
            facet_row=facet_row,
            facet_col=facet_col,
            barmode="group",
            title=f"Grouped metrics — {agg_mode}",
            error_y="y_err",
        )
    else:
        fig = px.bar(
            plot_df,
            x="x_cat",
            y="value",
            color="metric",
            facet_row=facet_row,
            facet_col=facet_col,
            barmode="group",
            title=f"Grouped metrics — {agg_mode}",
        )

    fig.update_layout(
        height=plot_height,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text="value")
    return fig


# Roles available for each categorical dimension
DIM_ROLES = ["Free", "Filter", "Mean group", "Grid rows", "Grid cols", "Color"]

# Exclusive roles — only one dimension may hold each at a time
EXCLUSIVE_ROLES = {"Grid rows", "Grid cols", "Color"}

# Roles that show a value filter MultiSelect
ROLES_WITH_VALUE_FILTER = {"Filter", "Mean group", "Grid rows", "Grid cols", "Color"}

# Derived n_samples bin dimension name
BIN_DIM = "n_samples_bin"

# All dimensions available in the role UI (categorical + derived bin)
ALL_DIMS = CATEGORICAL_DIMS + [BIN_DIM]

# Default role for each dimension
DIM_DEFAULT_ROLES = {
    "dataset": "Grid cols",
    "engine": "Color",
    "summarizer": "Grid rows",
    "data_type": "Filter",
    BIN_DIM: "Free",
}

MCQ_CATEGORICAL_DIMS = (
    "deployment",
    "subject",
    "level",
    "options_per_question",
    "spread_correct_answer_uniformly",
    "label_style",
    "run_key",
)

MCQ_DIM_DEFAULT_ROLES = {
    "deployment": "Color",
    "subject": "Grid rows",
    "level": "Grid cols",
    "options_per_question": "Filter",
    "spread_correct_answer_uniformly": "Free",
    "label_style": "Free",
    "run_key": "Free",
    BIN_DIM: "Free",
}

MCQ_X_AXIS_OPTIONS = [
    "subject",
    "deployment",
    "run_key",
    "run_idx",
    "n_samples",
    "answer_count_total",
    "options_per_question",
    "level",
]

MCQ_METRICS = list(mcq_explorer_metric_columns())

_DIM_LABELS = {BIN_DIM: "sample size bin"}


@dataclass(frozen=True)
class _ExplorerProfile:
    categorical_dims: tuple[str, ...]
    metrics: tuple[str, ...]
    x_axis_options: tuple[str, ...]
    default_y: str
    default_x: str


_CLUSTERING_PROFILE = _ExplorerProfile(
    categorical_dims=tuple(CATEGORICAL_DIMS),
    metrics=tuple(METRICS),
    x_axis_options=tuple(X_AXIS_OPTIONS),
    default_y="ari",
    default_x="k",
)

_MCQ_PROFILE = _ExplorerProfile(
    categorical_dims=MCQ_CATEGORICAL_DIMS,
    metrics=tuple(MCQ_METRICS),
    x_axis_options=tuple(MCQ_X_AXIS_OPTIONS),
    default_y="answer_key_parse_rate",
    default_x="subject",
)


def _build_profile_dim_widgets(
    categorical_dims: tuple[str, ...],
    dim_default_roles: dict[str, str],
    *,
    include_bin_dim: bool,
) -> tuple[dict[str, pn.widgets.Select], dict[str, pn.widgets.MultiSelect], list]:
    """Build role MultiSelect / Select widgets for one profile; optionally add BIN_DIM."""
    role_sels: dict[str, pn.widgets.Select] = {}
    val_filters: dict[str, pn.widgets.MultiSelect] = {}
    blocks: list = []
    dims = list(categorical_dims) + ([BIN_DIM] if include_bin_dim else [])
    for dim in dims:
        default_role = dim_default_roles.get(dim, "Free")
        role_sel = pn.widgets.Select(
            name=f"{dim} role",
            options=DIM_ROLES,
            value=default_role,
            width=160,
        )
        val_filter = pn.widgets.MultiSelect(
            name="values",
            options=[],
            value=[],
            height=100,
            width=175,
            visible=(default_role in ROLES_WITH_VALUE_FILTER),
        )
        role_sels[dim] = role_sel
        val_filters[dim] = val_filter
        label = _DIM_LABELS.get(dim, dim)
        blocks.append(
            pn.Column(
                pn.pane.Markdown(f"**{label}**"),
                role_sel,
                val_filter,
                width=190,
                margin=(0, 8, 0, 0),
            )
        )
    return role_sels, val_filters, blocks


# ===========================================================================
# n_samples binning helpers
# ===========================================================================

def parse_bin_edges(text: str) -> list[int]:
    """Parse comma-separated bin edges into a sorted, deduplicated list of ints."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    edges: list[int] = []
    for p in parts:
        try:
            edges.append(int(p))
        except ValueError:
            continue
    return sorted(set(edges))


def assign_n_samples_bin(series: pd.Series, edges: list[int]) -> pd.Series:
    """Assign each value in *series* to a human-readable bin label.

    Given edges ``[100, 280, 400]`` the bins are:
    ``"\u2264100"``, ``"101\u2013280"``, ``"281\u2013400"``, ``"401+"``.
    """
    if not edges:
        return pd.Series("\u2014", index=series.index)

    def _label(val: int) -> str:
        for i, edge in enumerate(edges):
            if val <= edge:
                lo = (edges[i - 1] + 1) if i > 0 else None
                return f"\u2264{edge}" if lo is None else f"{lo}\u2013{edge}"
        return f"{edges[-1] + 1}+"

    return series.apply(_label).astype(str)


# ===========================================================================
# Panel UI
# ===========================================================================

def create_sweep_explorer_ui() -> pn.viewable.Viewable:
    """Create the interactive Sweep Explorer tab with per-dimension role selectors."""

    # --- State ---
    _state: dict = {
        "df": pd.DataFrame(),
        "loaded": False,
        "load_summary": "_No data loaded._",
    }

    # --- Sweep type (clustering vs MCQ) ---
    sweep_type_sel = pn.widgets.RadioButtonGroup(
        name="Sweep type",
        options=["Clustering", "MCQ"],
        value="Clustering",
        button_type="default",
    )

    # --- Widgets: source toggle ---
    source_toggle = pn.widgets.RadioButtonGroup(
        name="Source",
        options=["Files", "Database"],
        value="Files",
        button_type="default",
    )

    # --- Widgets: data loading ---
    data_dir_input = pn.widgets.TextInput(
        name="Data directory",
        value=str(REPO_ROOT / "experimental_results"),
        width=420,
        placeholder="Path to folder containing .pkl files",
    )

    # --- Widgets: clustering sweep selector (Database path only) ---
    sweep_select = pn.widgets.Select(
        name="Clustering Sweep",
        options={"All": None},
        value=None,
        width=320,
        visible=False,
    )

    load_button = pn.widgets.Button(
        name="Load / Reload",
        button_type="primary",
        width=130,
    )
    load_status = pn.pane.Markdown("_No data loaded._", sizing_mode="stretch_width")

    # --- Widgets: axes ---
    y_axis_sel = pn.widgets.Select(
        name="Y axis (metric)",
        options=METRICS,
        value="ari",
        width=200,
    )
    x_axis_sel = pn.widgets.Select(
        name="X axis",
        options=X_AXIS_OPTIONS,
        value="k",
        width=150,
    )

    plot_type_sel = pn.widgets.RadioButtonGroup(
        name="Plot type",
        options=["Line plot", "Grouped metrics (bars)"],
        value="Line plot",
        button_type="default",
    )

    metrics_multi = pn.widgets.MultiSelect(
        name="Metrics (bars)",
        options=list(METRICS),
        value=["ari"],
        height=120,
        width=300,
        visible=False,
    )

    connect_lines_chk = pn.widgets.Checkbox(
        name="Connect points (show lines)",
        value=True,
    )

    # --- Widgets: aggregation ---
    agg_radio = pn.widgets.RadioButtonGroup(
        name="Aggregation",
        options=AGGREGATION_OPTIONS,
        value="Mean",
        button_type="default",
    )

    # --- Widgets: height ---
    height_slider = pn.widgets.IntSlider(
        name="Plot height (px)",
        start=300,
        end=2000,
        step=50,
        value=700,
        width=250,
    )

    # --- Widgets: k range ---
    k_slider = pn.widgets.RangeSlider(
        name="k range",
        start=2,
        end=20,
        value=(2, 20),
        step=1,
        width=300,
    )
    k_range_section = pn.Column(
        pn.pane.Markdown("### k Range"),
        k_slider,
        visible=True,
    )

    # --- Widgets: sample size binning ---
    bin_edges_input = pn.widgets.TextInput(
        name="Bin edges (comma-separated)",
        value="",
        placeholder="e.g. 100, 280, 400",
        width=300,
    )

    # --- Per-dimension role widgets (clustering row, MCQ row, shared sample-size bin) ---
    role_sels_c, val_filters_c, blocks_c = _build_profile_dim_widgets(
        tuple(CATEGORICAL_DIMS), DIM_DEFAULT_ROLES, include_bin_dim=False
    )
    role_sels_m, val_filters_m, blocks_m = _build_profile_dim_widgets(
        MCQ_CATEGORICAL_DIMS, MCQ_DIM_DEFAULT_ROLES, include_bin_dim=False
    )
    role_sels_bin, val_filters_bin, blocks_bin = _build_profile_dim_widgets(
        (), {BIN_DIM: "Free"}, include_bin_dim=True
    )

    role_sels_c.update(role_sels_bin)
    val_filters_c.update(val_filters_bin)
    role_sels_m.update(role_sels_bin)
    val_filters_m.update(val_filters_bin)

    dim_row_clustering = pn.Row(*blocks_c, sizing_mode="stretch_width")
    dim_row_mcq = pn.Row(*blocks_m, sizing_mode="stretch_width")
    dim_row_bin = pn.Row(*blocks_bin, sizing_mode="stretch_width")
    dim_row_mcq.visible = False

    def _is_mcq() -> bool:
        return sweep_type_sel.value == "MCQ"

    def _active_profile() -> _ExplorerProfile:
        return _MCQ_PROFILE if _is_mcq() else _CLUSTERING_PROFILE

    def _active_role_sels() -> dict[str, pn.widgets.Select]:
        return role_sels_m if _is_mcq() else role_sels_c

    def _active_val_filters() -> dict[str, pn.widgets.MultiSelect]:
        return val_filters_m if _is_mcq() else val_filters_c

    def _profile_dim_order() -> list[str]:
        p = _active_profile()
        return list(p.categorical_dims) + [BIN_DIM]

    # --- Plot pane ---
    plot_pane = pn.pane.Plotly(
        build_figure(
            pd.DataFrame(),
            "ari",
            "k",
            "summarizer",
            "dataset",
            "engine",
            "Mean",
            connect_lines=True,
        ),
        sizing_mode="stretch_width",
        height=700,
    )

    # --- Conflict check ---
    def _check_role_conflicts() -> str | None:
        """Return a warning string if any exclusive role is assigned to 2+ dims."""
        role_counts: dict[str, list[str]] = {}
        rs = _active_role_sels()
        for dim in _profile_dim_order():
            if dim not in rs:
                continue
            role = rs[dim].value
            if role in EXCLUSIVE_ROLES:
                role_counts.setdefault(role, []).append(dim)
        conflicts = [
            f"'{role}' assigned to: {', '.join(dims)}"
            for role, dims in role_counts.items()
            if len(dims) > 1
        ]
        return "; ".join(conflicts) if conflicts else None

    # --- Plot refresh logic ---
    def refresh_plot(event=None):
        df = _state.get("df", pd.DataFrame())
        base = _state.get("load_summary", "_No data loaded._")
        if df.empty:
            return

        # Compute n_samples_bin if bin edges are provided
        edges = parse_bin_edges(bin_edges_input.value)
        if edges and "n_samples" in df.columns:
            df = df.copy()
            df[BIN_DIM] = assign_n_samples_bin(df["n_samples"], edges)
        else:
            df = df.copy()
            if BIN_DIM in df.columns:
                df = df.drop(columns=[BIN_DIM])

        # Conflict guard
        conflict = _check_role_conflicts()
        if conflict:
            load_status.object = f"**Role conflict** — {conflict}. Assign each role to at most one dimension."
            return

        # Read roles (only for dims present in the data)
        rs = _active_role_sels()
        vf = _active_val_filters()
        active_dims = [d for d in _profile_dim_order() if d in df.columns]
        roles = {dim: rs[dim].value for dim in active_dims}

        # 1. Apply value filters for dims with a filterable role
        filtered = df
        for dim, role in roles.items():
            if role in ROLES_WITH_VALUE_FILTER:
                selected = vf[dim].value
                if selected:
                    filtered = filtered[filtered[dim].isin(selected)]

        # Apply k range (clustering only; MCQ uses k=1)
        if "k" in filtered.columns:
            k_lo, k_hi = k_slider.value
            filtered = filtered[(filtered["k"] >= k_lo) & (filtered["k"] <= k_hi)]

        # 2. Resolve facet/color dims from roles
        facet_row = next((d for d, r in roles.items() if r == "Grid rows"), None)
        facet_col = next((d for d, r in roles.items() if r == "Grid cols"), None)
        color_dim = next((d for d, r in roles.items() if r == "Color"), None)

        # 3. Dims with a visual or mean-group role are grouped in aggregation
        group_dims = [
            d
            for d, r in roles.items()
            if r in ("Grid rows", "Grid cols", "Color", "Mean group")
        ]

        x_axis = x_axis_sel.value
        agg_mode = agg_radio.value
        is_line = plot_type_sel.value == "Line plot"
        connect_lines = connect_lines_chk.value

        bar_color_ignored = bool(not is_line and color_dim)

        if is_line:
            y_metric = y_axis_sel.value
            aggregated = aggregate_df(
                filtered,
                y_metric,
                agg_mode,
                group_dims=group_dims,
                x_axis=x_axis,
            )
            fig = build_figure(
                aggregated,
                y_metric=y_metric,
                x_axis=x_axis,
                row_dim=facet_row,
                col_dim=facet_col,
                color_dim=color_dim,
                agg_mode=agg_mode,
                plot_height=height_slider.value,
                connect_lines=connect_lines,
            )
        else:
            chosen = list(metrics_multi.value or [])[:MAX_BAR_METRICS]
            long_df = aggregated_metrics_long(
                filtered,
                chosen,
                agg_mode,
                group_dims,
                x_axis,
                facet_row,
                facet_col,
            )
            fig = build_figure_bars(
                long_df,
                x_axis,
                facet_row,
                facet_col,
                agg_mode,
                plot_height=height_slider.value,
            )

        plot_pane.object = fig
        plot_pane.height = height_slider.value

        suffix = ""
        if bar_color_ignored:
            suffix = (
                "\n\n_Bar mode: **Color** role is ignored; "
                "legend shows metrics._"
            )
        load_status.object = base + suffix

    # --- Role change callback ---
    def make_role_callback(dim: str, vf: pn.widgets.MultiSelect):
        def on_role_change(event):
            new_role = event.new
            vf.visible = (new_role in ROLES_WITH_VALUE_FILTER)
            if new_role in ROLES_WITH_VALUE_FILTER and not vf.options:
                df = _state.get("df", pd.DataFrame())
                if dim == BIN_DIM:
                    edges = parse_bin_edges(bin_edges_input.value)
                    if edges and not df.empty and "n_samples" in df.columns:
                        bins = assign_n_samples_bin(df["n_samples"], edges)
                        opts = sorted(bins.unique().tolist())
                        vf.options = opts
                        vf.value = opts
                elif not df.empty and dim in df.columns:
                    opts = sorted(df[dim].unique().tolist())
                    vf.options = opts
                    vf.value = opts
            refresh_plot()
        return on_role_change

    # --- Source toggle callback ---
    def on_source_change(event=None):
        if _is_mcq() and source_toggle.value == "Files":
            source_toggle.value = "Database"
            return
        is_files = source_toggle.value == "Files"
        data_dir_input.visible = is_files
        sweep_select.visible = not is_files
        if not is_files:
            _populate_sweep_options()

    def _populate_sweep_options():
        """Populate DB scope dropdown (clustering sweep or MCQ request)."""
        try:
            from panel_app.helpers import get_db_connection
            from study_query_llm.db.raw_call_repository import RawCallRepository
            from study_query_llm.services.sweep_query_service import SweepQueryService
            from study_query_llm.services.sweep_request_service import SweepRequestService

            db = get_db_connection()
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                qsvc = SweepQueryService(repo)
                if _is_mcq():
                    rsvc = SweepRequestService(repo)
                    reqs = rsvc.list_requests(
                        sweep_type=SWEEP_TYPE_MCQ,
                        include_fulfilled=True,
                    )
                    opts: dict = {"All": None}
                    for r in reqs:
                        label = f"{r['name']}  (id={r['id']}, {r.get('request_status', '?')})"
                        opts[label] = r["id"]
                    sweep_select.name = "MCQ request"
                else:
                    sweeps = qsvc.list_clustering_sweeps()
                    opts = {"All": None}
                    for sw in sweeps:
                        label = f"{sw['name']}  ({sw['n_runs']} runs)"
                        opts[label] = sw["id"]
                    sweep_select.name = "Clustering sweep"
            sweep_select.options = opts
            sweep_select.value = None
        except Exception as exc:
            logger.warning("Could not populate sweep list: %s", exc)

    source_toggle.param.watch(on_source_change, "value")

    def _apply_profile_widgets(event=None):
        """Axis options, row visibility, and source rules when sweep type changes."""
        prof = _active_profile()
        _state["df"] = pd.DataFrame()
        _state["loaded"] = False
        _state["load_summary"] = "_Sweep type changed — click **Load / Reload**._"
        load_status.object = _state["load_summary"]
        y_axis_sel.options = list(prof.metrics)
        if y_axis_sel.value not in prof.metrics:
            y_axis_sel.value = prof.default_y
        metrics_multi.options = list(prof.metrics)
        if metrics_multi.value and all(m in prof.metrics for m in metrics_multi.value):
            pass
        else:
            metrics_multi.value = (
                [prof.default_y]
                if prof.default_y in prof.metrics
                else ([prof.metrics[0]] if prof.metrics else [])
            )
        x_axis_sel.options = list(prof.x_axis_options)
        if x_axis_sel.value not in prof.x_axis_options:
            x_axis_sel.value = prof.default_x
        dim_row_clustering.visible = not _is_mcq()
        dim_row_mcq.visible = _is_mcq()
        k_range_section.visible = not _is_mcq()
        if _is_mcq():
            source_toggle.value = "Database"
            data_dir_input.visible = False
            sweep_select.visible = True
            _populate_sweep_options()
        else:
            on_source_change()

    # --- Load logic ---
    def load_data(event=None):
        load_status.object = "_Loading…_"
        try:
            if source_toggle.value == "Database":
                df = _load_from_database()
            else:
                df = _load_from_files()

            _state["df"] = df
            _state["loaded"] = True

            if df.empty:
                if source_toggle.value == "Database":
                    load_status.object = "**No sweep run groups found** in the database."
                else:
                    load_status.object = "**No matching pkl files found** in that directory."
                _state["load_summary"] = load_status.object
                return

            cat_dims = (
                list(MCQ_CATEGORICAL_DIMS) if _is_mcq() else list(CATEGORICAL_DIMS)
            )
            vf_act = _active_val_filters()
            rs_act = _active_role_sels()
            for dim in cat_dims:
                if dim in df.columns:
                    opts = sorted(df[dim].unique().tolist())
                    vf_act[dim].options = opts
                    if rs_act[dim].value in ROLES_WITH_VALUE_FILTER:
                        vf_act[dim].value = opts

            # Update n_samples_bin options if bin edges are set
            edges = parse_bin_edges(bin_edges_input.value)
            if edges and "n_samples" in df.columns:
                bins = assign_n_samples_bin(df["n_samples"], edges)
                bin_opts = sorted(bins.unique().tolist())
                val_filters_bin[BIN_DIM].options = bin_opts
                if rs_act[BIN_DIM].value in ROLES_WITH_VALUE_FILTER:
                    val_filters_bin[BIN_DIM].value = bin_opts

            if "k" in df.columns:
                k_min = int(df["k"].min())
                k_max = int(df["k"].max())
                k_slider.start = k_min
                k_slider.end = k_max
                k_slider.value = (k_min, k_max)
                k_part = f"k: {k_min}–{k_max}"
            else:
                k_part = ""

            dim_summary = " | ".join(
                f"{dim}: {df[dim].nunique()}"
                for dim in cat_dims
                if dim in df.columns
            )
            if source_toggle.value == "Database":
                source_hint = "database"
            else:
                source_hint = data_dir_input.value.strip()
            load_status.object = (
                f"**Loaded:** {len(df)} rows from {source_hint} | "
                f"{dim_summary}"
                + (f" | {k_part}" if k_part else "")
            )
            _state["load_summary"] = load_status.object
            refresh_plot()
        except Exception as e:
            logger.error("Failed to load sweep data: %s", e, exc_info=True)
            load_status.object = f"**Error:** {e}"
            _state["load_summary"] = load_status.object

    def _load_from_files() -> pd.DataFrame:
        data_dir = data_dir_input.value.strip()
        return load_sweep_data(data_dir)

    def _load_from_database() -> pd.DataFrame:
        from panel_app.helpers import get_db_connection
        from study_query_llm.db.raw_call_repository import RawCallRepository
        from study_query_llm.services.sweep_query_service import SweepQueryService

        db = get_db_connection()
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            svc = SweepQueryService(repo)
            scope_id = sweep_select.value  # None means "All"
            if _is_mcq():
                df = svc.get_mcq_metrics_df(mcq_request_id=scope_id)
                extra_pct = [
                    c
                    for c in df.columns
                    if c.startswith("pct_") and c not in MCQ_METRICS
                ]
                if extra_pct:
                    cur = list(y_axis_sel.options)
                    merged = list(dict.fromkeys(cur + sorted(extra_pct)))
                    y_axis_sel.options = merged
                    m_opts = list(metrics_multi.options)
                    merged_m = list(dict.fromkeys(m_opts + sorted(extra_pct)))
                    metrics_multi.options = merged_m
            else:
                df = svc.get_sweep_metrics_df(clustering_sweep_id=scope_id)
        return df

    # --- Bin edges change callback ---
    def _update_bin_dim_options(event=None):
        """Recompute n_samples_bin filter options when bin edges change."""
        df = _state.get("df", pd.DataFrame())
        if df.empty:
            refresh_plot()
            return
        edges = parse_bin_edges(bin_edges_input.value)
        vf = val_filters_bin[BIN_DIM]
        rs_bin = role_sels_bin[BIN_DIM]
        if edges and "n_samples" in df.columns:
            bins = assign_n_samples_bin(df["n_samples"], edges)
            opts = sorted(bins.unique().tolist())
            vf.options = opts
            if rs_bin.value in ROLES_WITH_VALUE_FILTER:
                vf.value = opts
        else:
            vf.options = []
            vf.value = []
        refresh_plot()

    # --- Wire callbacks ---
    load_button.on_click(load_data)

    for dim in role_sels_c:
        if dim == BIN_DIM:
            continue
        role_sels_c[dim].param.watch(make_role_callback(dim, val_filters_c[dim]), "value")
        val_filters_c[dim].param.watch(refresh_plot, "value")
    for dim in role_sels_m:
        if dim == BIN_DIM:
            continue
        role_sels_m[dim].param.watch(make_role_callback(dim, val_filters_m[dim]), "value")
        val_filters_m[dim].param.watch(refresh_plot, "value")
    role_sels_bin[BIN_DIM].param.watch(
        make_role_callback(BIN_DIM, val_filters_bin[BIN_DIM]), "value"
    )
    val_filters_bin[BIN_DIM].param.watch(refresh_plot, "value")

    sweep_type_sel.param.watch(_apply_profile_widgets, "value")

    bin_edges_input.param.watch(_update_bin_dim_options, "value")

    def _sync_plot_type_widgets(event=None):
        is_line = plot_type_sel.value == "Line plot"
        y_axis_sel.visible = is_line
        metrics_multi.visible = not is_line
        connect_lines_chk.visible = is_line
        refresh_plot()

    plot_type_sel.param.watch(_sync_plot_type_widgets, "value")

    for widget in [y_axis_sel, x_axis_sel, agg_radio, metrics_multi, connect_lines_chk]:
        widget.param.watch(refresh_plot, "value")
    for widget in [height_slider, k_slider]:
        widget.param.watch(refresh_plot, "value_throttled")

    # --- Layout ---
    dim_roles_column = pn.Column(
        dim_row_clustering,
        dim_row_mcq,
        dim_row_bin,
        sizing_mode="stretch_width",
    )

    controls = pn.Column(
        pn.pane.Markdown("### Sweep type"),
        sweep_type_sel,
        pn.layout.Divider(),
        pn.pane.Markdown("### Data Source"),
        source_toggle,
        pn.Row(data_dir_input, sweep_select, load_button),
        load_status,
        pn.layout.Divider(),
        pn.pane.Markdown("### Axes & plot"),
        plot_type_sel,
        pn.Row(y_axis_sel, x_axis_sel),
        metrics_multi,
        connect_lines_chk,
        pn.layout.Divider(),
        pn.pane.Markdown(
            "### Sample Size Binning\n"
            "_Optional. Define bin edges to group runs by sample size "
            "(e.g. for ESTELA \u2264280). The derived dimension appears in the role selectors below._"
        ),
        bin_edges_input,
        pn.layout.Divider(),
        pn.pane.Markdown(
            "### Dimension Roles\n"
            "_Each dimension can be: **Free** (pass-through), **Filter** (include/exclude values), "
            "**Mean group** (split aggregation means without facet/color), **Grid rows**, **Grid cols**, or **Color**. "
            "Each exclusive role (**Grid rows**, **Grid cols**, **Color**) may only be assigned to one dimension. "
            "Dimensions with a plot role also show a value selector to include/exclude specific groups._"
        ),
        dim_roles_column,
        pn.layout.Divider(),
        pn.pane.Markdown("### Aggregation"),
        agg_radio,
        pn.layout.Divider(),
        k_range_section,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display"),
        height_slider,
        sizing_mode="stretch_width",
        margin=(10, 10),
    )

    _sync_plot_type_widgets()

    return pn.Column(
        pn.pane.Markdown("## Sweep Explorer"),
        controls,
        pn.layout.Divider(),
        plot_pane,
        sizing_mode="stretch_width",
        margin=(10, 20),
    )
