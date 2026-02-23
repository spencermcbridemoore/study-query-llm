"""
Sweep Explorer tab — interactive Plotly-based viewer for pkl experiment results.

Loads no_pca_50runs_*.pkl and experimental_sweep_*.pkl files, flattens them into
a tidy DataFrame, and renders configurable faceted line/box plots with full
dimensional control (rows, cols, color, x, y, aggregation, filters).
"""

from __future__ import annotations

import pickle
import sys
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

# Metrics available on Y-axis
METRICS = [
    "objective",
    "dispersion",
    "silhouette",
    "ari",
    "cosine_sim",
    "cosine_sim_norm",
]

# Categorical dimensions available for faceting / color / filters
CATEGORICAL_DIMS = ["dataset", "engine", "summarizer", "data_type"]

# Numeric dimensions usable on X-axis
X_AXIS_OPTIONS = ["k", "run_idx", "n_samples"]

AGGREGATION_OPTIONS = ["Raw runs", "Mean", "Mean ± stdev", "Last k only"]


# ===========================================================================
# Data loading helpers
# ===========================================================================

def _dist_from_z(result: dict) -> Optional[np.ndarray]:
    dist = result.get("dist")
    if dist is not None:
        return np.asarray(dist)
    Z = result.get("Z")
    if Z is None:
        return None
    Z = np.asarray(Z)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = Z / np.maximum(norms, 1e-12)
    return np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, 2.0)


def _try_ari(gt, labels):
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(gt, labels))
    except Exception:
        return None


def _try_silhouette(dist, labels):
    try:
        from sklearn.metrics import silhouette_score
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        return None


def _rows_from_50runs(item: dict) -> list[dict]:
    """Flatten one no_pca_50runs_*.pkl item into a list of row dicts."""
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

    dist = _dist_from_z(result)

    rows = []
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
            lab = np.asarray(labels_all[i]) if (labels_all and i < len(labels_all)) else None

            n = n_samples
            if n == 0 and lab is not None:
                n = len(lab)

            dispersion = (float(ob) / n) if (ob is not None and n > 0) else None
            cosine_sim = (1.0 - dispersion) if dispersion is not None else None
            cosine_sim_norm = ((cosine_sim + 1.0) / 2.0) if cosine_sim is not None else None
            sil = _try_silhouette(dist, lab) if (dist is not None and lab is not None) else None
            ari = _try_ari(gt, lab) if (gt is not None and lab is not None and len(lab) == len(gt)) else None

            rows.append({
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
            })
    return rows


def _rows_from_sweep(item: dict) -> list[dict]:
    """Flatten one experimental_sweep_*.pkl item into a list of row dicts."""
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
        dist_arr = _dist_from_z(result)

    rows = []
    for k_str, entry in by_k.items():
        try:
            k = int(k_str)
        except ValueError:
            continue
        labels = entry.get("labels")
        if labels is not None:
            labels = np.asarray(labels)
        n = len(labels) if labels is not None else (len(gt) if gt is not None else 0)

        ob_raw = entry.get("objective")
        if isinstance(ob_raw, dict):
            ob = ob_raw.get("value")
        elif isinstance(ob_raw, (int, float)):
            ob = float(ob_raw)
        else:
            ob = None

        dispersion = (float(ob) / n) if (ob is not None and n > 0) else None
        cosine_sim = (1.0 - dispersion) if dispersion is not None else None
        cosine_sim_norm = ((cosine_sim + 1.0) / 2.0) if cosine_sim is not None else None

        # Silhouette: prefer stability dict, fall back to precomputed
        stab = entry.get("stability")
        sil = None
        if isinstance(stab, dict):
            sil = stab.get("silhouette", {}).get("mean")
        elif dist_arr is not None and labels is not None and len(labels) == dist_arr.shape[0]:
            sil = _try_silhouette(dist_arr, labels)

        ari = _try_ari(gt, labels) if (gt is not None and labels is not None and len(labels) == len(gt)) else None

        rows.append({
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
        })
    return rows


def load_sweep_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan data_dir for both pkl patterns and flatten into a tidy DataFrame.

    Returns an empty DataFrame with the correct schema if nothing is found.
    """
    data_dir = Path(data_dir)
    all_rows: list[dict] = []

    # --- 50-runs files ---
    for p in sorted(data_dir.glob("no_pca_50runs_*.pkl")):
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

def aggregate_df(
    df: pd.DataFrame,
    y_metric: str,
    agg_mode: str,
    group_dims: list[str] | None = None,
) -> pd.DataFrame:
    """
    Transform the flat DataFrame according to the aggregation mode.

    Args:
        group_dims: Which categorical dimensions to group by. Only dims that
            have a visual role (Grid rows/cols/Color) should be included so that
            "Free" dims don't accidentally split aggregated series. Defaults to
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

    group_cols = [c for c in group_dims + ["k", "n_samples"] if c in df.columns]

    if agg_mode == "Last k only":
        cat_cols = [c for c in group_dims if c in df.columns]
        idx = df.groupby(cat_cols)["k"].transform("max") if cat_cols else df["k"].max()
        filtered = df[df["k"] == idx].copy()
        out = filtered.groupby(group_cols, as_index=False)[y_metric].mean() if group_cols else filtered
        out["y_err"] = np.nan
        return out

    if agg_mode in ("Mean", "Mean ± stdev"):
        if not group_cols:
            # No grouping dims — aggregate everything into one series
            mean_val = df[y_metric].mean()
            std_val = df[y_metric].std() if agg_mode == "Mean ± stdev" else np.nan
            # Keep k progression by grouping on k alone
            agg = df.groupby(["k"], as_index=False).agg(
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
) -> go.Figure:
    """
    Build a Plotly figure from the prepared DataFrame.

    Uses px.line for standard modes, px.scatter for Last k only,
    and adds error bands for Mean±stdev.
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

    common_kwargs = dict(
        x=x_axis if x_axis in plot_df.columns else "k",
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

    return fig


# Roles available for each categorical dimension
DIM_ROLES = ["Free", "Filter", "Grid rows", "Grid cols", "Color"]

# Exclusive roles — only one dimension may hold each at a time
EXCLUSIVE_ROLES = {"Grid rows", "Grid cols", "Color"}

# Default role for each categorical dimension
DIM_DEFAULT_ROLES = {
    "dataset": "Grid cols",
    "engine": "Color",
    "summarizer": "Grid rows",
    "data_type": "Filter",
}


# ===========================================================================
# Panel UI
# ===========================================================================

def create_sweep_explorer_ui() -> pn.viewable.Viewable:
    """Create the interactive Sweep Explorer tab with per-dimension role selectors."""

    # --- State ---
    _state: dict = {"df": pd.DataFrame(), "loaded": False}

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

    # --- Per-dimension role widgets ---
    # role_sels[dim]   = Select widget for the role
    # val_filters[dim] = MultiSelect for value filtering (visible only when role == "Filter")
    # dim_blocks[dim]  = Column containing both widgets (rendered in the layout)
    role_sels: dict[str, pn.widgets.Select] = {}
    val_filters: dict[str, pn.widgets.MultiSelect] = {}
    dim_blocks: dict[str, pn.Column] = {}

    for dim in CATEGORICAL_DIMS:
        default_role = DIM_DEFAULT_ROLES.get(dim, "Free")
        role_sel = pn.widgets.Select(
            name=f"{dim} role",
            options=DIM_ROLES,
            value=default_role,
            width=160,
        )
        val_filter = pn.widgets.MultiSelect(
            name=f"values",
            options=[],
            value=[],
            height=100,
            width=175,
            visible=(default_role == "Filter"),
        )
        role_sels[dim] = role_sel
        val_filters[dim] = val_filter
        dim_blocks[dim] = pn.Column(
            pn.pane.Markdown(f"**{dim}**"),
            role_sel,
            val_filter,
            width=190,
            margin=(0, 8, 0, 0),
        )

    # --- Plot pane ---
    plot_pane = pn.pane.Plotly(
        build_figure(pd.DataFrame(), "ari", "k", "summarizer", "dataset", "engine", "Mean"),
        sizing_mode="stretch_width",
        height=700,
    )

    # --- Conflict check ---
    def _check_role_conflicts() -> str | None:
        """Return a warning string if any exclusive role is assigned to 2+ dims."""
        role_counts: dict[str, list[str]] = {}
        for dim in CATEGORICAL_DIMS:
            role = role_sels[dim].value
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
        if df.empty:
            return

        # Conflict guard
        conflict = _check_role_conflicts()
        if conflict:
            load_status.object = f"**Role conflict** — {conflict}. Assign each role to at most one dimension."
            return

        # Read roles
        roles = {dim: role_sels[dim].value for dim in CATEGORICAL_DIMS}

        # 1. Apply value filters for dims with role == "Filter"
        filtered = df.copy()
        for dim, role in roles.items():
            if role == "Filter":
                selected = val_filters[dim].value
                if selected:
                    filtered = filtered[filtered[dim].isin(selected)]

        # Apply k range
        k_lo, k_hi = k_slider.value
        filtered = filtered[(filtered["k"] >= k_lo) & (filtered["k"] <= k_hi)]

        # 2. Resolve facet/color dims from roles
        facet_row = next((d for d, r in roles.items() if r == "Grid rows"), None)
        facet_col = next((d for d, r in roles.items() if r == "Grid cols"), None)
        color_dim = next((d for d, r in roles.items() if r == "Color"), None)

        # 3. Dims with a visual role are grouped in aggregation; Free/Filter dims are not
        group_dims = [d for d, r in roles.items() if r in ("Grid rows", "Grid cols", "Color")]

        y_metric = y_axis_sel.value
        x_axis = x_axis_sel.value
        agg_mode = agg_radio.value

        aggregated = aggregate_df(filtered, y_metric, agg_mode, group_dims=group_dims)

        fig = build_figure(
            aggregated,
            y_metric=y_metric,
            x_axis=x_axis,
            row_dim=facet_row,
            col_dim=facet_col,
            color_dim=color_dim,
            agg_mode=agg_mode,
            plot_height=height_slider.value,
        )
        plot_pane.object = fig
        plot_pane.height = height_slider.value

    # --- Role change callback ---
    def make_role_callback(dim: str):
        def on_role_change(event):
            new_role = event.new
            vf = val_filters[dim]
            vf.visible = (new_role == "Filter")
            # Populate filter options from loaded data if switching to Filter and empty
            if new_role == "Filter" and not vf.options:
                df = _state.get("df", pd.DataFrame())
                if not df.empty and dim in df.columns:
                    opts = sorted(df[dim].unique().tolist())
                    vf.options = opts
                    vf.value = opts
            refresh_plot()
        return on_role_change

    # --- Source toggle callback ---
    def on_source_change(event=None):
        is_files = source_toggle.value == "Files"
        data_dir_input.visible = is_files

    source_toggle.param.watch(on_source_change, "value")

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
                return

            k_min = int(df["k"].min())
            k_max = int(df["k"].max())

            for dim in CATEGORICAL_DIMS:
                if dim in df.columns:
                    opts = sorted(df[dim].unique().tolist())
                    val_filters[dim].options = opts
                    if role_sels[dim].value == "Filter":
                        val_filters[dim].value = opts

            k_slider.start = k_min
            k_slider.end = k_max
            k_slider.value = (k_min, k_max)

            dim_summary = " | ".join(
                f"{dim}: {df[dim].nunique()}"
                for dim in CATEGORICAL_DIMS
                if dim in df.columns
            )
            if source_toggle.value == "Database":
                source_hint = "database"
            else:
                source_hint = data_dir_input.value.strip()
            load_status.object = (
                f"**Loaded:** {len(df)} rows from {source_hint} | "
                f"{dim_summary} | k: {k_min}–{k_max}"
            )
            refresh_plot()
        except Exception as e:
            logger.error("Failed to load sweep data: %s", e, exc_info=True)
            load_status.object = f"**Error:** {e}"

    def _load_from_files() -> pd.DataFrame:
        data_dir = data_dir_input.value.strip()
        return load_sweep_data(data_dir)

    def _load_from_database() -> pd.DataFrame:
        from panel_app.helpers import get_db_connection
        from study_query_llm.db.raw_call_repository import RawCallRepository
        from study_query_llm.services.sweep_query_service import SweepQueryService

        db = get_db_connection()
        session = db.get_session()
        repo = RawCallRepository(session)
        svc = SweepQueryService(repo)
        return svc.get_sweep_metrics_df()

    # --- Wire callbacks ---
    load_button.on_click(load_data)

    for dim in CATEGORICAL_DIMS:
        role_sels[dim].param.watch(make_role_callback(dim), "value")
        val_filters[dim].param.watch(refresh_plot, "value")

    for widget in [y_axis_sel, x_axis_sel, agg_radio, height_slider, k_slider]:
        widget.param.watch(refresh_plot, "value")

    # --- Layout ---
    dim_roles_row = pn.Row(*[dim_blocks[dim] for dim in CATEGORICAL_DIMS])

    controls = pn.Column(
        pn.pane.Markdown("### Data Source"),
        source_toggle,
        pn.Row(data_dir_input, load_button),
        load_status,
        pn.layout.Divider(),
        pn.pane.Markdown("### Axes"),
        pn.Row(y_axis_sel, x_axis_sel),
        pn.layout.Divider(),
        pn.pane.Markdown(
            "### Dimension Roles\n"
            "_Each dimension can be: **Free** (pass-through), **Filter** (include/exclude values), "
            "**Grid rows**, **Grid cols**, or **Color**. Each exclusive role may only be assigned to one dimension._"
        ),
        dim_roles_row,
        pn.layout.Divider(),
        pn.pane.Markdown("### Aggregation"),
        agg_radio,
        pn.layout.Divider(),
        pn.pane.Markdown("### k Range"),
        k_slider,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display"),
        height_slider,
        sizing_mode="stretch_width",
        margin=(10, 10),
    )

    return pn.Column(
        pn.pane.Markdown("## Sweep Explorer"),
        controls,
        pn.layout.Divider(),
        plot_pane,
        sizing_mode="stretch_width",
        margin=(10, 20),
    )
