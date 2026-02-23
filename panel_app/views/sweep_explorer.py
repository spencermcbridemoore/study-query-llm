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

def aggregate_df(df: pd.DataFrame, y_metric: str, agg_mode: str) -> pd.DataFrame:
    """
    Transform the flat DataFrame according to the aggregation mode.

    Returns a DataFrame ready for plotting, adding `y_err` column for Mean±stdev.
    """
    if df.empty:
        return df

    if agg_mode == "Raw runs":
        out = df.copy()
        out["y_err"] = np.nan
        return out

    group_cols = [c for c in CATEGORICAL_DIMS + ["k", "n_samples"] if c in df.columns]

    if agg_mode == "Last k only":
        idx = df.groupby([c for c in CATEGORICAL_DIMS if c in df.columns])["k"].transform("max")
        filtered = df[df["k"] == idx].copy()
        out = filtered.groupby(group_cols, as_index=False)[y_metric].mean()
        out["y_err"] = np.nan
        return out

    if agg_mode in ("Mean", "Mean ± stdev"):
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


# ===========================================================================
# Panel UI
# ===========================================================================

def create_sweep_explorer_ui() -> pn.viewable.Viewable:
    """Create the interactive Sweep Explorer tab."""

    # --- State ---
    _state: dict = {"df": pd.DataFrame(), "loaded": False}

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

    # --- Widgets: grid layout ---
    dim_options = ["None"] + CATEGORICAL_DIMS
    row_dim_sel = pn.widgets.Select(
        name="Grid rows",
        options=dim_options,
        value="summarizer",
        width=160,
    )
    col_dim_sel = pn.widgets.Select(
        name="Grid cols",
        options=dim_options,
        value="dataset",
        width=160,
    )
    color_dim_sel = pn.widgets.Select(
        name="Color by",
        options=dim_options,
        value="engine",
        width=160,
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

    # --- Widgets: filters (populated after load) ---
    dataset_filter = pn.widgets.MultiSelect(
        name="Dataset filter",
        options=[],
        value=[],
        height=100,
        width=200,
    )
    engine_filter = pn.widgets.MultiSelect(
        name="Engine filter",
        options=[],
        value=[],
        height=100,
        width=200,
    )
    summarizer_filter = pn.widgets.MultiSelect(
        name="Summarizer filter",
        options=[],
        value=[],
        height=80,
        width=200,
    )
    k_slider = pn.widgets.RangeSlider(
        name="k range",
        start=2,
        end=20,
        value=(2, 20),
        step=1,
        width=250,
    )

    # --- Plot pane ---
    plot_pane = pn.pane.Plotly(
        build_figure(pd.DataFrame(), "ari", "k", "summarizer", "dataset", "engine", "Mean"),
        sizing_mode="stretch_width",
        height=700,
    )

    # --- Load logic ---
    def load_data(event=None):
        data_dir = data_dir_input.value.strip()
        load_status.object = "_Loading…_"
        try:
            df = load_sweep_data(data_dir)
            _state["df"] = df
            _state["loaded"] = True

            if df.empty:
                load_status.object = "**No matching pkl files found** in that directory."
                return

            n_files_hint = f"{len(df)} rows from {data_dir}"

            # Populate filters
            datasets = sorted(df["dataset"].unique().tolist())
            engines = sorted(df["engine"].unique().tolist())
            summarizers = sorted(df["summarizer"].unique().tolist())
            k_min = int(df["k"].min())
            k_max = int(df["k"].max())

            dataset_filter.options = datasets
            dataset_filter.value = datasets
            engine_filter.options = engines
            engine_filter.value = engines
            summarizer_filter.options = summarizers
            summarizer_filter.value = summarizers
            k_slider.start = k_min
            k_slider.end = k_max
            k_slider.value = (k_min, k_max)

            load_status.object = (
                f"**Loaded:** {n_files_hint} | "
                f"datasets: {len(datasets)} | "
                f"engines: {len(engines)} | "
                f"summarizers: {len(summarizers)} | "
                f"k: {k_min}–{k_max}"
            )
            refresh_plot()
        except Exception as e:
            logger.error("Failed to load sweep data: %s", e, exc_info=True)
            load_status.object = f"**Error:** {e}"

    # --- Plot refresh logic ---
    def refresh_plot(event=None):
        df = _state.get("df", pd.DataFrame())
        if df.empty:
            return

        # Apply filters
        filtered = df.copy()
        if dataset_filter.value:
            filtered = filtered[filtered["dataset"].isin(dataset_filter.value)]
        if engine_filter.value:
            filtered = filtered[filtered["engine"].isin(engine_filter.value)]
        if summarizer_filter.value:
            filtered = filtered[filtered["summarizer"].isin(summarizer_filter.value)]
        k_lo, k_hi = k_slider.value
        filtered = filtered[(filtered["k"] >= k_lo) & (filtered["k"] <= k_hi)]

        y_metric = y_axis_sel.value
        x_axis = x_axis_sel.value
        agg_mode = agg_radio.value

        aggregated = aggregate_df(filtered, y_metric, agg_mode)

        fig = build_figure(
            aggregated,
            y_metric=y_metric,
            x_axis=x_axis,
            row_dim=row_dim_sel.value,
            col_dim=col_dim_sel.value,
            color_dim=color_dim_sel.value,
            agg_mode=agg_mode,
            plot_height=height_slider.value,
        )
        plot_pane.object = fig
        plot_pane.height = height_slider.value

    # --- Wire callbacks ---
    load_button.on_click(load_data)

    for widget in [
        y_axis_sel, x_axis_sel, row_dim_sel, col_dim_sel, color_dim_sel,
        agg_radio, height_slider,
        dataset_filter, engine_filter, summarizer_filter, k_slider,
    ]:
        widget.param.watch(refresh_plot, "value")

    # --- Layout ---
    controls = pn.Column(
        pn.pane.Markdown("### Data Source"),
        pn.Row(data_dir_input, load_button),
        load_status,
        pn.layout.Divider(),
        pn.pane.Markdown("### Axes"),
        pn.Row(y_axis_sel, x_axis_sel),
        pn.layout.Divider(),
        pn.pane.Markdown("### Grid Layout"),
        pn.Row(row_dim_sel, col_dim_sel, color_dim_sel),
        pn.layout.Divider(),
        pn.pane.Markdown("### Aggregation"),
        agg_radio,
        pn.layout.Divider(),
        pn.pane.Markdown("### Filters"),
        pn.Row(dataset_filter, engine_filter, summarizer_filter),
        k_slider,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display"),
        height_slider,
        width=480,
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
