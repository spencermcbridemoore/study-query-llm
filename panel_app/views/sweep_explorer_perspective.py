"""
Sweep Explorer (Perspective) — same sweep data as the Plotly tab, interactive pivots.

Uses ``pn.pane.Perspective`` for drag-and-drop layout instead of per-dimension roles.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import panel as pn

from study_query_llm.experiments.sweep_request_types import SWEEP_TYPE_MCQ
from study_query_llm.utils.logging_config import get_logger

from panel_app.views.sweep_explorer import (
    AGGREGATION_OPTIONS,
    BIN_DIM,
    METRICS,
    X_AXIS_OPTIONS,
    _CLUSTERING_PROFILE,
    _MCQ_PROFILE,
    aggregate_df,
    assign_n_samples_bin,
    parse_bin_edges,
)
from panel_app.views.sweep_explorer_load import fetch_sweep_dataframe

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _dims_for_grouping(
    df: pd.DataFrame,
    *,
    bin_edges_text: str,
    profile_categorical_dims: tuple[str, ...],
) -> list[str]:
    """Categorical columns present in *df* that can be used as group dimensions."""
    w = df.copy()
    edges = parse_bin_edges(bin_edges_text)
    if edges and "n_samples" in w.columns:
        w[BIN_DIM] = assign_n_samples_bin(w["n_samples"], edges)
    else:
        w = w.drop(columns=[BIN_DIM], errors="ignore")
    opts = [c for c in profile_categorical_dims if c in w.columns]
    if BIN_DIM in w.columns:
        opts.append(BIN_DIM)
    return opts


def create_sweep_explorer_perspective_ui() -> pn.viewable.Viewable:
    """Sweep Explorer tab with Perspective datagrid and explicit pre-aggregation."""

    _state: dict = {
        "df": pd.DataFrame(),
        "loaded": False,
        "load_summary": "_No data loaded._",
    }

    sweep_type_sel = pn.widgets.RadioButtonGroup(
        name="Sweep type",
        options=["Clustering", "MCQ"],
        value="Clustering",
        button_type="default",
    )

    source_toggle = pn.widgets.RadioButtonGroup(
        name="Source",
        options=["Files", "Database"],
        value="Files",
        button_type="default",
    )

    data_dir_input = pn.widgets.TextInput(
        name="Data directory",
        value=str(REPO_ROOT / "experimental_results"),
        width=420,
        placeholder="Path to folder containing .pkl files",
    )

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

    bin_edges_input = pn.widgets.TextInput(
        name="Bin edges (comma-separated)",
        value="",
        placeholder="e.g. 100, 280, 400",
        width=300,
    )

    agg_radio = pn.widgets.RadioButtonGroup(
        name="Preaggregate",
        options=AGGREGATION_OPTIONS,
        value="Mean",
        button_type="default",
    )

    y_metric_sel = pn.widgets.Select(
        name="Y metric (single-metric aggregation)",
        options=list(METRICS),
        value="ari",
        width=260,
    )

    x_axis_sel = pn.widgets.Select(
        name="X axis (aggregation / ordering)",
        options=list(X_AXIS_OPTIONS),
        value="k",
        width=160,
    )

    group_dims_sel = pn.widgets.MultiSelect(
        name="Group dimensions",
        options=[],
        value=[],
        height=140,
        width=320,
    )

    perspective_pane = pn.pane.Perspective(
        pd.DataFrame(),
        plugin="datagrid",
        sizing_mode="stretch_width",
        height=700,
        settings=True,
    )

    def _is_mcq() -> bool:
        return sweep_type_sel.value == "MCQ"

    def _active_profile():
        return _MCQ_PROFILE if _is_mcq() else _CLUSTERING_PROFILE

    def _apply_profile_widgets(event=None):
        prof = _active_profile()
        _state["df"] = pd.DataFrame()
        _state["loaded"] = False
        _state["load_summary"] = "_Sweep type changed — click **Load / Reload**._"
        load_status.object = _state["load_summary"]
        y_metric_sel.options = list(prof.metrics)
        if y_metric_sel.value not in prof.metrics:
            y_metric_sel.value = prof.default_y
        x_axis_sel.options = list(prof.x_axis_options)
        if x_axis_sel.value not in prof.x_axis_options:
            x_axis_sel.value = prof.default_x
        k_range_section.visible = not _is_mcq()
        if _is_mcq():
            source_toggle.value = "Database"
            data_dir_input.visible = False
            sweep_select.visible = True
            _populate_sweep_options()
        else:
            on_source_change()

    def _populate_sweep_options():
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
                        label = (
                            f"{r['name']}  (id={r['id']}, {r.get('request_status', '?')})"
                        )
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

    def on_source_change(event=None):
        if _is_mcq() and source_toggle.value == "Files":
            source_toggle.value = "Database"
            return
        is_files = source_toggle.value == "Files"
        data_dir_input.visible = is_files
        sweep_select.visible = not is_files
        if not is_files:
            _populate_sweep_options()

    def _sync_group_dims_from_df(df: pd.DataFrame):
        prof = _active_profile()
        opts = _dims_for_grouping(
            df,
            bin_edges_text=bin_edges_input.value,
            profile_categorical_dims=prof.categorical_dims,
        )
        group_dims_sel.options = opts
        prev = set(group_dims_sel.value or [])
        kept = [c for c in opts if c in prev]
        group_dims_sel.value = kept if kept else opts

    def refresh_perspective(event=None):
        df = _state.get("df", pd.DataFrame())
        base = _state.get("load_summary", "_No data loaded._")
        if df.empty:
            perspective_pane.object = pd.DataFrame()
            return

        out = df.copy()
        if "k" in out.columns:
            k_lo, k_hi = k_slider.value
            out = out[(out["k"] >= k_lo) & (out["k"] <= k_hi)]

        edges = parse_bin_edges(bin_edges_input.value)
        if edges and "n_samples" in out.columns:
            out[BIN_DIM] = assign_n_samples_bin(out["n_samples"], edges)
        else:
            out = out.drop(columns=[BIN_DIM], errors="ignore")

        y_metric = y_metric_sel.value
        if y_metric not in out.columns:
            load_status.object = base + f"\n\n_Metric `{y_metric}` not in filtered data._"
            perspective_pane.object = pd.DataFrame()
            return

        gd = list(group_dims_sel.value or [])
        valid = set(group_dims_sel.options)
        gd = [c for c in gd if c in valid]
        if not gd:
            gd = [c for c in group_dims_sel.options]

        agg_mode = agg_radio.value
        x_axis = x_axis_sel.value

        if agg_mode == "Raw runs":
            prepared = out
        else:
            prepared = aggregate_df(
                out,
                y_metric,
                agg_mode,
                group_dims=gd,
                x_axis=x_axis,
            )

        perspective_pane.object = prepared
        load_status.object = base

    def load_data(event=None):
        load_status.object = "_Loading…_"
        try:
            df, extra_pct = fetch_sweep_dataframe(
                is_mcq=_is_mcq(),
                source_is_database=(source_toggle.value == "Database"),
                data_dir=data_dir_input.value.strip(),
                scope_id=sweep_select.value,
            )
            if extra_pct:
                cur = list(y_metric_sel.options)
                merged = list(dict.fromkeys(cur + sorted(extra_pct)))
                y_metric_sel.options = merged

            _state["df"] = df
            _state["loaded"] = True

            if df.empty:
                if source_toggle.value == "Database":
                    load_status.object = "**No sweep run groups found** in the database."
                else:
                    load_status.object = "**No matching pkl files found** in that directory."
                _state["load_summary"] = load_status.object
                perspective_pane.object = pd.DataFrame()
                return

            if "k" in df.columns:
                k_min = int(df["k"].min())
                k_max = int(df["k"].max())
                k_slider.start = k_min
                k_slider.end = k_max
                k_slider.value = (k_min, k_max)
                k_part = f"k: {k_min}–{k_max}"
            else:
                k_part = ""

            prof = _active_profile()
            cat_dims = list(prof.categorical_dims)
            dim_summary = " | ".join(
                f"{dim}: {df[dim].nunique()}" for dim in cat_dims if dim in df.columns
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

            _sync_group_dims_from_df(df)
            refresh_perspective()
        except Exception as e:
            logger.error("Failed to load sweep data: %s", e, exc_info=True)
            load_status.object = f"**Error:** {e}"
            _state["load_summary"] = load_status.object

    source_toggle.param.watch(on_source_change, "value")

    load_button.on_click(load_data)

    sweep_type_sel.param.watch(_apply_profile_widgets, "value")

    def _on_bin_edges_change(event=None):
        df0 = _state.get("df", pd.DataFrame())
        if not df0.empty:
            _sync_group_dims_from_df(df0)
        refresh_perspective()

    bin_edges_input.param.watch(_on_bin_edges_change, "value")

    for w in (
        agg_radio,
        y_metric_sel,
        x_axis_sel,
        group_dims_sel,
    ):
        w.param.watch(refresh_perspective, "value")
    k_slider.param.watch(refresh_perspective, "value_throttled")

    on_source_change()

    controls = pn.Column(
        pn.pane.Markdown("## Sweep Explorer (Perspective)"),
        pn.pane.Markdown("### Sweep type"),
        sweep_type_sel,
        pn.layout.Divider(),
        pn.pane.Markdown("### Data Source"),
        source_toggle,
        pn.Row(data_dir_input, sweep_select, load_button),
        load_status,
        pn.layout.Divider(),
        pn.pane.Markdown(
            "### Sample Size Binning\n"
            "_Optional. Bin edges add a derived `n_samples_bin` column for grouping._"
        ),
        bin_edges_input,
        pn.layout.Divider(),
        k_range_section,
        pn.layout.Divider(),
        pn.pane.Markdown("### Preaggregate"),
        agg_radio,
        pn.Row(y_metric_sel, x_axis_sel),
        group_dims_sel,
        pn.pane.Markdown(
            "_When not **Raw runs**, a single **Y metric** is aggregated; "
            "**Group dimensions** select which categoricals (plus optional bin) define groups._"
        ),
        sizing_mode="stretch_width",
        margin=(10, 10),
    )

    return pn.Column(
        controls,
        pn.layout.Divider(),
        perspective_pane,
        sizing_mode="stretch_width",
        margin=(10, 20),
    )
