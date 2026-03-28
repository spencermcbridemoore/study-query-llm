"""Pure logic tests for Sweep Explorer (no Panel server)."""

from __future__ import annotations

import pandas as pd

from panel_app.views.sweep_explorer import (
    aggregate_df,
    aggregated_metrics_long,
    build_figure,
    build_figure_bars,
)


def test_aggregate_df_includes_mean_group_in_groupby():
    df = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "engine": ["e1", "e1", "e1", "e1"],
            "k": [2, 2, 2, 2],
            "run_idx": [0, 1, 0, 1],
            "n_samples": [10, 10, 10, 10],
            "ari": [0.2, 0.4, 0.3, 0.5],
        }
    )
    # Mean group splits by dataset; no facet on engine — engine still in rows but same value
    g_dims = ["dataset"]
    out = aggregate_df(df, "ari", "Mean", group_dims=g_dims, x_axis="k")
    assert len(out) == 2
    assert set(out["dataset"].tolist()) == {"a", "b"}
    assert abs(float(out.loc[out["dataset"] == "a", "ari"].iloc[0]) - 0.3) < 1e-9


def test_aggregated_metrics_long_distinct_x_when_group_dims_exceed_x():
    df = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "k": [2, 2, 2, 2],
            "ari": [0.1, 0.3, 0.2, 0.4],
            "silhouette": [0.5, 0.6, 0.55, 0.65],
        }
    )
    long_df = aggregated_metrics_long(
        df,
        ["ari", "silhouette"],
        "Mean",
        group_dims=["dataset"],
        x_axis="k",
        facet_row=None,
        facet_col=None,
    )
    assert not long_df.empty
    x_cats = long_df["x_cat"].unique().tolist()
    assert len(x_cats) == 2
    assert all("a" in xc or "b" in xc for xc in x_cats)


def test_build_figure_markers_only_disables_lines():
    df = pd.DataFrame(
        {
            "k": [1, 2, 3],
            "ari": [0.1, 0.2, 0.15],
        }
    )
    fig = build_figure(
        df,
        y_metric="ari",
        x_axis="k",
        row_dim=None,
        col_dim=None,
        color_dim=None,
        agg_mode="Mean",
        connect_lines=False,
    )
    modes = {t.mode for t in fig.data}
    assert modes == {"markers"}


def test_build_figure_bars_returns_figure():
    long_df = pd.DataFrame(
        {
            "x_cat": ["1", "1", "2", "2"],
            "metric": ["ari", "sil", "ari", "sil"],
            "value": [0.1, 0.2, 0.3, 0.4],
            "y_err": [float("nan")] * 4,
        }
    )
    fig = build_figure_bars(long_df, "k", None, None, "Mean", plot_height=400)
    assert fig.layout.height == 400
    assert len(fig.data) >= 1


def test_create_sweep_explorer_perspective_ui_returns_viewable():
    from panel_app.views.sweep_explorer_perspective import create_sweep_explorer_perspective_ui

    ui = create_sweep_explorer_perspective_ui()
    assert ui is not None
