"""Tests for sweep selection-curve CSV export helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from study_query_llm.experiments.selection_curve_export import (
    _selection_curve_files,
    default_sweep_method_names,
    iter_selection_curve_metric_rows,
    write_metrics_csv,
)


def test_default_sweep_methods_are_registry_sweep_select() -> None:
    names = default_sweep_method_names()
    assert "kmeans+normalize+pca+sweep" in names
    assert "gmm+normalize+pca+sweep" in names
    assert all("sweep" in n for n in names)


def test_selection_curve_files_filters_suffix() -> None:
    uris = {
        "kmeans_selection_curve.json": "file:///a.json",
        "gmm_summary.json": "file:///b.json",
        "gmm_selection_curve.json": "file:///c.json",
    }
    found = _selection_curve_files(uris)
    assert found == [
        ("gmm_selection_curve.json", "file:///c.json"),
        ("kmeans_selection_curve.json", "file:///a.json"),
    ]


def test_iter_selection_curve_metric_rows_expands_points() -> None:
    session = MagicMock()
    ar = MagicMock()
    ar.id = 10
    ar.analysis_group_id = 100
    ar.result_json = {
        "uris": {
            "kmeans_selection_curve.json": "file:///x.json",
        }
    }
    md = MagicMock()
    md.name = "kmeans+normalize+pca+sweep"
    md.version = "1.0"

    pairs_all = MagicMock()
    pairs_all.all.return_value = [(ar, md)]
    pairs_ob = MagicMock()
    pairs_ob.order_by.return_value = pairs_all
    pairs_f_method = MagicMock()
    pairs_f_method.filter.return_value = pairs_ob
    pairs_f_active = MagicMock()
    pairs_f_active.filter.return_value = pairs_f_method
    pairs_f_key = MagicMock()
    pairs_f_key.filter.return_value = pairs_f_active
    pairs_query_root = MagicMock()
    pairs_query_root.join.return_value = pairs_f_key

    prov_all = MagicMock()
    prov_all.all.return_value = []
    prov_ob = MagicMock()
    prov_ob.order_by.return_value = prov_all
    prov_f = MagicMock()
    prov_f.filter.return_value = prov_ob
    prov_root = MagicMock()
    prov_root.filter.return_value = prov_ob

    def query_side_effect(*args, **kwargs):
        return pairs_query_root if len(args) == 2 else prov_root

    session.query.side_effect = query_side_effect

    svc = MagicMock()
    svc.storage.read_from_uri.return_value = (
        b'{"metric":"silhouette","selection_rule":"kneedle","chosen_k":5,'
        b'"points":[{"k":2,"silhouette":0.1},{"k":5,"silhouette":0.4}]}'
    )

    rows = list(
        iter_selection_curve_metric_rows(
            session,
            svc,
            method_names=("kmeans+normalize+pca+sweep",),
        )
    )
    assert len(rows) == 2
    assert rows[0]["candidate_k"] == 2
    assert rows[0]["silhouette"] == 0.1
    assert rows[0]["chosen_k"] == 5
    assert rows[1]["candidate_k"] == 5


def test_write_metrics_csv_roundtrip_columns(tmp_path) -> None:
    p = tmp_path / "out.csv"
    n = write_metrics_csv([{"a": 1, "b": 2}, {"b": 3, "c": 4}], str(p))
    assert n == 2
    text = p.read_text(encoding="utf-8")
    assert "a" in text and "b" in text and "c" in text
