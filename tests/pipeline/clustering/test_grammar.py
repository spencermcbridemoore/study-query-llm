"""Tests for bundled clustering method-name grammar."""

from __future__ import annotations

import pytest

from study_query_llm.pipeline.clustering.grammar import parse_method_name

CANONICAL_PARSE_CASES: list[tuple[str, str, tuple[str, ...], str]] = [
    ("kmeans+fixed-k", "kmeans", (), "fixed-k"),
    ("kmeans+normalize+fixed-k", "kmeans", ("normalize",), "fixed-k"),
    ("kmeans+pca+fixed-k", "kmeans", ("pca",), "fixed-k"),
    ("kmeans+normalize+pca+fixed-k", "kmeans", ("normalize", "pca"), "fixed-k"),
    ("spherical-kmeans+approx+fixed-k", "spherical-kmeans", ("normalize",), "fixed-k"),
    (
        "spherical-kmeans+approx+pca+fixed-k",
        "spherical-kmeans",
        ("normalize", "pca"),
        "fixed-k",
    ),
    ("gmm+fixed-k", "gmm", (), "fixed-k"),
    ("gmm+normalize+fixed-k", "gmm", ("normalize",), "fixed-k"),
    ("gmm+pca+fixed-k", "gmm", ("pca",), "fixed-k"),
    ("gmm+normalize+pca+fixed-k", "gmm", ("normalize", "pca"), "fixed-k"),
    ("hdbscan+normalize+fixed", "hdbscan", ("normalize",), "fixed"),
    ("hdbscan+pca+fixed", "hdbscan", ("pca",), "fixed"),
    ("hdbscan+normalize+pca+fixed", "hdbscan", ("normalize", "pca"), "fixed"),
    ("agglomerative+normalize+fixed-k", "agglomerative", ("normalize",), "fixed-k"),
    ("agglomerative+pca+fixed-k", "agglomerative", ("pca",), "fixed-k"),
    ("dbscan+fixed-eps", "dbscan", (), "fixed-eps"),
    ("dbscan+normalize+fixed-eps", "dbscan", ("normalize",), "fixed-eps"),
    ("dbscan+pca+fixed-eps", "dbscan", ("pca",), "fixed-eps"),
    ("dbscan+normalize+pca+fixed-eps", "dbscan", ("normalize", "pca"), "fixed-eps"),
    ("kmeans+normalize+pca+sweep", "kmeans", ("normalize", "pca"), "sweep"),
    ("gmm+normalize+pca+sweep", "gmm", ("normalize", "pca"), "sweep"),
    ("hdbscan+fixed", "hdbscan", (), "fixed"),
    ("agglomerative+fixed-k", "agglomerative", (), "fixed-k"),
]

@pytest.mark.parametrize(
    ("name", "expected_base", "expected_chain", "expected_fit"),
    CANONICAL_PARSE_CASES,
)
def test_parse_method_name_canonical_table(
    name: str,
    expected_base: str,
    expected_chain: tuple[str, ...],
    expected_fit: str,
) -> None:
    base, chain, fit_tok = parse_method_name(name)
    assert base == expected_base
    assert chain == expected_chain
    assert fit_tok == expected_fit


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "+",
        "onlyone",
        "kmeans+oops+fixed-k",
        "kmeans+normalize+pca",
        "dbscan+fixed-k",
        "kmeans+normalize+pca+fixed-eps",
    ],
)
def test_parse_method_name_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_method_name(bad)
