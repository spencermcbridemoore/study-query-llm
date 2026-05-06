"""Tests for registry expansion + backfill planning helpers."""

from __future__ import annotations

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.experiments.clustering_analysis_backfill import (
    build_all_targets,
    build_manifest,
    expand_parameter_variants,
    preflight_manifest_blocking_issues,
    validate_parameters_for_method,
    validate_registry_expansion_or_raise,
)
from study_query_llm.pipeline.clustering.registry import get_algorithm_spec, iter_algorithm_specs


def test_validate_registry_expansion_or_raise_covers_all_specs() -> None:
    validate_registry_expansion_or_raise(n_samples=300, embedding_dim=1536)


def test_expand_fixed_k_rejects_k_equals_one() -> None:
    spec = get_algorithm_spec("kmeans+fixed-k")
    assert spec is not None
    with pytest.raises(ValueError, match="k=1"):
        expand_parameter_variants(
            spec,
            n_samples=100,
            embedding_dim=64,
            fixed_k_range=range(1, 4),
        )


def test_sweep_k_range_includes_one_without_schema_error() -> None:
    spec = get_algorithm_spec("kmeans+normalize+pca+sweep")
    assert spec is not None
    out = expand_parameter_variants(spec, n_samples=200, embedding_dim=64)
    assert len(out) == 1
    assert 1 in out[0]["k_range"]


def test_validate_parameters_rejects_unknown_keys() -> None:
    spec = get_algorithm_spec("kmeans+fixed-k")
    assert spec is not None
    schema = dict(spec.parameters_schema or {})
    with pytest.raises(ValueError, match="unknown_clustering_method_parameters"):
        validate_parameters_for_method(
            "kmeans+fixed-k",
            {"k": 3, "bogus": 1},
            schema=schema,
        )


def test_build_manifest_halts_on_batch_collision() -> None:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        sdf = Group(
            group_type="dataset_dataframe",
            name="df",
            metadata_json={"row_count": 50},
        )
        session.add(sdf)
        session.flush()

        snap = Group(
            group_type="dataset_snapshot",
            name="snap",
            metadata_json={
                "source_dataframe_group_id": int(sdf.id),
                "row_count": 50,
            },
        )
        session.add(snap)
        session.flush()

        for gid_suffix in ("a", "b"):
            batch = Group(
                group_type="embedding_batch",
                name=f"emb_{gid_suffix}",
                metadata_json={
                    "source_dataframe_group_id": int(sdf.id),
                    "entry_max": 50,
                    "embedding_engine": "text-embedding-3-small",
                    "provider": "openai",
                },
            )
            session.add(batch)
        session.flush()

        manifest = build_manifest(
            session,
            embedding_engine="text-embedding-3-small",
            snapshot_ids=[int(snap.id)],
        )

    issues = preflight_manifest_blocking_issues(manifest)
    assert issues
    assert manifest["collisions"]


def test_build_all_targets_unique_analysis_keys() -> None:
    targets = build_all_targets(
        snapshot_group_id=9,
        embedding_batch_group_id=18,
        n_samples=120,
        embedding_dim=256,
        specs=iter_algorithm_specs(),
    )
    keys = [t.analysis_run_key for t in targets]
    assert len(keys) == len(set(keys))

