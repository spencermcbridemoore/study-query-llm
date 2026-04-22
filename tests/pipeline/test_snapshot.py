"""Tests for pipeline.snapshot stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "snapshot.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _fixture_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [
            FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv"),
            FileFetchSpec(relative_path="data/test.csv", url="https://example.test/test.csv"),
        ]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {"dataset": "fixture_dataset", "revision": "abc123"},
        }

    return DatasetAcquireConfig(
        slug="fixture_dataset",
        file_specs=file_specs,
        source_metadata=source_metadata,
        default_parser=_fixture_parser,
        default_parser_id="fixture_dataset.default",
        default_parser_version="v1",
    )


def _fixture_parser(ctx) -> list[SnapshotRow]:
    assert (ctx.artifact_dir_local / "data" / "train.csv").is_file()
    assert (ctx.artifact_dir_local / "data" / "test.csv").is_file()
    return [
        SnapshotRow(position=0, source_id="row-0", text="hello", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="row-1", text="world", label=1, label_name="b"),
        SnapshotRow(position=2, source_id="row-2", text="again", label=1, label_name="b"),
    ]


def test_subquery_spec_canonical_dict_omits_category_filter_when_unset() -> None:
    spec = SubquerySpec()
    canonical = spec.to_canonical_dict()
    assert "category_filter" not in canonical


def test_subquery_spec_rejects_empty_category_filter() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SubquerySpec(category_filter={})


def test_subquery_spec_rejects_empty_inner_list() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        SubquerySpec(category_filter={"qid": []})


def test_subquery_spec_rejects_non_scalar_candidate() -> None:
    with pytest.raises(ValueError, match="str/int/float/bool/None"):
        SubquerySpec(category_filter={"qid": [{"nested": 1}]})


def test_subquery_spec_canonical_dict_normalizes_category_filter() -> None:
    spec = SubquerySpec(category_filter={"qid": ["Q2", "Q1", "Q1"], "module": ["LF"]})
    assert spec.to_canonical_dict()["category_filter"] == {
        "module": ["LF"],
        "qid": ["Q1", "Q2"],
    }


def test_subquery_spec_freezes_nested_category_filter() -> None:
    raw_filter = {"qid": ["Q2", "Q1"]}
    spec = SubquerySpec(category_filter=raw_filter)
    raw_filter["qid"].append("Q3")

    assert tuple(spec.category_filter["qid"]) == ("Q1", "Q2")
    with pytest.raises(TypeError):
        spec.category_filter["qid"] = ("Q4",)


def test_snapshot_writes_parquet_manifest_and_depends_on_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    spec = _fixture_spec()
    payload_by_url = {
        "https://example.test/train.csv": b"id,text\n1,hello\n2,world\n",
        "https://example.test/test.csv": b"id,text\n3,test\n",
    }
    artifact_dir = str((tmp_path / "artifacts").resolve())

    dataset_result = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda url: payload_by_url[url],
    )
    parsed = parse(
        dataset_result.group_id,
        parser=_fixture_parser,
        parser_id="fixture_dataset.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    first = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    second = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )

    assert first.group_id == second.group_id
    payload_uri = first.artifact_uris["subquery_spec.json"]
    payload = json.loads(Path(payload_uri).read_text(encoding="utf-8"))
    assert payload["row_count"] == 3
    assert payload["resolved_index"] == [[0, "row-0"], [1, "row-1"], [2, "row-2"]]


def test_snapshot_sampling_requires_seed_and_changes_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    spec = _fixture_spec()
    payload_by_url = {
        "https://example.test/train.csv": b"id,text\n1,hello\n2,world\n",
        "https://example.test/test.csv": b"id,text\n3,test\n",
    }
    artifact_dir = str((tmp_path / "artifacts").resolve())
    dataset_result = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda url: payload_by_url[url],
    )
    parsed = parse(
        dataset_result.group_id,
        parser=_fixture_parser,
        parser_id="fixture_dataset.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    seeded_a = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(sample_n=2, sampling_seed=17, label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    seeded_b = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(sample_n=2, sampling_seed=17, label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    seeded_c = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(sample_n=2, sampling_seed=19, label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )

    assert seeded_a.group_id == seeded_b.group_id
    assert seeded_a.group_id != seeded_c.group_id

    try:
        snapshot(
            parsed.group_id,
            subquery_spec=SubquerySpec(sample_n=2, label_mode="all"),
            db=db,
            artifact_dir=artifact_dir,
        )
    except ValueError as exc:
        assert "sampling_seed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unseeded sampling to fail")
