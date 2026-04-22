"""Integration coverage for Estela pickle acquisition and parsing."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.estela import (
    ESTELA_DATASET_SLUG,
    ESTELA_PICKLE_RELATIVE_PATH,
)
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SubquerySpec


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "estela_snapshot.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _pickle_bytes(payload: object) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def test_estela_snapshot_default_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    spec = ACQUIRE_REGISTRY[ESTELA_DATASET_SLUG]
    payload = _pickle_bytes(
        {
            "record_a": {"prompt": "This is the first uncategorized Estela prompt."},
            "record_b": {"nested": {"Prompt": "Second uncategorized prompt goes here."}},
            "record_c": [{"user_prompt": "  Third prompt with \x00 null byte  "}],
            "record_d": {"prompt": "short"},
            "record_e": {"not_prompt": "ignored"},
        }
    )
    payload_by_url = {file_spec.url: payload for file_spec in spec.file_specs()}

    acquired = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda url: payload_by_url[url],
    )
    parsed = parse(acquired.group_id, db=db, artifact_dir=artifact_dir)
    parsed_reuse = parse(acquired.group_id, db=db, artifact_dir=artifact_dir)
    snapshot_all = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    snapshot_unlabeled = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="unlabeled"),
        db=db,
        artifact_dir=artifact_dir,
    )
    snapshot_labeled = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="labeled"),
        db=db,
        artifact_dir=artifact_dir,
    )

    assert parsed.group_id == parsed_reuse.group_id
    assert parsed_reuse.metadata["reused"] is True
    assert snapshot_all.metadata["row_count"] == 3
    assert snapshot_unlabeled.metadata["row_count"] == 3
    assert snapshot_labeled.metadata["row_count"] == 0
    assert snapshot_all.group_id != snapshot_unlabeled.group_id

    table = pq.read_table(parsed.artifact_uris["dataframe.parquet"])
    labels = table.column("label").to_pylist()
    extras = [json.loads(value) for value in table.column("extra_json").to_pylist()]
    assert labels == [None, None, None]
    assert {extra["subset_profile"] for extra in extras} == {"all_uncategorized"}
    assert {extra["source_file"] for extra in extras} == {ESTELA_PICKLE_RELATIVE_PATH}

    with db.session_scope() as session:
        dataframe_group = session.query(Group).filter(Group.id == parsed.group_id).first()
        assert dataframe_group is not None
        metadata = dict(dataframe_group.metadata_json or {})
        assert metadata["dataset_slug"] == ESTELA_DATASET_SLUG
        assert metadata["parser_id"] == "estela.default"
        assert metadata["parser_version"] == "v1"
