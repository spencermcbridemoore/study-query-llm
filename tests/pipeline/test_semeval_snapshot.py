"""Integration coverage for SemEval five-way snapshot parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.datasets.source_specs.semeval2013_sra_5way import (
    SEMEVAL2013_SRA_5WAY_SLUG,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SubquerySpec


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "semeval_snapshot.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def test_semeval_snapshot_default_parser(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    spec = ACQUIRE_REGISTRY[SEMEVAL2013_SRA_5WAY_SLUG]
    relative_payloads = {
        "README.md": b"# semeval mirror\n",
        "semevalFormatProcessing-5way/answers.csv": (
            "id1\tQuestion one\tReference one\n"
            "id1\tQuestion one\tReference one\tStudent answer one\n"
            "id1\tQuestion one\tReference one\n"
            "id2\tQuestion two\tReference two\n"
            "id3\tQuestion three\tReference three\tStudent answer three\n"
        ).encode("utf-8"),
        "semevalFormatProcessing-5way/trainingGold.txt": (
            "id\tqid\ttestSet\tmodule\tcount\taccuracy\n"
            "id1\tQ1\t\tScience\t1\tcorrect\n"
            "id2\tQ2\t\tScience\t2\tcontradictory\n"
        ).encode("utf-8"),
        "semevalFormatProcessing-5way/trainingGold-partial.txt": (
            "id\tqid\ttestSet\tmodule\tcount\taccuracy\n"
        ).encode("utf-8"),
        "semevalFormatProcessing-5way/testGold-UA.txt": (
            "id\tqid\ttestSet\tmodule\tcount\taccuracy\n"
            "id3\tQ3\tunseen-answers\tScience\t3\tpartially_correct_incomplete\n"
        ).encode("utf-8"),
        "semevalFormatProcessing-5way/testGold-UQ.txt": (
            "id\tqid\ttestSet\tmodule\tcount\taccuracy\n"
        ).encode("utf-8"),
        "semevalFormatProcessing-5way/partialEntailmentGold.txt": (
            "id\tqid\ttestSet\tmodule\tcount\taccuracy\n"
        ).encode("utf-8"),
    }
    payload_by_url = {
        file_spec.url: relative_payloads[file_spec.relative_path] for file_spec in spec.file_specs()
    }

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
    snapshot_labeled = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="labeled"),
        db=db,
        artifact_dir=artifact_dir,
    )
    snapshot_unlabeled = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="unlabeled"),
        db=db,
        artifact_dir=artifact_dir,
    )

    assert parsed.group_id == parsed_reuse.group_id
    assert parsed_reuse.metadata["reused"] is True
    assert snapshot_all.metadata["row_count"] == 3
    assert snapshot_labeled.metadata["row_count"] == 3
    assert snapshot_unlabeled.metadata["row_count"] == 0

    table = pq.read_table(parsed.artifact_uris["dataframe.parquet"])
    assert table.column("source_id").to_pylist() == ["id1", "id2", "id3"]
    assert table.column("text").to_pylist() == [
        "Student answer one",
        "Reference two",
        "Student answer three",
    ]
    assert table.column("label_name").to_pylist() == [
        "correct",
        "contradictory",
        "partially_correct_incomplete",
    ]
    extras = [json.loads(value) for value in table.column("extra_json").to_pylist()]
    assert extras[0]["answer_row_count"] == 3
    assert extras[0]["student_answer_count"] == 1
    assert extras[0]["gold_file"] == "trainingGold.txt"
    assert extras[1]["student_answer_count"] == 0
    assert extras[2]["test_set"] == "unseen-answers"

    with db.session_scope() as session:
        dataframe_group = session.query(Group).filter(Group.id == parsed.group_id).first()
        assert dataframe_group is not None
        metadata = dict(dataframe_group.metadata_json or {})
        assert metadata["dataset_slug"] == SEMEVAL2013_SRA_5WAY_SLUG
        assert metadata["parser_id"] == "semeval2013_sra_5way.default"
        assert metadata["parser_version"] == "v1"
