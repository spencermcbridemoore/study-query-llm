"""Integration coverage for AuSeM snapshot parser profiles."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.ausem import (
    AUSEM_DATASET_SLUG,
    parse_ausem_problem2_snapshot,
)
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SubquerySpec


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "ausem_snapshot.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _csv_bytes(records: list[dict[str, object]]) -> bytes:
    frame = pd.DataFrame(records)
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_ausem_snapshot_default_and_problem_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    spec = ACQUIRE_REGISTRY[AUSEM_DATASET_SLUG]
    relative_payloads = {
        "Student_Explanations/problem1.csv": _csv_bytes(
            [
                {"correct": True, "student_id": "Student 1", "Text": "problem1-a"},
                {"correct": False, "student_id": "Student 2", "Text": "problem1-b"},
            ]
        ),
        "Student_Explanations/problem2.csv": _csv_bytes(
            [{"correct": True, "student_id": "Student 3", "Text": "problem2-a"}]
        ),
        "Student_Explanations/problem3.csv": _csv_bytes(
            [{"correct": False, "student_id": "Student 4", "Text": "problem3-a"}]
        ),
        "Student_Explanations/problem4.csv": _csv_bytes(
            [{"correct": True, "student_id": "Student 5", "Text": "problem4-a"}]
        ),
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

    full_df = parse(
        acquired.group_id,
        db=db,
        artifact_dir=artifact_dir,
    )
    full_df_reuse = parse(
        acquired.group_id,
        db=db,
        artifact_dir=artifact_dir,
    )
    problem2_df = parse(
        acquired.group_id,
        parser=parse_ausem_problem2_snapshot,
        parser_id="ausem.problem2",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    full = snapshot(
        full_df.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    full_reuse = snapshot(
        full_df.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    problem2 = snapshot(
        problem2_df.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )

    assert full_df.group_id == full_df_reuse.group_id
    assert full.group_id == full_reuse.group_id
    assert full_reuse.metadata["reused"] is True
    assert full.metadata["row_count"] == 5
    assert problem2.metadata["row_count"] == 1
    assert problem2.group_id != full.group_id

    full_table = pq.read_table(full_df.artifact_uris["dataframe.parquet"])
    problem2_table = pq.read_table(problem2_df.artifact_uris["dataframe.parquet"])
    full_extras = [json.loads(value) for value in full_table.column("extra_json").to_pylist()]
    problem2_extras = [json.loads(value) for value in problem2_table.column("extra_json").to_pylist()]
    assert {extra["subset_profile"] for extra in full_extras} == {"all"}
    assert {extra["problem"] for extra in full_extras} == {
        "problem1",
        "problem2",
        "problem3",
        "problem4",
    }
    assert {extra["subset_profile"] for extra in problem2_extras} == {"problem=2"}
    assert {extra["problem"] for extra in problem2_extras} == {"problem2"}

    with db.session_scope() as session:
        full_group = session.query(Group).filter(Group.id == full_df.group_id).first()
        problem2_group = session.query(Group).filter(Group.id == problem2_df.group_id).first()
        assert full_group is not None
        assert problem2_group is not None
        full_md = dict(full_group.metadata_json or {})
        problem2_md = dict(problem2_group.metadata_json or {})
        assert full_md["parser_id"] == "ausem.default"
        assert full_md["parser_version"] == "v1"
        assert problem2_md["parser_id"] == "ausem.problem2"
        assert problem2_md["parser_version"] == "v1"
