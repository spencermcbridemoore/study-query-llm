"""Integration coverage for sources_uncertainty_qc snapshot parsing."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.datasets.source_specs.sources_uncertainty_zenodo import (
    SOURCES_UNCERTAINTY_QC_SLUG,
    parse_sources_uncertainty_pm_snapshot,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.snapshot import snapshot


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "sources_uncertainty_snapshot.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _xlsx_bytes(records: list[dict[str, object]]) -> bytes:
    frame = pd.DataFrame(records)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
    return buf.getvalue()


def test_sources_uncertainty_snapshot_default_and_pm_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    spec = ACQUIRE_REGISTRY[SOURCES_UNCERTAINTY_QC_SLUG]
    payload = _xlsx_bytes(
        [
            {
                "ResponseId": "R-1",
                "response": "local variability source",
                "Experiment": "PM",
                "code": "L",
                "updated_code": "L",
            },
            {
                "ResponseId": "R-2",
                "response": "observer introduces uncertainty",
                "Experiment": "PM",
                "code": "O",
                "updated_code": "O",
            },
            {
                "ResponseId": "R-3",
                "response": "procedural difference across groups",
                "Experiment": "BM",
                "code": "P",
                "updated_code": "P",
            },
            {
                "ResponseId": "R-4",
                "response": "systematic bias in setup",
                "Experiment": "SG",
                "code": "S",
                "updated_code": "S",
            },
            {
                "ResponseId": "R-5",
                "response": "pm row recoded to systematic",
                "Experiment": "PM",
                "code": "L",
                "updated_code": "S",
            },
        ]
    )

    acquired = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda _url: payload,
    )

    full = snapshot(
        acquired.group_id,
        representation="raw_all_v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    full_reuse = snapshot(
        acquired.group_id,
        representation="raw_all_v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    pm_only = snapshot(
        acquired.group_id,
        parser=parse_sources_uncertainty_pm_snapshot,
        representation="raw_exp_pm_v1",
        db=db,
        artifact_dir=artifact_dir,
    )

    assert full.group_id == full_reuse.group_id
    assert full_reuse.metadata["reused"] is True
    assert full.metadata["row_count"] == 5
    assert full.metadata["label_count"] == 4
    assert pm_only.metadata["row_count"] == 3
    assert pm_only.metadata["label_count"] == 3
    assert pm_only.group_id != full.group_id

    full_table = pq.read_table(full.artifact_uris["snapshot.parquet"])
    pm_table = pq.read_table(pm_only.artifact_uris["snapshot.parquet"])
    full_extras = [
        json.loads(value)
        for value in full_table.column("extra_json").to_pylist()
    ]
    pm_extras = [json.loads(value) for value in pm_table.column("extra_json").to_pylist()]
    assert {extra["subset_profile"] for extra in full_extras} == {"all"}
    assert {extra["subset_profile"] for extra in pm_extras} == {"experiment=PM"}
    assert {extra["experiment"] for extra in pm_extras} == {"PM"}

    with db.session_scope() as session:
        full_group = session.query(Group).filter(Group.id == full.group_id).first()
        pm_group = session.query(Group).filter(Group.id == pm_only.group_id).first()
        assert full_group is not None
        assert pm_group is not None
        full_md = dict(full_group.metadata_json or {})
        pm_md = dict(pm_group.metadata_json or {})
        assert full_md["representation"] == "raw_all_v1"
        assert pm_md["representation"] == "raw_exp_pm_v1"
        assert full_md["parser_identity"].endswith("parse_sources_uncertainty_snapshot")
        assert pm_md["parser_identity"].endswith("parse_sources_uncertainty_pm_snapshot")
