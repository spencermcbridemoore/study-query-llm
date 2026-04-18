"""Structural acceptance test for BANKING77 pipeline stages."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.banking77 import (
    BANKING77_DATASET_SLUG,
    banking77_file_specs,
)
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import AnalysisResult, Group, GroupLink
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.analyze import analyze
from study_query_llm.pipeline.embed import embed
from study_query_llm.pipeline.snapshot import snapshot


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "bank77_e2e.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _parquet_bytes(rows: list[dict[str, object]]) -> bytes:
    table = pa.table(
        {
            "text": [str(row["text"]) for row in rows],
            "label": [int(row["label"]) for row in rows],
            "label_text": [str(row["label_text"]) for row in rows],
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def test_bank77_pipeline_chain_structural_acceptance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    spec = ACQUIRE_REGISTRY[BANKING77_DATASET_SLUG]

    fixture_rows = {
        "data/train-00000-of-00001.parquet": [
            {"text": "transfer failed", "label": 0, "label_text": "transfer_issue"},
            {"text": "cash withdrawal not recognised", "label": 1, "label_text": "cash_issue"},
        ],
        "data/test-00000-of-00001.parquet": [
            {"text": "beneficiary blocked", "label": 0, "label_text": "transfer_issue"},
            {"text": "card payment pending", "label": 1, "label_text": "cash_issue"},
        ],
    }
    payload_by_url = {}
    for file_spec in banking77_file_specs():
        payload_by_url[file_spec.url] = _parquet_bytes(
            fixture_rows[file_spec.relative_path]
        )

    acquired = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda url: payload_by_url[url],
    )
    snapped = snapshot(
        acquired.group_id,
        db=db,
        artifact_dir=artifact_dir,
    )

    def fake_embedding_fetcher(**kwargs):
        texts = kwargs["texts"]
        return np.asarray(
            [[float(idx), float(len(text))] for idx, text in enumerate(texts)],
            dtype=np.float64,
        )

    embedded = embed(
        snapped.group_id,
        deployment="fixture-embedding-model",
        provider="fixture-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_embedding_fetcher,
    )

    def fake_method_runner(**kwargs):
        embeddings = np.asarray(kwargs["embeddings"], dtype=np.float64)
        return {
            "scalar_results": {
                "row_count": float(embeddings.shape[0]),
                "dimension": float(embeddings.shape[1]),
            },
            "structured_results": {
                "labels_present": [0, 1],
            },
            "artifacts": {
                "bank77_acceptance.json": b'{"status":"ok"}',
            },
            "result_ref": "bank77_acceptance.json",
        }

    analyzed = analyze(
        embedded.group_id,
        method_name="bank77_structural_acceptance",
        run_key="fixture_bank77_run",
        db=db,
        artifact_dir=artifact_dir,
        method_runner=fake_method_runner,
    )

    assert "data/train-00000-of-00001.parquet" in acquired.artifact_uris
    assert "data/test-00000-of-00001.parquet" in acquired.artifact_uris
    assert snapped.metadata["row_count"] == 4
    assert snapped.metadata["label_count"] == 2
    assert embedded.metadata["row_count"] == 4
    assert embedded.metadata["dimension"] == 2
    assert analyzed.run_id is not None
    assert "bank77_acceptance.json" in analyzed.artifact_uris

    with db.session_scope() as session:
        run_row_group = session.query(Group).filter(Group.id == analyzed.group_id).first()
        assert run_row_group is not None
        assert run_row_group.group_type == "analysis_run"

        request_groups = (
            session.query(Group)
            .filter(Group.group_type == "analysis_request")
            .all()
        )
        assert len(request_groups) == 1
        request_group_id = int(request_groups[0].id)

        contains = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == request_group_id,
                GroupLink.child_group_id == analyzed.group_id,
                GroupLink.link_type == "contains",
            )
            .first()
        )
        assert contains is not None

        depends_on = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == analyzed.group_id,
                GroupLink.child_group_id == embedded.group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert depends_on is not None

        analysis_result_count = (
            session.query(AnalysisResult)
            .filter(AnalysisResult.analysis_group_id == analyzed.group_id)
            .count()
        )
        assert analysis_result_count >= 3
