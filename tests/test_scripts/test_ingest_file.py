"""Tests for scripts/ingest_file.py."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from study_query_llm.algorithms.data_methods import register_data_methods
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, ProvenancedRun
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "ingest_file.py"


@pytest.fixture
def ingest_mod():
    spec = importlib.util.spec_from_file_location("ingest_file", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prepare_database(sqlite_path: Path) -> None:
    db_url = f"sqlite:///{sqlite_path}"
    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_data_methods(method_service)


def test_compute_sha256_matches_hashlib(
    ingest_mod,
    tmp_path: Path,
) -> None:
    payload_file = tmp_path / "sample.csv"
    payload_file.write_text("a,b\n1,2\n", encoding="utf-8")
    expected = hashlib.sha256(payload_file.read_bytes()).hexdigest()
    assert ingest_mod._compute_sha256(payload_file) == expected


def test_main_dry_run_computes_hash_and_performs_no_writes(
    ingest_mod,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sqlite_path = tmp_path / "dry_run.sqlite"
    _prepare_database(sqlite_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("ItemID,Prompt\n1,hello\n", encoding="utf-8")

    expected_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{sqlite_path}")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)

    rc = ingest_mod.main(
        [
            "--path",
            str(csv_path),
            "--name",
            "midterm2_questions",
            "--version",
            "v1",
            "--content-type",
            "text/csv",
            "--parse-as",
            "csv",
            "--expected-columns",
            "ItemID,Prompt",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run: no writes performed" in out
    assert f"expected_sha256={expected_sha}" in out
    assert "planned_calls:" in out
    assert "file_artifact.basic@0.1" in out
    assert "csv_parse.basic@0.1" in out

    db = DatabaseConnectionV2(f"sqlite:///{sqlite_path}", enable_pgvector=False)
    with db.session_scope() as session:
        assert int(session.query(ProvenancedRun).count()) == 0
        assert int(session.query(CallArtifact).count()) == 0


def test_main_chained_csv_parse_writes_source_and_imported_runs(
    ingest_mod,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sqlite_path = tmp_path / "ingest.sqlite"
    _prepare_database(sqlite_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "ItemID,Prompt,Correct\n1,What is 2+2?,4\n2,Capital of France?,Paris\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{sqlite_path}")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)

    rc = ingest_mod.main(
        [
            "--path",
            str(csv_path),
            "--name",
            "midterm2_questions",
            "--version",
            "v1",
            "--content-type",
            "text/csv",
            "--parse-as",
            "csv",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "source_run_id=" in out
    assert "imported_run_id=" in out
    assert "schema_summary=row_count=2,column_count=3" in out

    db = DatabaseConnectionV2(f"sqlite:///{sqlite_path}", enable_pgvector=False)
    with db.session_scope() as session:
        runs = session.query(ProvenancedRun).order_by(ProvenancedRun.id.asc()).all()
        assert len(runs) == 2
        roles = [(row.metadata_json or {}).get("pipeline_stage_role") for row in runs]
        assert roles == ["source", "imported"]
        imported_context = (runs[1].metadata_json or {}).get("pipeline_stage_context") or {}
        assert imported_context.get("dataset_name") == "midterm2_questions"
        assert imported_context.get("row_count") == 2
        assert imported_context.get("column_count") == 3
        assert int(session.query(CallArtifact).count()) == 2

