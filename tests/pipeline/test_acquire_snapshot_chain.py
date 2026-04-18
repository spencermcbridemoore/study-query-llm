"""End-to-end fixture test for acquire -> snapshot chain."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow

REPO = Path(__file__).resolve().parent.parent.parent
LINT_SCRIPT = REPO / "scripts" / "check_persistence_contract.py"


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "chain.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _chain_parser(_ctx) -> list[SnapshotRow]:
    return [
        SnapshotRow(position=0, source_id="id-0", text="row zero", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="id-1", text="row one", label=1, label_name="b"),
    ]


def _chain_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [
            FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv"),
        ]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {"dataset": "chain_fixture", "revision": "r1"},
        }

    return DatasetAcquireConfig(
        slug="chain_fixture",
        file_specs=file_specs,
        source_metadata=source_metadata,
        default_parser=_chain_parser,
    )


def _load_lint_module():
    spec = importlib.util.spec_from_file_location("check_persistence_contract", LINT_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_acquire_snapshot_chain_and_lint_clean(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    source_bytes = {"https://example.test/train.csv": b"id,text\n1,hello\n"}
    spec = _chain_spec()

    acquired = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda url: source_bytes[url],
    )
    snapped = snapshot(
        acquired.group_id,
        parser=_chain_parser,
        db=db,
        artifact_dir=artifact_dir,
    )

    assert acquired.group_id > 0
    assert snapped.group_id > 0
    assert "snapshot.parquet" in snapped.artifact_uris

    lint_mod = _load_lint_module()
    violations = lint_mod.lint_pipeline_dir(REPO / "src" / "study_query_llm" / "pipeline")
    assert violations == []
