"""Tests for validate_and_backfill_run_snapshots script."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import ProvenanceService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _seed_db(sqlite_path: str) -> int:
    db = DatabaseConnectionV2(f"sqlite:///{sqlite_path}", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        prov = ProvenanceService(repo)
        prov.create_dataset_snapshot_group(
            snapshot_name="dbpedia_286_seed42_labeled",
            source_dataset="dbpedia",
            sample_size=286,
            label_mode="labeled",
            sampling_method="seeded_random_filtered_10_1000_chars",
            sampling_seed=42,
        )
        run_id = repo.create_group(
            group_type="clustering_run",
            name="run_missing_snapshot",
            metadata_json={
                "dataset": "dbpedia",
                "n_samples": 286,
                "run_key": "dbpedia_engine_sum_286_50runs",
            },
        )
        return int(run_id)


def test_validate_script_dry_run_and_apply():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        run_id = _seed_db(db_path)
        env = os.environ.copy()
        env["DATABASE_URL"] = f"sqlite:///{db_path}"

        dry_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validate_and_backfill_run_snapshots.py"),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert dry_run.returncode == 0
        assert "backfilled=0 (dry-run; use --apply)" in dry_run.stdout

        apply_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validate_and_backfill_run_snapshots.py"),
                "--apply",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert apply_run.returncode == 0
        assert "backfilled=1" in apply_run.stdout

        db = DatabaseConnectionV2(f"sqlite:///{db_path}", enable_pgvector=False)
        db.init_db()
        with db.session_scope() as session:
            run = session.query(Group).filter(Group.id == run_id).first()
            assert run is not None
            meta = run.metadata_json or {}
            assert "dataset_snapshot_ids" in meta
            assert len(meta["dataset_snapshot_ids"]) == 1
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
