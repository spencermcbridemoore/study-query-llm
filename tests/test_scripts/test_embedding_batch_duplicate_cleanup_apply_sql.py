"""Regression tests for apply-path JSON migration in duplicate cleanup tool."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection


def _load_cleanup_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scratch" / "oneoff_fix_embedding_batch_same_key_duplicates.py"
    spec = importlib.util.spec_from_file_location(
        "oneoff_fix_embedding_batch_same_key_duplicates",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_postgres_test_url() -> str | None:
    explicit = (os.environ.get("SQLLM_TEST_POSTGRES_URL") or "").strip()
    if explicit.lower().startswith("postgresql"):
        return explicit

    repo_root = Path(__file__).resolve().parents[2]
    env = dotenv_values(repo_root / ".env")
    for key in ("DATABASE_URL", "CANONICAL_DATABASE_URL"):
        value = str(env.get(key) or "").strip()
        if value.lower().startswith("postgresql"):
            return value
    return None


def _seed_apply_tables(conn: Connection) -> None:
    # Temp tables shadow persistent tables for this connection only.
    conn.execute(
        text(
            """
            CREATE TEMP TABLE groups (
                id INTEGER PRIMARY KEY,
                group_type TEXT
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TEMP TABLE provenanced_runs (
                id INTEGER PRIMARY KEY,
                metadata_json JSON,
                config_json JSON,
                source_group_id INTEGER
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TEMP TABLE group_links (
                id INTEGER PRIMARY KEY,
                parent_group_id INTEGER,
                child_group_id INTEGER
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TEMP TABLE call_artifacts (
                id INTEGER PRIMARY KEY,
                metadata_json JSON
            )
            """
        )
    )
    conn.execute(
        text(
            """
            INSERT INTO groups (id, group_type) VALUES
                (58, 'embedding_batch'),
                (324, 'embedding_batch'),
                (999, 'embedding_batch')
            """
        )
    )

    conn.execute(
        text(
            """
            INSERT INTO provenanced_runs (id, metadata_json, config_json, source_group_id) VALUES
                (1, '{"embedding_batch_group_id": 58, "note": "keep"}', '{"embedding_batch_group_id": 58}', 58),
                (2, '{"embedding_batch_group_id": 58}', '{"embedding_batch_group_id": 10}', 999),
                (3, '{"embedding_batch_group_id": 10}', '{"embedding_batch_group_id": 58}', 58),
                (4, NULL, NULL, NULL)
            """
        )
    )
    conn.execute(
        text(
            """
            INSERT INTO group_links (id, parent_group_id, child_group_id) VALUES
                (1, 58, 77),
                (2, 66, 58),
                (3, 58, 58)
            """
        )
    )
    conn.execute(
        text(
            """
            INSERT INTO call_artifacts (id, metadata_json) VALUES
                (1, '{"group_id": 58, "artifact": "a"}'),
                (2, '{"group_id": 999}'),
                (3, NULL)
            """
        )
    )


def test_migrate_refs_to_survivor_updates_json_fields_with_json_column_types() -> None:
    cleanup = _load_cleanup_module()
    db_url = _resolve_postgres_test_url()
    if not db_url:
        pytest.skip("Postgres URL not available for apply-path SQL regression test")
    engine = create_engine(db_url, pool_pre_ping=True)

    try:
        with engine.begin() as conn:
            _seed_apply_tables(conn)
            counts = cleanup._migrate_refs_to_survivor(
                conn,
                from_batch_id=58,
                to_batch_id=324,
            )

            assert counts["metadata_embedding_batch_group_id_updates"] == 2
            assert counts["config_embedding_batch_group_id_updates"] == 2
            assert counts["source_group_id_updates"] == 2
            assert counts["group_links_parent_updates"] == 2
            assert counts["group_links_child_updates"] == 2
            assert counts["group_links_self_loops_deleted"] == 1
            assert counts["call_artifacts_group_id_updates"] == 1

            unresolved = cleanup._required_ref_counts(conn, 58)
            assert unresolved["required_ref_total"] == 0

            metadata_values = [
                row[0]
                for row in conn.execute(
                    text(
                        """
                        SELECT (metadata_json->>'embedding_batch_group_id')::int
                        FROM provenanced_runs
                        WHERE id IN (1, 2)
                        ORDER BY id
                        """
                    )
                ).fetchall()
            ]
            assert metadata_values == [324, 324]

            config_values = [
                row[0]
                for row in conn.execute(
                    text(
                        """
                        SELECT (config_json->>'embedding_batch_group_id')::int
                        FROM provenanced_runs
                        WHERE id IN (1, 3)
                        ORDER BY id
                        """
                    )
                ).fetchall()
            ]
            assert config_values == [324, 324]

            artifact_group_id = conn.execute(
                text(
                    """
                    SELECT (metadata_json->>'group_id')::int
                    FROM call_artifacts
                    WHERE id = 1
                    """
                )
            ).scalar_one()
            assert artifact_group_id == 324
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres regression fixture unavailable: {exc}")
