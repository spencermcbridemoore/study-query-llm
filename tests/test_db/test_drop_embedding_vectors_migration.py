"""Tests for dropping legacy embedding_vectors table migration."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.migrations.drop_embedding_vectors import (
    drop_embedding_vectors_table,
)


def test_drop_embedding_vectors_table_is_unconditional(tmp_path: Path) -> None:
    db_path = (tmp_path / "drop_embedding_vectors.sqlite3").resolve()
    database_url = f"sqlite:///{db_path.as_posix()}"
    db = DatabaseConnectionV2(database_url, enable_pgvector=False)
    db.init_db()

    with db.engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE embedding_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id INTEGER NOT NULL,
                    vector TEXT,
                    dimension INTEGER NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO embedding_vectors (call_id, vector, dimension) VALUES (1, '[0.1,0.2]', 2)"
            )
        )

    first_drop_count = drop_embedding_vectors_table(database_url)
    assert first_drop_count == 1
    assert inspect(db.engine).has_table("embedding_vectors") is False

    second_drop_count = drop_embedding_vectors_table(database_url)
    assert second_drop_count == 0
