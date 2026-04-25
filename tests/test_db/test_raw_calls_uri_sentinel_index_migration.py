"""Tests for raw_calls URI sentinel index migration helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.db.migrations.add_raw_calls_uri_sentinel_index import (
    _render_create_index_sql,
    _validate_index_name,
    add_raw_calls_uri_sentinel_index,
)


def test_render_create_index_sql_contains_partial_non_blob_predicate() -> None:
    sql = _render_create_index_sql(index_name="idx_raw_calls_uri_non_blob_sentinel")
    assert "CREATE INDEX IF NOT EXISTS idx_raw_calls_uri_non_blob_sentinel" in sql
    assert "ON raw_calls" in sql
    assert "response_json::jsonb ->> 'uri'" in sql
    assert "!~*" in sql
    assert "blob\\.core\\.windows\\.net" in sql


def test_validate_index_name_rejects_invalid_names() -> None:
    assert _validate_index_name("idx_raw_calls_uri_non_blob_sentinel") == (
        "idx_raw_calls_uri_non_blob_sentinel"
    )
    with pytest.raises(ValueError):
        _validate_index_name("bad-name-with-dashes")
    with pytest.raises(ValueError):
        _validate_index_name("1starts_with_digit")


def test_add_raw_calls_uri_sentinel_index_is_noop_on_sqlite(tmp_path: Path) -> None:
    db_path = (tmp_path / "sentinel.sqlite3").resolve()
    database_url = f"sqlite:///{db_path.as_posix()}"
    changed = add_raw_calls_uri_sentinel_index(database_url=database_url)
    assert changed == 0
