"""Tests for call_artifacts blob-URI check migration helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.db.migrations.add_call_artifacts_blob_uri_check import (
    _render_add_constraint_sql,
    _validate_constraint_name,
    add_call_artifacts_blob_uri_check,
)


def test_render_add_constraint_sql_contains_not_valid_and_blob_regex() -> None:
    sql = _render_add_constraint_sql(constraint_name="call_artifacts_uri_must_be_blob")
    assert "ALTER TABLE call_artifacts" in sql
    assert "CHECK (uri ~*" in sql
    assert "blob\\.core\\.windows\\.net" in sql
    assert "NOT VALID" in sql


def test_validate_constraint_name_rejects_invalid_names() -> None:
    assert _validate_constraint_name("call_artifacts_uri_must_be_blob") == (
        "call_artifacts_uri_must_be_blob"
    )
    with pytest.raises(ValueError):
        _validate_constraint_name("bad-name-with-dashes")
    with pytest.raises(ValueError):
        _validate_constraint_name("1starts_with_digit")


def test_add_call_artifacts_blob_uri_check_is_noop_on_sqlite(tmp_path: Path) -> None:
    db_path = (tmp_path / "blob_constraint.sqlite3").resolve()
    database_url = f"sqlite:///{db_path.as_posix()}"
    changed = add_call_artifacts_blob_uri_check(database_url=database_url)
    assert changed == 0
