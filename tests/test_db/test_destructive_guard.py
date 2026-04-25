"""
Regression tests for destructive DDL guardrails in BaseDatabaseConnection.
"""

import pytest

from study_query_llm.db import _base_connection as base_connection_module
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.write_intent import WriteIntent

_PG_URL_A = "postgresql+psycopg2://guard:secret@guard-a.invalid:5432/guard_db"
_PG_URL_A_TIMEOUT = f"{_PG_URL_A}?connect_timeout=1"
_PG_URL_A_OTHER_DB_TIMEOUT = (
    "postgresql+psycopg2://guard:secret@guard-a.invalid:5432/other_guard_db"
    "?connect_timeout=1"
)
_PG_URL_B_TIMEOUT = (
    "postgresql+psycopg2://guard:secret@guard-b.invalid:5432/guard_db"
    "?connect_timeout=1"
)


@pytest.fixture(autouse=True)
def _clear_destructive_guard_env(monkeypatch):
    """Ensure tests are isolated from shell-level destructive env settings."""
    monkeypatch.delenv("SQLLM_ALLOW_DESTRUCTIVE_DDL", raising=False)
    monkeypatch.delenv("JETSTREAM_DATABASE_URL", raising=False)
    monkeypatch.delenv("CANONICAL_DATABASE_URL", raising=False)


def test_drop_all_tables_postgres_without_override_raises_runtimeerror():
    db = DatabaseConnectionV2(
        _PG_URL_A,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    with pytest.raises(RuntimeError, match="Set SQLLM_ALLOW_DESTRUCTIVE_DDL=1"):
        db.drop_all_tables()


def test_recreate_db_postgres_without_override_raises_runtimeerror():
    db = DatabaseConnectionV2(
        _PG_URL_A,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    with pytest.raises(RuntimeError, match="Set SQLLM_ALLOW_DESTRUCTIVE_DDL=1"):
        db.recreate_db()


def test_drop_all_tables_sqlite_memory_and_file_succeeds(tmp_path):
    memory_db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    memory_db.init_db()
    memory_db.drop_all_tables()

    sqlite_file = tmp_path / "guard.sqlite"
    file_db = DatabaseConnectionV2(
        f"sqlite:///{sqlite_file.as_posix()}",
        enable_pgvector=False,
    )
    file_db.init_db()
    file_db.drop_all_tables()


def test_override_allows_non_jetstream_target_then_fails_downstream(monkeypatch):
    monkeypatch.setenv("SQLLM_ALLOW_DESTRUCTIVE_DDL", "1")
    db = DatabaseConnectionV2(
        _PG_URL_A_TIMEOUT,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    with pytest.raises(Exception) as exc_info:
        db.drop_all_tables()

    assert not isinstance(exc_info.value, RuntimeError)


def test_jetstream_match_is_hard_stopped_even_with_override(monkeypatch):
    monkeypatch.setenv("SQLLM_ALLOW_DESTRUCTIVE_DDL", "1")
    monkeypatch.setenv("JETSTREAM_DATABASE_URL", _PG_URL_A_TIMEOUT)
    db = DatabaseConnectionV2(
        _PG_URL_A_TIMEOUT,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    with pytest.raises(RuntimeError, match="non-overridable"):
        db.drop_all_tables()


def test_same_host_port_different_dbname_is_not_jetstream_match(monkeypatch):
    monkeypatch.setenv("SQLLM_ALLOW_DESTRUCTIVE_DDL", "1")
    monkeypatch.setenv("JETSTREAM_DATABASE_URL", _PG_URL_A_TIMEOUT)
    db = DatabaseConnectionV2(
        _PG_URL_A_OTHER_DB_TIMEOUT,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    with pytest.raises(Exception) as exc_info:
        db.drop_all_tables()

    assert not isinstance(exc_info.value, RuntimeError)


def test_target_comparison_normalizes_localhost_aliases():
    url_localhost = "postgresql+psycopg2://guard:secret@localhost:5432/guard_db"
    url_loopback = "postgresql+psycopg2://guard:secret@127.0.0.1:5432/guard_db"
    url_other = _PG_URL_B_TIMEOUT

    assert base_connection_module._same_postgres_target(url_localhost, url_loopback)
    assert not base_connection_module._same_postgres_target(url_localhost, url_other)
