"""Tests for lane resolution and write-intent primitives."""

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.lane import (
    CANONICAL_DATABASE_URL_ENV,
    JETSTREAM_DATABASE_URL_ENV,
    LOCAL_DATABASE_URL_ENV,
    CanonicalIdentityConflict,
    Lane,
    resolve_lane,
)
from study_query_llm.db.write_intent import (
    LaneIntentMismatch,
    WriteIntent,
    assert_intent_matches_lane,
    parse_write_intent,
)

_CANONICAL_URL = "postgresql+psycopg2://u:p@canonical.example:5432/study_query_jetstream"
_LOCAL_URL = "postgresql+psycopg2://study:study@127.0.0.1:5433/study_query_local"
_REMOTE_OTHER_URL = "postgresql+psycopg2://u:p@remote.example:5432/other_db"


@pytest.fixture(autouse=True)
def _clear_lane_env(monkeypatch):
    monkeypatch.delenv(CANONICAL_DATABASE_URL_ENV, raising=False)
    monkeypatch.delenv(JETSTREAM_DATABASE_URL_ENV, raising=False)
    monkeypatch.delenv(LOCAL_DATABASE_URL_ENV, raising=False)
    monkeypatch.delenv("SQLLM_WRITE_INTENT", raising=False)
    monkeypatch.delenv("ARTIFACT_STORAGE_BACKEND", raising=False)


def test_resolve_lane_handles_sqlite_memory_and_file():
    assert resolve_lane("sqlite:///:memory:") == Lane.SQLITE_MEMORY
    assert resolve_lane("sqlite:///tmp/example.sqlite") == Lane.SQLITE_FILE


def test_resolve_lane_uses_canonical_database_url(monkeypatch):
    monkeypatch.setenv(CANONICAL_DATABASE_URL_ENV, _CANONICAL_URL)
    assert resolve_lane(_CANONICAL_URL) == Lane.CANONICAL


def test_resolve_lane_falls_back_to_jetstream_alias(monkeypatch):
    monkeypatch.setenv(JETSTREAM_DATABASE_URL_ENV, _CANONICAL_URL)
    assert resolve_lane(_CANONICAL_URL) == Lane.CANONICAL


def test_resolve_lane_raises_on_canonical_identity_conflict(monkeypatch):
    monkeypatch.setenv(CANONICAL_DATABASE_URL_ENV, _CANONICAL_URL)
    monkeypatch.setenv(JETSTREAM_DATABASE_URL_ENV, _REMOTE_OTHER_URL)
    with pytest.raises(CanonicalIdentityConflict):
        resolve_lane(_CANONICAL_URL)


def test_resolve_lane_uses_local_database_url_match(monkeypatch):
    monkeypatch.setenv(LOCAL_DATABASE_URL_ENV, _LOCAL_URL)
    assert resolve_lane(_LOCAL_URL) == Lane.LOCAL_POSTGRES


def test_resolve_lane_treats_loopback_postgres_as_local():
    url = "postgresql+psycopg2://study:study@localhost:5432/sandbox_db"
    assert resolve_lane(url) == Lane.LOCAL_POSTGRES


def test_resolve_lane_returns_unknown_for_nonmatching_remote():
    assert resolve_lane(_REMOTE_OTHER_URL) == Lane.UNKNOWN


def test_parse_write_intent_accepts_case_insensitive_values():
    assert parse_write_intent("CANONICAL") == WriteIntent.CANONICAL
    assert parse_write_intent("read_mirror") == WriteIntent.READ_MIRROR
    assert parse_write_intent("Sandbox") == WriteIntent.SANDBOX


def test_parse_write_intent_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid write intent"):
        parse_write_intent("invalid-intent")


def test_assert_intent_matches_lane_allows_supported_pairs():
    assert_intent_matches_lane(WriteIntent.CANONICAL, Lane.CANONICAL)
    assert_intent_matches_lane(WriteIntent.READ_MIRROR, Lane.LOCAL_POSTGRES)
    assert_intent_matches_lane(WriteIntent.SANDBOX, Lane.SQLITE_MEMORY)


def test_assert_intent_matches_lane_rejects_mismatch():
    with pytest.raises(LaneIntentMismatch, match="incompatible"):
        assert_intent_matches_lane(WriteIntent.CANONICAL, Lane.LOCAL_POSTGRES)


def test_connection_phase1a_defaults_intent_from_lane():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False, quiet=True)
    assert db.lane == Lane.SQLITE_MEMORY
    assert db.write_intent == WriteIntent.SANDBOX


def test_connection_phase1a_uses_env_write_intent(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLLM_WRITE_INTENT", "read_mirror")
    sqlite_file = tmp_path / "phase1a.sqlite"
    db = DatabaseConnectionV2(
        f"sqlite:///{sqlite_file.as_posix()}",
        enable_pgvector=False,
        quiet=True,
    )
    assert db.write_intent == WriteIntent.READ_MIRROR


def test_connection_rejects_invalid_explicit_write_intent():
    with pytest.raises(ValueError, match="Invalid write intent"):
        DatabaseConnectionV2(
            "sqlite:///:memory:",
            enable_pgvector=False,
            write_intent="not-real",
            quiet=True,
        )


def test_preflight_banner_prints_lane_and_intent(monkeypatch, capsys):
    monkeypatch.setenv(CANONICAL_DATABASE_URL_ENV, _CANONICAL_URL)
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "azure_blob")
    monkeypatch.setenv("SQLLM_WRITE_INTENT", "canonical")
    _ = DatabaseConnectionV2(_CANONICAL_URL, enable_pgvector=False)
    captured = capsys.readouterr()
    assert "STUDY-QUERY-LLM - DB SESSION" in captured.err
    assert "lane:" in captured.err
    assert "intent:" in captured.err
    assert "artifact_back:" in captured.err


def test_opt_in_env_enforces_lane_intent_mismatch(monkeypatch):
    monkeypatch.setenv("SQLLM_WRITE_INTENT", "canonical")
    with pytest.raises(LaneIntentMismatch, match="incompatible"):
        DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False, quiet=True)


def test_non_sqlite_requires_explicit_or_env_intent():
    with pytest.raises(ValueError, match="require an explicit write intent"):
        DatabaseConnectionV2(_REMOTE_OTHER_URL, enable_pgvector=False, quiet=True)
