"""Unit tests for scripts/check_raw_calls_uri_sentinel.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "check_raw_calls_uri_sentinel.py"


def _mod():
    spec = importlib.util.spec_from_file_location("check_raw_calls_uri_sentinel", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_index_name_rejects_invalid_value() -> None:
    mod = _mod()
    assert mod._validate_index_name("idx_raw_calls_uri_non_blob_sentinel") == (
        "idx_raw_calls_uri_non_blob_sentinel"
    )
    with pytest.raises(ValueError):
        mod._validate_index_name("bad-index-name")


def test_resolve_database_url_prefers_explicit_then_env(monkeypatch) -> None:
    mod = _mod()
    monkeypatch.setenv("CANONICAL_DATABASE_URL", "postgresql://canonical")
    monkeypatch.setenv("DATABASE_URL", "postgresql://database")
    explicit = mod._resolve_database_url(explicit_url="postgresql://explicit", env_var="CANONICAL_DATABASE_URL")
    assert explicit == "postgresql://explicit"
    from_env = mod._resolve_database_url(explicit_url=None, env_var="CANONICAL_DATABASE_URL")
    assert from_env == "postgresql://canonical"
