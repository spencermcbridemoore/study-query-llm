"""Unit tests for scripts/db_target_guardrails.py."""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from db_target_guardrails import is_loopback_target, parse_postgres_target, same_db_target


def test_parse_postgres_target_normalizes_localhost() -> None:
    target = parse_postgres_target("postgresql://u:p@localhost:5433/mydb")
    assert target.host == "127.0.0.1"
    assert target.port == 5433
    assert target.dbname == "mydb"
    assert target.username == "u"


def test_same_db_target_treats_localhost_and_loopback_as_equal() -> None:
    assert same_db_target(
        "postgresql://u:p@localhost:5433/mydb",
        "postgresql://u:p@127.0.0.1:5433/mydb",
    )


def test_is_loopback_target_false_for_remote_host() -> None:
    assert not is_loopback_target("postgresql://u:p@example.com:5432/mydb")
