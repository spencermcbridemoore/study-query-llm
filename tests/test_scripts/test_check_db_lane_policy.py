"""Unit tests for scripts/check_db_lane_policy.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "check_db_lane_policy.py"


def _mod():
    spec = importlib.util.spec_from_file_location("check_db_lane_policy", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_file_flags_missing_write_intent_and_raw_create_engine(tmp_path: Path) -> None:
    mod = _mod()
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "\n".join(
            [
                "from study_query_llm.db.connection_v2 import DatabaseConnectionV2",
                "from sqlalchemy import create_engine",
                "def run():",
                "    DatabaseConnectionV2('postgresql://u:p@h/db', enable_pgvector=False)",
                "    create_engine('postgresql://u:p@h/db')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = mod._check_file(file_path)
    messages = [item[2] for item in violations]
    assert any("missing explicit write_intent" in message for message in messages)
    assert any("Direct create_engine call outside allowlist" in message for message in messages)


def test_check_file_accepts_explicit_write_intent_without_create_engine(tmp_path: Path) -> None:
    mod = _mod()
    file_path = tmp_path / "sample_ok.py"
    file_path.write_text(
        "\n".join(
            [
                "from study_query_llm.db.connection_v2 import DatabaseConnectionV2",
                "from study_query_llm.db.write_intent import WriteIntent",
                "def run():",
                "    DatabaseConnectionV2(",
                "        'sqlite:///:memory:',",
                "        enable_pgvector=False,",
                "        write_intent=WriteIntent.SANDBOX,",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = mod._check_file(file_path)
    assert violations == []
