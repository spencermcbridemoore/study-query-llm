"""2.1: the analysis_request identity sites must agree (PG + SQLite).

The partial UNIQUE index (models_v2), the migration's duplicate probe, and the
remediation finder all encode the same identity contract. This pins that contract
behind one shared module and asserts the four agree: the index ENFORCES exactly
the identities the probe FINDS and the extractor CLASSIFIES, with container/half
(NULL-distinct) rows ignored by all three.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from study_query_llm.db import models_v2
from study_query_llm.db.analysis_request_identity import (
    ANALYSIS_REQUEST_UNIQUE_INDEX_NAME,
    build_duplicate_probe_sql,
    build_unique_index_sql,
    extract_identity,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository

# The exact committed DDL. The canonical Postgres index was built from these
# strings; a change here is a real schema change and must be deliberate.
EXPECTED_PG = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_analysis_request_identity "
    "ON groups ((metadata_json ->> 'method_name'), (metadata_json ->> 'input_id'), "
    "(metadata_json ->> 'run_key')) WHERE group_type = 'analysis_request'"
)
EXPECTED_SQLITE = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_analysis_request_identity "
    "ON groups (json_extract(metadata_json, '$.method_name'), "
    "json_extract(metadata_json, '$.input_id'), "
    "json_extract(metadata_json, '$.run_key')) WHERE group_type = 'analysis_request'"
)

# name -> metadata_json. Two genuine duplicates (m/1/k), two containers (all-null),
# one half identity, one distinct identity. Only (m,1,k) is a duplicate.
_FIXTURES = [
    ("dup1", {"method_name": "m", "input_id": 1, "run_key": "k"}),
    ("dup2", {"method_name": "m", "input_id": 1, "run_key": "k"}),
    ("container1", {}),
    ("container2", {}),
    ("half", {"method_name": "m"}),
    ("other", {"method_name": "m", "input_id": 2, "run_key": "k"}),
]
_EXPECTED_DUP_IDENTITIES = {("m", "1", "k")}


def test_index_ddl_is_byte_stable() -> None:
    assert build_unique_index_sql("postgresql") == EXPECTED_PG
    assert build_unique_index_sql("sqlite") == EXPECTED_SQLITE
    # models_v2 re-exports must equal the builder output (no drift between the
    # model constants and the shared contract).
    assert models_v2.ANALYSIS_REQUEST_UNIQUE_INDEX_SQL_POSTGRESQL == EXPECTED_PG
    assert models_v2.ANALYSIS_REQUEST_UNIQUE_INDEX_SQL_SQLITE == EXPECTED_SQLITE


def test_extract_identity_classifies() -> None:
    assert extract_identity({"method_name": "m", "input_id": 1, "run_key": "k"}) == ("m", "1", "k")
    # falsy-but-present values are part of the identity (0 is not "missing").
    assert extract_identity({"method_name": "m", "input_id": 0, "run_key": "k"}) == ("m", "0", "k")
    # any missing/None field -> not an identity (container/half rows).
    assert extract_identity({"method_name": "m", "input_id": None, "run_key": "k"}) is None
    assert extract_identity({"method_name": "m"}) is None
    assert extract_identity({}) is None
    assert extract_identity(None) is None


def _sqlite_db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False, quiet=True)
    db.init_db()  # builds the unique index via after_create
    return db


def test_sqlite_index_enforces_only_full_identity() -> None:
    db = _sqlite_db()
    with db.session_scope() as session:
        RawCallRepository(session).create_group(
            group_type="analysis_request",
            name="dup1",
            metadata_json={"method_name": "m", "input_id": 1, "run_key": "k"},
        )
    # second identical all-present identity collides on the unique index
    with pytest.raises(IntegrityError):
        with db.session_scope() as session:
            RawCallRepository(session).create_group(
                group_type="analysis_request",
                name="dup2",
                metadata_json={"method_name": "m", "input_id": 1, "run_key": "k"},
            )
    # container (all-null) and half rows are NULL-distinct: never collide
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        repo.create_group(group_type="analysis_request", name="c1", metadata_json={})
        repo.create_group(group_type="analysis_request", name="c2", metadata_json={})
        repo.create_group(
            group_type="analysis_request", name="half", metadata_json={"method_name": "m"}
        )


def _seed_then_probe_identities(db: DatabaseConnectionV2, dialect: str) -> set[tuple[str, ...]]:
    """Drop the index, seed the fixtures, return the probe's duplicate identity set."""
    with db.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX IF EXISTS {ANALYSIS_REQUEST_UNIQUE_INDEX_NAME}"))
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        for name, meta in _FIXTURES:
            repo.create_group(group_type="analysis_request", name=name, metadata_json=meta)
    with db.session_scope() as session:
        rows = session.execute(text(build_duplicate_probe_sql(dialect))).mappings().all()
    return {(r["method_name"], str(r["input_id"]), r["run_key"]) for r in rows}


def test_probe_and_extractor_agree_sqlite() -> None:
    db = _sqlite_db()
    probe_identities = _seed_then_probe_identities(db, "sqlite")
    assert probe_identities == _EXPECTED_DUP_IDENTITIES

    # the Python extractor must classify the same rows as duplicates
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    with db.session_scope() as session:
        for group in session.query(Group).filter(Group.group_type == "analysis_request").all():
            identity = extract_identity(group.metadata_json)
            if identity is not None:
                buckets[identity].append(int(group.id))
    extractor_dups = {key for key, ids in buckets.items() if len(ids) > 1}
    assert extractor_dups == _EXPECTED_DUP_IDENTITIES
    assert extractor_dups == probe_identities


def test_postgres_arm_renders_and_optionally_agrees() -> None:
    # Always: the Postgres rendering is well-formed and uses the ->> operator.
    pg_probe = build_duplicate_probe_sql("postgresql")
    assert "metadata_json ->> 'method_name'" in pg_probe
    assert pg_probe.count("IS NOT NULL") == 3

    url = (os.environ.get("TEST_POSTGRES_URL") or "").strip()
    if not url:
        pytest.skip(
            "set TEST_POSTGRES_URL (throwaway Postgres; may need "
            "SQLLM_ALLOW_DESTRUCTIVE_DDL=1) to run the Postgres agreement arm"
        )
    db = DatabaseConnectionV2(url, enable_pgvector=False, quiet=True)
    db.init_db()
    probe_identities = _seed_then_probe_identities(db, "postgresql")
    assert probe_identities == _EXPECTED_DUP_IDENTITIES
