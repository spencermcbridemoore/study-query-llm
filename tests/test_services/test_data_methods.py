"""Tests for register-first data method catalog and registrar."""

from __future__ import annotations

from study_query_llm.algorithms.data_methods import (
    DATA_METHODS,
    MATURITY_RUNNER_WIRED,
    register_data_methods,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import MethodDefinition
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_data_method_specs_have_required_fields() -> None:
    required_fields = {
        "name",
        "version",
        "role",
        "code_ref",
        "description",
        "parameters_schema",
        "maturity",
    }
    for spec in DATA_METHODS:
        missing = required_fields - set(spec.keys())
        assert not missing, (
            f"Spec for {spec.get('name')}@{spec.get('version')} missing {sorted(missing)}"
        )
        assert spec["maturity"] == MATURITY_RUNNER_WIRED
        assert isinstance(spec["parameters_schema"], dict)


def test_data_method_catalog_has_distinct_name_version_pairs() -> None:
    keys = [(spec["name"], spec["version"]) for spec in DATA_METHODS]
    assert len(keys) == len(set(keys))


def test_register_data_methods_is_idempotent() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        first = register_data_methods(method_svc)
        second = register_data_methods(method_svc)

        assert first == second
        assert session.query(MethodDefinition).count() == len(DATA_METHODS)


def test_register_data_methods_persists_parameters_schema() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        register_data_methods(method_svc)

        for spec in DATA_METHODS:
            row = method_svc.get_method(str(spec["name"]), version=str(spec["version"]))
            assert row is not None
            assert row.parameters_schema == spec["parameters_schema"]

