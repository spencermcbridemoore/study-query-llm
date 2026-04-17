"""Tests for the register-only text-classification method catalog.

Covers:
* Idempotent registration via
  :func:`register_text_classification_methods`.
* Required schema fields per entry in
  :data:`TEXT_CLASSIFICATION_METHODS`.
* No two entries collide on ``(name, version)``.
"""

from __future__ import annotations

from study_query_llm.algorithms.text_classification_methods import (
    MATURITY_REGISTERED_ONLY,
    TEXT_CLASSIFICATION_METHODS,
    register_text_classification_methods,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import MethodDefinition
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


def test_text_classification_method_specs_have_required_fields():
    """Each entry has the fields the registrar (and downstream readers)
    will need."""
    required_top_level = {
        "name",
        "version",
        "role",
        "description",
        "parameters_schema",
        "maturity",
    }
    for spec in TEXT_CLASSIFICATION_METHODS:
        missing = required_top_level - set(spec.keys())
        assert not missing, (
            f"Spec for {spec.get('name')}@{spec.get('version')} missing "
            f"required fields: {sorted(missing)}"
        )
        assert isinstance(spec["name"], str) and spec["name"]
        assert isinstance(spec["version"], str) and spec["version"]
        assert isinstance(spec["role"], str) and spec["role"]
        assert isinstance(spec["description"], str) and spec["description"]
        assert isinstance(spec["parameters_schema"], dict)
        assert spec["parameters_schema"].get("type") == "object"
        assert isinstance(spec["parameters_schema"].get("properties"), dict)
        assert spec["maturity"] == MATURITY_REGISTERED_ONLY


def test_text_classification_methods_use_distinct_names():
    """No two catalog entries share the same (name, version) pair."""
    keys = [(spec["name"], spec["version"]) for spec in TEXT_CLASSIFICATION_METHODS]
    assert len(keys) == len(set(keys)), (
        f"Duplicate (name, version) pairs in TEXT_CLASSIFICATION_METHODS: {keys}"
    )


def test_text_classification_methods_register_only_subset_present():
    """The five register-only methods explicitly in scope for this prep are
    present (training-heavy variants are deferred and must NOT appear)."""
    keys = {(spec["name"], spec["version"]) for spec in TEXT_CLASSIFICATION_METHODS}
    expected = {
        ("knn_prototype_classifier", "0.1"),
        ("linear_probe_logreg", "0.1"),
        ("label_embedding_zero_shot", "0.1"),
        ("prompted_llm_classifier", "0.1"),
        ("mixture_of_experts_classifier", "0.1"),
    }
    assert keys == expected


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_register_text_classification_methods_idempotent():
    """Calling the registrar twice does not proliferate rows and returns
    the same id mapping each time."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        first = register_text_classification_methods(method_svc)
        second = register_text_classification_methods(method_svc)

        assert first == second
        count = session.query(MethodDefinition).count()
        assert count == len(TEXT_CLASSIFICATION_METHODS)
        assert set(first.keys()) == {
            f"{spec['name']}@{spec['version']}"
            for spec in TEXT_CLASSIFICATION_METHODS
        }


def test_register_text_classification_methods_persists_schema():
    """Each registered row carries the catalog's parameters_schema."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        register_text_classification_methods(method_svc)
        for spec in TEXT_CLASSIFICATION_METHODS:
            row = method_svc.get_method(spec["name"], version=spec["version"])
            assert row is not None
            assert row.parameters_schema == spec["parameters_schema"]
