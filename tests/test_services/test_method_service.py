"""
Tests for MethodService.

Tests versioned method registration, default-to-active lookup, and result recording.
"""

import pytest
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_register_method(db_connection):
    """Test method registration."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        method_id = method_svc.register_method(
            name="extract_correct_answers",
            version="1",
            code_ref="scripts/parse_quiz.py",
            code_commit="abc123",
            description="Extract correct answer from quiz text",
        )

        method = method_svc.get_method("extract_correct_answers", version="1")
        assert method is not None
        assert method.id == method_id
        assert method.name == "extract_correct_answers"
        assert method.version == "1"
        assert method.is_active is True
        assert method.code_ref == "scripts/parse_quiz.py"
        assert method.code_commit == "abc123"


def test_get_method_defaults_to_active(db_connection):
    """Test that get_method returns active version when version is None."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        method_svc.register_method(
            name="parse_quiz",
            version="1",
            code_ref="scripts/parse_quiz.py",
        )
        method_svc.register_method(
            name="parse_quiz",
            version="2",
            code_ref="scripts/parse_quiz_v2.py",
        )

        active = method_svc.get_method("parse_quiz")
        assert active is not None
        assert active.version == "2"
        assert active.is_active is True

        v1 = method_svc.get_method("parse_quiz", version="1")
        assert v1 is not None
        assert v1.version == "1"
        assert v1.is_active is False


def test_version_upgrade_deactivates_previous(db_connection):
    """Test that registering a new version deactivates the previous."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        method_svc.register_method(
            name="extract_ari",
            version="1",
        )
        method_svc.register_method(
            name="extract_ari",
            version="2",
        )

        v1 = method_svc.get_method("extract_ari", version="1")
        v2 = method_svc.get_method("extract_ari", version="2")
        assert v1.is_active is False
        assert v2.is_active is True


def test_record_result_and_query(db_connection):
    """Test recording results and querying them."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        method_svc = MethodService(repo)

        run_id = provenance.create_run_group(algorithm="test_sweep")
        method_id = method_svc.register_method(
            name="extract_metrics",
            version="1",
            code_ref="experiments/result_metrics.py",
        )

        result_id = method_svc.record_result(
            method_definition_id=method_id,
            source_group_id=run_id,
            result_key="ari",
            result_value=0.85,
            result_json={"per_k": {"5": 0.85, "10": 0.78}},
        )

        assert result_id > 0

        results = method_svc.query_results(
            method_name="extract_metrics",
            source_group_id=run_id,
        )
        assert len(results) == 1
        assert results[0].result_key == "ari"
        assert results[0].result_value == 0.85
        assert results[0].result_json["per_k"]["5"] == 0.85


def test_query_results_filters(db_connection):
    """Test query_results with various filters."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        method_svc = MethodService(repo)

        run_id = provenance.create_run_group(algorithm="test_sweep")
        method_id = method_svc.register_method(
            name="extract_metrics",
            version="1",
        )

        method_svc.record_result(
            method_definition_id=method_id,
            source_group_id=run_id,
            result_key="ari",
            result_value=0.85,
        )
        method_svc.record_result(
            method_definition_id=method_id,
            source_group_id=run_id,
            result_key="silhouette",
            result_value=0.42,
        )

        by_key = method_svc.query_results(
            source_group_id=run_id,
            result_key="ari",
        )
        assert len(by_key) == 1
        assert by_key[0].result_key == "ari"

        by_method = method_svc.query_results(method_name="extract_metrics")
        assert len(by_method) == 2


def test_get_method_nonexistent(db_connection):
    """Test get_method returns None for nonexistent method."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        assert method_svc.get_method("nonexistent") is None
        assert method_svc.get_method("extract_ari", version="99") is None


def test_resolve_method_input_requirements_defaults_to_embedding_required(db_connection):
    """Absent input contract should keep embedding-backed defaults."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        method_svc.register_method(
            name="requirements_default",
            version="1",
        )

        requirements = method_svc.resolve_method_input_requirements("requirements_default")
        assert requirements.snapshot is True
        assert requirements.embedding_batch is True


def test_resolve_method_input_requirements_snapshot_only_contract(db_connection):
    """Explicit contract should allow snapshot-only execution."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        method_svc.register_method(
            name="requirements_snapshot_only",
            version="1",
            input_schema={
                "required_inputs": {
                    "snapshot": True,
                    "embedding_batch": False,
                }
            },
        )

        requirements = method_svc.resolve_method_input_requirements(
            "requirements_snapshot_only",
            version="1",
        )
        assert requirements.snapshot is True
        assert requirements.embedding_batch is False


def test_resolve_method_input_requirements_normalizes_malformed_read_values(db_connection):
    """Legacy malformed contract values should normalize to safe defaults on read."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        method_svc.register_method(
            name="requirements_malformed_read",
            version="1",
            input_schema={
                "required_inputs": {
                    "snapshot": True,
                    "embedding_batch": True,
                }
            },
        )
        method_row = method_svc.get_method("requirements_malformed_read", version="1")
        assert method_row is not None
        method_row.input_schema = {
            "required_inputs": {
                "snapshot": "yes",
                "embedding_batch": "no",
            }
        }
        session.flush()

        requirements = method_svc.resolve_method_input_requirements(
            "requirements_malformed_read",
            version="1",
        )
        assert requirements.snapshot is True
        assert requirements.embedding_batch is True


def test_register_method_rejects_invalid_required_inputs_shape(db_connection):
    """New registrations should reject clearly invalid required_inputs shapes."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        with pytest.raises(ValueError, match="required_inputs must be a JSON object"):
            method_svc.register_method(
                name="requirements_invalid_shape",
                version="1",
                input_schema={"required_inputs": "snapshot_only"},
            )


def test_register_method_rejects_non_boolean_required_inputs_flags(db_connection):
    """New registrations should reject non-boolean required_inputs flags."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        with pytest.raises(ValueError, match="snapshot must be a boolean"):
            method_svc.register_method(
                name="requirements_invalid_snapshot_flag",
                version="1",
                input_schema={
                    "required_inputs": {
                        "snapshot": "true",
                        "embedding_batch": True,
                    }
                },
            )
