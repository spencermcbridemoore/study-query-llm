"""Tests for unified provenanced run service and compatibility mapping."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.mcq_run_persistence import persist_mcq_probe_result
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_CLUSTERING_RUN,
    GROUP_TYPE_MCQ_RUN,
)
from study_query_llm.services.provenanced_run_service import (
    RUN_KIND_ANALYSIS_EXECUTION,
    RUN_KIND_EXECUTION,
    RUN_KIND_METHOD_EXECUTION,
    ProvenancedRunService,
)
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_record_method_and_analysis_execution_runs() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_service = SweepRequestService(repo)
        request_id = request_service.create_request(
            request_name="provenance_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 3, "n_restarts": 2},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=50,
            sweep_type="clustering",
            execution_mode="standalone",
        )
        req = request_service.get_request(request_id)
        run_key = str(req["expected_run_keys"][0])
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="r1",
            metadata_json={"run_key": run_key},
        )
        method_service = MethodService(repo)
        method_id = method_service.register_method(
            name="cosine_kllmeans_no_pca",
            version="1.0",
            description="test method",
        )
        service = ProvenancedRunService(repo)
        method_run_id = service.record_method_execution(
            request_group_id=request_id,
            run_key=run_key,
            source_group_id=run_id,
            result_group_id=run_id,
            method_definition_id=method_id,
            config_json={"k_min": 2, "k_max": 3},
            determinism_class="pseudo_deterministic",
            run_status="completed",
        )
        analysis_run_id = service.record_analysis_execution(
            request_group_id=request_id,
            source_group_id=run_id,
            method_definition_id=method_id,
            analysis_key="stability_report",
            config_json={"window": 10},
            run_status="completed",
        )
        assert method_run_id > 0
        assert analysis_run_id > 0

        # Upsert semantics for method execution key.
        method_run_id_2 = service.record_method_execution(
            request_group_id=request_id,
            run_key=run_key,
            source_group_id=run_id,
            result_group_id=run_id,
            method_definition_id=method_id,
            config_json={"k_min": 2, "k_max": 3},
            determinism_class="pseudo_deterministic",
            run_status="completed",
        )
        assert method_run_id_2 == method_run_id

        rows = repo.list_provenanced_runs(request_group_id=request_id)
        kinds = {row.run_kind for row in rows}
        assert kinds == {RUN_KIND_EXECUTION}
        roles = {str((row.metadata_json or {}).get("execution_role") or "") for row in rows}
        assert roles == {RUN_KIND_METHOD_EXECUTION, RUN_KIND_ANALYSIS_EXECUTION}


def test_unified_execution_view_compatibility_maps_legacy_rows() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_service = SweepRequestService(repo)
        request_id = request_service.create_request(
            request_name="compat_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=20,
            sweep_type="clustering",
        )
        req = request_service.get_request(request_id)
        run_key = str(req["expected_run_keys"][0])
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="legacy_run",
            metadata_json={"run_key": run_key},
        )
        method_service = MethodService(repo)
        method_id = method_service.register_method(
            name="legacy_analysis",
            version="1.0",
            description="legacy compatibility",
        )
        method_service.record_result(
            method_definition_id=method_id,
            source_group_id=run_id,
            result_key="chi_square",
            result_value=1.23,
        )

        service = ProvenancedRunService(repo)
        rows = service.list_unified_execution_view(request_id)
        assert any(
            row.get("run_kind") == RUN_KIND_EXECUTION
            and row.get("execution_role") == RUN_KIND_METHOD_EXECUTION
            and bool((row.get("metadata_json") or {}).get("compatibility_mapped"))
            for row in rows
        )
        assert any(
            row.get("run_kind") == RUN_KIND_EXECUTION
            and row.get("execution_role") == RUN_KIND_ANALYSIS_EXECUTION
            and bool((row.get("metadata_json") or {}).get("compatibility_mapped"))
            for row in rows
        )


def test_unified_execution_view_compatibility_uses_snapshot_metadata() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_service = SweepRequestService(repo)
        request_id = request_service.create_request(
            request_name="compat_snapshot_meta",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=20,
            sweep_type="clustering",
        )
        req = request_service.get_request(request_id)
        run_key = str(req["expected_run_keys"][0])
        snapshot_id = repo.create_group(
            group_type="dataset_snapshot",
            name="dbpedia_20_seed42_labeled",
            metadata_json={
                "snapshot_name": "dbpedia_20_seed42_labeled",
                "source_dataset": "dbpedia",
                "sample_size": 20,
                "label_mode": "labeled",
                "sampling_method": "seeded",
            },
        )
        repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="legacy_run_with_meta_snapshot",
            metadata_json={
                "run_key": run_key,
                "dataset_snapshot_ids": [int(snapshot_id)],
            },
        )

        service = ProvenancedRunService(repo)
        rows = service.list_unified_execution_view(request_id)
        method_rows = [
            row
            for row in rows
            if row.get("run_kind") == RUN_KIND_EXECUTION
            and row.get("execution_role") == RUN_KIND_METHOD_EXECUTION
            and str(row.get("run_key") or "") == run_key
            and bool((row.get("metadata_json") or {}).get("compatibility_mapped"))
        ]
        assert method_rows
        assert int(method_rows[0].get("input_snapshot_group_id") or 0) == int(snapshot_id)


def test_unified_execution_view_compatibility_falls_back_to_snapshot_links() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_service = SweepRequestService(repo)
        request_id = request_service.create_request(
            request_name="compat_snapshot_link",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=20,
            sweep_type="clustering",
        )
        req = request_service.get_request(request_id)
        run_key = str(req["expected_run_keys"][0])
        snapshot_id = repo.create_group(
            group_type="dataset_snapshot",
            name="dbpedia_20_seed99_labeled",
            metadata_json={
                "snapshot_name": "dbpedia_20_seed99_labeled",
                "source_dataset": "dbpedia",
                "sample_size": 20,
                "label_mode": "labeled",
                "sampling_method": "seeded",
            },
        )
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="legacy_run_with_link_snapshot",
            metadata_json={"run_key": run_key},
        )
        repo.create_group_link(
            parent_group_id=int(run_id),
            child_group_id=int(snapshot_id),
            link_type="depends_on",
        )

        service = ProvenancedRunService(repo)
        rows = service.list_unified_execution_view(request_id)
        method_rows = [
            row
            for row in rows
            if row.get("run_kind") == RUN_KIND_EXECUTION
            and row.get("execution_role") == RUN_KIND_METHOD_EXECUTION
            and str(row.get("run_key") or "") == run_key
            and bool((row.get("metadata_json") or {}).get("compatibility_mapped"))
        ]
        assert method_rows
        assert int(method_rows[0].get("input_snapshot_group_id") or 0) == int(snapshot_id)


def test_mcq_persistence_writes_explicit_method_execution_provenance() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_service = SweepRequestService(repo)
        request_id = request_service.create_request(
            request_name="mcq_new_runs",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 2, "template_version": "v1"},
            parameter_axes={
                "levels": ["high school"],
                "subjects": ["physics"],
                "deployments": ["gpt-4o-mini"],
                "options_per_question": [4],
                "questions_per_test": [5],
                "label_styles": ["upper"],
                "spread_correct_answer_uniformly": [False],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        request = request_service.get_request(request_id)
        assert request is not None
        run_key = str(request["expected_run_keys"][0])
        target = dict((request.get("run_key_to_target") or {}).get(run_key) or {})
    run_id = persist_mcq_probe_result(
        db=db,
        request_id=int(request_id),
        run_key=run_key,
        target=target,
        probe_details={"summary": {"valid_runs": 2}},
    )
    assert run_id > 0
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_rows = repo.list_provenanced_runs(
            request_group_id=int(request_id),
            run_kind=RUN_KIND_METHOD_EXECUTION,
        )
        assert len(method_rows) == 1
        row = method_rows[0]
        assert row.run_kind == RUN_KIND_EXECUTION
        assert row.run_key == run_key
        assert row.source_group_id == run_id
        assert row.result_group_id == run_id
        assert row.determinism_class == "non_deterministic"
        assert row.method_definition_id is not None


def test_unified_execution_view_mcq_legacy_rows_still_compatibility_mapped() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_service = SweepRequestService(repo)
        request_id = request_service.create_request(
            request_name="mcq_legacy_rows",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 1, "template_version": "v1"},
            parameter_axes={
                "levels": ["high school"],
                "subjects": ["physics"],
                "deployments": ["gpt-4o-mini"],
                "options_per_question": [4],
                "questions_per_test": [5],
                "label_styles": ["upper"],
                "spread_correct_answer_uniformly": [False],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        request = request_service.get_request(request_id)
        assert request is not None
        run_key = str(request["expected_run_keys"][0])
        repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="legacy_mcq_run",
            metadata_json={"run_key": run_key},
        )
        service = ProvenancedRunService(repo)
        rows = service.list_unified_execution_view(request_id)
        compat_method_rows = [
            row
            for row in rows
            if row.get("run_kind") == RUN_KIND_EXECUTION
            and row.get("execution_role") == RUN_KIND_METHOD_EXECUTION
            and bool((row.get("metadata_json") or {}).get("compatibility_mapped"))
        ]
        assert compat_method_rows
