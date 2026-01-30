"""
Tests for v1 → v2 migration script.
"""

import pytest
from datetime import datetime, timezone
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.inference_repository import InferenceRepository
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models import InferenceRun


@pytest.fixture
def v1_db():
    """Fixture for v1 database (SQLite in-memory)."""
    db = DatabaseConnection("sqlite:///:memory:")
    db.init_db()
    return db


@pytest.fixture
def v2_db():
    """Fixture for v2 database (SQLite in-memory for testing)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_migrate_single_inference_run(v1_db, v2_db):
    """Test migrating a single inference run."""
    # Insert into v1
    with v1_db.session_scope() as v1_session:
        v1_repo = InferenceRepository(v1_session)
        v1_id = v1_repo.insert_inference_run(
            prompt="What is 2+2?",
            response="4",
            provider="azure_openai_gpt-4",
            tokens=10,
            latency_ms=250.5,
            metadata={"temperature": 0.7},
        )
    
    # Migrate manually (simulating migration script logic)
    with v1_db.session_scope() as v1_session:
        inference_run = v1_session.query(InferenceRun).filter_by(id=v1_id).first()
        
        # Convert to v2 format
        request_json = {"prompt": inference_run.prompt}
        response_json = {"text": inference_run.response}
        tokens_json = {"total": inference_run.tokens} if inference_run.tokens else None
        
        # Insert into v2
        with v2_db.session_scope() as v2_session:
            v2_repo = RawCallRepository(v2_session)
            v2_id = v2_repo.insert_raw_call(
                provider=inference_run.provider,
                request_json=request_json,
                response_json=response_json,
                status="success",
                latency_ms=inference_run.latency_ms,
                tokens_json=tokens_json,
                metadata_json=inference_run.metadata_json or {},
            )
    
    # Verify migration
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        raw_call = v2_repo.get_raw_call_by_id(v2_id)
        
        assert raw_call is not None
        assert raw_call.provider == "azure_openai_gpt-4"
        assert raw_call.status == "success"
        assert raw_call.request_json == {"prompt": "What is 2+2?"}
        assert raw_call.response_json == {"text": "4"}
        assert raw_call.latency_ms == 250.5
        assert raw_call.tokens_json == {"total": 10}
        assert raw_call.metadata_json == {"temperature": 0.7}


def test_migrate_batch_grouping(v1_db, v2_db):
    """Test migrating batch_id to Group + GroupMember."""
    batch_id = "test-batch-123"
    
    # Insert into v1 with batch_id
    with v1_db.session_scope() as v1_session:
        v1_repo = InferenceRepository(v1_session)
        call_ids = []
        for i in range(3):
            call_id = v1_repo.insert_inference_run(
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                provider="test",
                batch_id=batch_id,
            )
            call_ids.append(call_id)
    
    # Migrate batches (simulating migration script logic)
    with v1_db.session_scope() as v1_session:
        v1_repo = InferenceRepository(v1_session)
        runs = v1_repo.get_inferences_by_batch_id(batch_id)
        
        # First migrate the calls (simplified - in real migration, this happens first)
        call_id_mapping = {}
        with v2_db.session_scope() as v2_session:
            v2_repo = RawCallRepository(v2_session)
            
            for run in runs:
                v2_id = v2_repo.insert_raw_call(
                    provider=run.provider,
                    request_json={"prompt": run.prompt},
                    response_json={"text": run.response},
                    status="success",
                )
                call_id_mapping[run.id] = v2_id
        
        # Then create group and add members
        with v2_db.session_scope() as v2_session:
            v2_repo = RawCallRepository(v2_session)
            
            group_id = v2_repo.create_group(
                group_type="batch",
                name=f"batch_{batch_id}",
                description=f"Migrated batch from v1 (original batch_id: {batch_id})",
                metadata_json={"original_batch_id": batch_id},
            )
            
            for idx, run in enumerate(runs):
                v2_call_id = call_id_mapping[run.id]
                v2_repo.add_call_to_group(group_id, v2_call_id, position=idx)
    
    # Verify migration
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        
        # Find the group
        from study_query_llm.db.models_v2 import Group
        groups = v2_session.query(Group).filter_by(group_type="batch").all()
        assert len(groups) == 1
        
        group = groups[0]
        assert group.name == f"batch_{batch_id}"
        assert group.metadata_json["original_batch_id"] == batch_id
        
        # Verify calls in group
        calls = v2_repo.get_calls_in_group(group.id)
        assert len(calls) == 3


def test_migrate_multiple_batches(v1_db, v2_db):
    """Test migrating multiple batches."""
    # Insert into v1 with different batch_ids
    with v1_db.session_scope() as v1_session:
        v1_repo = InferenceRepository(v1_session)
        
        # Batch 1
        for i in range(2):
            v1_repo.insert_inference_run(
                prompt=f"Batch1 Prompt {i}",
                response=f"Batch1 Response {i}",
                provider="test",
                batch_id="batch-1",
            )
        
        # Batch 2
        for i in range(3):
            v1_repo.insert_inference_run(
                prompt=f"Batch2 Prompt {i}",
                response=f"Batch2 Response {i}",
                provider="test",
                batch_id="batch-2",
            )
    
    # Migrate (simplified version)
    with v1_db.session_scope() as v1_session:
        v1_repo = InferenceRepository(v1_session)
        
        # Get all batch_ids
        batch_ids = ["batch-1", "batch-2"]
        
        for batch_id in batch_ids:
            runs = v1_repo.get_inferences_by_batch_id(batch_id)
            
            # Migrate calls and create group
            with v2_db.session_scope() as v2_session:
                v2_repo = RawCallRepository(v2_session)
                
                call_id_mapping = {}
                for run in runs:
                    v2_id = v2_repo.insert_raw_call(
                        provider=run.provider,
                        request_json={"prompt": run.prompt},
                        response_json={"text": run.response},
                        status="success",
                    )
                    call_id_mapping[run.id] = v2_id
                
                group_id = v2_repo.create_group("batch", f"batch_{batch_id}")
                for idx, run in enumerate(runs):
                    v2_repo.add_call_to_group(group_id, call_id_mapping[run.id], position=idx)
    
    # Verify
    with v2_db.session_scope() as v2_session:
        from study_query_llm.db.models_v2 import Group
        groups = v2_session.query(Group).filter_by(group_type="batch").all()
        assert len(groups) == 2
        
        v2_repo = RawCallRepository(v2_session)
        for group in groups:
            calls = v2_repo.get_calls_in_group(group.id)
            if "batch-1" in group.name:
                assert len(calls) == 2
            elif "batch-2" in group.name:
                assert len(calls) == 3


def test_migrate_inference_without_batch(v1_db, v2_db):
    """Test migrating inference runs without batch_id."""
    # Insert into v1 without batch_id
    with v1_db.session_scope() as v1_session:
        v1_repo = InferenceRepository(v1_session)
        v1_id = v1_repo.insert_inference_run(
            prompt="Standalone prompt",
            response="Standalone response",
            provider="test",
            batch_id=None,
        )
    
    # Migrate
    with v1_db.session_scope() as v1_session:
        run = v1_session.query(InferenceRun).filter_by(id=v1_id).first()
        
        with v2_db.session_scope() as v2_session:
            v2_repo = RawCallRepository(v2_session)
            v2_id = v2_repo.insert_raw_call(
                provider=run.provider,
                request_json={"prompt": run.prompt},
                response_json={"text": run.response},
                status="success",
            )
    
    # Verify - should not be in any group
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        groups = v2_repo.get_groups_for_call(v2_id)
        assert len(groups) == 0
