"""Unit tests for BANK77 bootstrap script helpers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.provenance_service import ProvenanceService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "create_bank77_snapshot_and_embeddings.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("bank77_bootstrap_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load BANK77 bootstrap script module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_label_contract_enforces_cardinality() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="label cardinality changed"):
        module._validate_label_contract({0: "intent_a", 1: "intent_b"})


def test_build_manifest_entries_preserves_split_index_ids() -> None:
    module = _load_script_module()
    rows = [
        {
            "split": "train",
            "source_id": "train:0",
            "text": "Transfer money",
            "label": 10,
            "intent": "beneficiary_not_allowed",
        },
        {
            "split": "test",
            "source_id": "test:2",
            "text": "Card payment failed",
            "label": 4,
            "intent": "cash_withdrawal_not_recognised",
        },
    ]
    entries = module._build_manifest_entries(rows)
    assert [entry["position"] for entry in entries] == [0, 1]
    assert [entry["source_id"] for entry in entries] == ["train:0", "test:2"]
    assert [entry["split"] for entry in entries] == ["train", "test"]
    assert [entry["label"] for entry in entries] == [10, 4]


def test_compute_intent_means_is_deterministic_for_bank77_contract() -> None:
    module = _load_script_module()
    label_id_to_intent = {idx: f"intent_{idx}" for idx in range(77)}
    rows = [
        {
            "split": "train",
            "source_id": f"train:{idx}",
            "text": f"row-{idx}",
            "label": idx,
            "intent": label_id_to_intent[idx],
        }
        for idx in range(77)
    ]
    full_matrix = np.arange(77 * 3, dtype=np.float64).reshape(77, 3)

    means_matrix, mapping_rows, counts_by_label = module._compute_intent_means(
        full_matrix=full_matrix,
        rows=rows,
        label_id_to_intent=label_id_to_intent,
    )

    np.testing.assert_allclose(means_matrix, full_matrix)
    assert means_matrix.shape == (77, 3)
    assert len(mapping_rows) == 77
    assert mapping_rows[0]["label_id"] == 0
    assert mapping_rows[-1]["label_id"] == 76
    assert mapping_rows[0]["row_index"] == 0
    assert mapping_rows[-1]["row_index"] == 76
    assert counts_by_label[0] == 1
    assert counts_by_label[76] == 1


def test_verify_bootstrap_and_read_means_payload_roundtrip() -> None:
    module = _load_script_module()
    snapshot_name = "bank77_test_snapshot"
    source_dataset = "mteb/banking77"
    full_dataset_key = "bank77:test:full"
    means_dataset_key = "bank77:test:means"
    embedding_engine = "test-embedding-engine"
    provider = "azure"
    prev_backend = os.environ.get("ARTIFACT_STORAGE_BACKEND")
    os.environ["ARTIFACT_STORAGE_BACKEND"] = "local"

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = DatabaseConnectionV2(f"sqlite:///{db_path}", enable_pgvector=False)
        db.init_db()
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            provenance = ProvenanceService(repo)
            artifacts = ArtifactService(repository=repo)

            entries = [
                {
                    "position": idx,
                    "source_id": f"train:{idx}",
                    "split": "train",
                    "text": f"text-{idx}",
                    "label": idx,
                    "intent": f"intent_{idx}",
                }
                for idx in range(77)
            ]
            snapshot_id = provenance.create_dataset_snapshot_group(
                snapshot_name=snapshot_name,
                source_dataset=source_dataset,
                sample_size=len(entries),
                label_mode="labeled",
                sampling_method="full_train_test_concat_deterministic",
                metadata={
                    "manifest_hash": "fake-hash",
                    "label_count": 77,
                },
            )
            artifacts.store_dataset_snapshot_manifest(
                snapshot_group_id=int(snapshot_id),
                snapshot_name=snapshot_name,
                entries=entries,
                metadata={"label_count": 77},
            )

            full_batch_id = provenance.create_embedding_batch_group(
                deployment=embedding_engine,
                metadata={
                    "dataset_key": full_dataset_key,
                    "provider": provider,
                    "entry_max": 77,
                    "key_version": module.CACHE_KEY_VERSION,
                },
            )
            full_matrix = np.arange(77 * 4, dtype=np.float64).reshape(77, 4)
            artifacts.store_embedding_matrix(
                int(full_batch_id),
                full_matrix,
                dataset_key=full_dataset_key,
                embedding_engine=embedding_engine,
                provider=provider,
                entry_max=77,
                key_version=module.CACHE_KEY_VERSION,
            )
            repo.create_group_link(
                parent_group_id=int(full_batch_id),
                child_group_id=int(snapshot_id),
                link_type="depends_on",
                metadata_json={"relation": "embedding_source_snapshot"},
            )

            means_batch_id = provenance.create_embedding_batch_group(
                deployment=embedding_engine,
                metadata={
                    "dataset_key": means_dataset_key,
                    "provider": provider,
                    "entry_max": 77,
                    "key_version": module.CACHE_KEY_VERSION,
                    "representation": "intent_means",
                    "source_snapshot_group_id": int(snapshot_id),
                    "source_embedding_batch_group_id": int(full_batch_id),
                },
            )
            means_matrix = full_matrix.copy()
            artifacts.store_embedding_matrix(
                int(means_batch_id),
                means_matrix,
                dataset_key=means_dataset_key,
                embedding_engine=embedding_engine,
                provider=provider,
                entry_max=77,
                key_version=module.CACHE_KEY_VERSION,
                metadata={"representation": "intent_means"},
            )
            mapping_rows = [
                {
                    "row_index": idx,
                    "label_id": idx,
                    "intent": f"intent_{idx}",
                    "count": 1,
                }
                for idx in range(77)
            ]
            artifacts.store_metrics(
                run_id=int(means_batch_id),
                metrics={
                    "schema_version": "bank77.intent_mapping.v1",
                    "mapping_type": module.MAPPING_TYPE,
                    "ordered_label_ids": list(range(77)),
                    "ordered_intents": [f"intent_{idx}" for idx in range(77)],
                    "rows": mapping_rows,
                },
                step_name=module.MAPPING_STEP_NAME,
                metadata={"mapping_type": module.MAPPING_TYPE},
            )
            repo.create_group_link(
                parent_group_id=int(means_batch_id),
                child_group_id=int(snapshot_id),
                link_type="depends_on",
                metadata_json={"relation": "embedding_source_snapshot"},
            )
            repo.create_group_link(
                parent_group_id=int(means_batch_id),
                child_group_id=int(full_batch_id),
                link_type="depends_on",
                metadata_json={"relation": "derived_from_full_embedding_matrix"},
            )

        means_loaded, mapping_payload, means_group_id, _, _ = module.read_means_payload(
            db,
            means_dataset_key=means_dataset_key,
            provider=provider,
            embedding_engine=embedding_engine,
            expected_label_count=77,
        )
        assert means_loaded.shape == (77, 4)
        assert means_group_id > 0
        assert len(mapping_payload["ordered_label_ids"]) == 77

        summary = module.verify_bootstrap(
            db,
            snapshot_name=snapshot_name,
            source_dataset=source_dataset,
            full_dataset_key=full_dataset_key,
            means_dataset_key=means_dataset_key,
            provider=provider,
            embedding_engine=embedding_engine,
            expected_row_count=77,
            expected_label_count=77,
        )
        assert summary["full_embedding_shape"] == [77, 4]
        assert summary["means_shape"] == [77, 4]
        assert int(summary["expected_label_count"]) == 77
    finally:
        if prev_backend is None:
            os.environ.pop("ARTIFACT_STORAGE_BACKEND", None)
        else:
            os.environ["ARTIFACT_STORAGE_BACKEND"] = prev_backend
        try:
            os.unlink(db_path)
        except OSError:
            pass
