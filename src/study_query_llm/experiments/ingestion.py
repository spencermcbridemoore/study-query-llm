"""In-memory sweep result ingestion to database."""

import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.services.provenanced_run_service import ProvenancedRunService
from study_query_llm.experiments.result_metrics import (
    METRICS as _METRICS,
    extract_by_k_metrics as _extract_by_k_metrics,
)
from study_query_llm.experiments.sweep_io import serialize_sweep_result


def run_key_exists_in_db(db: DatabaseConnectionV2, run_key: str) -> bool:
    """Check if a run_key already exists in the database (idempotency check)."""
    with db.session_scope() as session:
        existing = (
            session.query(Group)
            .filter(
                Group.group_type == "clustering_run",
                sa_text("metadata_json->>'run_key' = :rk"),
            )
            .params(rk=run_key)
            .first()
        )
        return existing is not None


def ingest_result_to_db(
    result: Any,
    metadata: Dict[str, Any],
    ground_truth_labels: Optional[np.ndarray],
    db: DatabaseConnectionV2,
    run_key: str,
) -> Optional[int]:
    """Save an in-memory sweep result directly to NeonDB as Group/GroupLink entries.

    This mirrors the logic in ``scripts/ingest_sweep_to_db.py`` but operates on
    the live result object rather than a pkl file, so it can be called immediately
    after each run completes (no intermediate file required).

    Args:
        result: The ``SweepResult`` object returned by ``run_sweep()``.
        metadata: The metadata dict you would normally pass to
            ``save_single_sweep_result()`` (must include ``benchmark_source``,
            ``embedding_engine``, ``summarizer``, ``n_restarts``).
        ground_truth_labels: Ground-truth label array (``None`` for datasets
            without ground truth such as estela).
        db: Active ``DatabaseConnectionV2`` instance.
        run_key: Unique idempotency key, e.g.
            ``"dbpedia_embed_v_4_0_gpt_5_chat_300_50runs"``.

    Returns:
        The run-group ID written to the DB, or ``None`` if the key already
        existed (skipped) or the write failed.
    """
    result_dict = serialize_sweep_result(result)
    dataset = metadata.get("benchmark_source", "unknown")
    engine = metadata.get("embedding_engine") or metadata.get("embedding_deployment", "?")
    summarizer = str(metadata.get("summarizer", "None"))
    n_restarts = metadata.get("n_restarts", 50)
    n_samples = metadata.get("actual_entry_count", 0)
    data_type = metadata.get("data_type", "50runs")

    # Optional snapshot provenance linkage for reproducibility.
    raw_snapshot_ids = metadata.get("dataset_snapshot_ids")
    if raw_snapshot_ids is None and metadata.get("dataset_snapshot_id") is not None:
        raw_snapshot_ids = [metadata.get("dataset_snapshot_id")]
    snapshot_ids: list[int] = []
    if raw_snapshot_ids is not None:
        if not isinstance(raw_snapshot_ids, (list, tuple)):
            raw_snapshot_ids = [raw_snapshot_ids]
        for sid in raw_snapshot_ids:
            try:
                snapshot_ids.append(int(sid))
            except (TypeError, ValueError):
                continue
    # Keep snapshot linkage deterministic/idempotent across write paths.
    snapshot_ids = sorted(set(snapshot_ids))
    primary_snapshot_id: Optional[int] = snapshot_ids[0] if snapshot_ids else None

    try:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            provenance = ProvenanceService(repo)

            existing = session.query(Group).filter(
                Group.group_type == "clustering_run",
                sa_text("metadata_json->>'run_key' = :rk"),
            ).params(rk=run_key).first()
            if existing:
                print(f"      [SKIP] Already in DB: run_key={run_key} (group {existing.id})")
                return None

            run_metadata = {
                "algorithm": "cosine_kllmeans_no_pca",
                "run_key": run_key,
                "dataset": dataset,
                "embedding_engine": engine,
                "summarizer": summarizer,
                "n_restarts": n_restarts,
                "n_samples": n_samples,
                "data_type": data_type,
                "k_range": [
                    metadata.get("sweep_config", {}).get("k_min", 2),
                    metadata.get("sweep_config", {}).get("k_max", 20),
                ],
                "source": "digested_during_calculation",
                "ingested_at": datetime.utcnow().isoformat(),
                **{k: v for k, v in metadata.items() if k not in ("sweep_config",)},
            }
            if snapshot_ids:
                run_metadata["dataset_snapshot_ids"] = snapshot_ids

            try:
                run_id = provenance.create_run_group(
                    algorithm="cosine_kllmeans_no_pca",
                    config=run_metadata,
                    name=f"sweep_{dataset}_{engine}_{data_type}",
                    description=f"{dataset}/{engine}/{summarizer} ({n_restarts} restarts)",
                )
            except IntegrityError:
                # Concurrent writer created this run_key after our existence check.
                session.rollback()
                existing = session.query(Group).filter(
                    Group.group_type == "clustering_run",
                    sa_text("metadata_json->>'run_key' = :rk"),
                ).params(rk=run_key).first()
                if existing:
                    print(
                        f"      [SKIP] Concurrently created run_key={run_key} "
                        f"(group {existing.id})"
                    )
                    return existing.id
                raise

            run_group = repo.get_group_by_id(run_id)
            run_group.metadata_json = run_metadata
            session.flush()

            # Persist canonical sweep artifact via ArtifactService (blob-first pipeline)
            artifact_service = ArtifactService(repository=repo)
            artifact_id = artifact_service.store_sweep_results(
                run_id=run_id,
                sweep_results=result_dict,
                step_name="sweep_complete",
                metadata={
                    "run_key": run_key,
                    "dataset": dataset,
                    "embedding_engine": engine,
                    "summarizer": summarizer,
                    "n_restarts": n_restarts,
                    "n_samples": n_samples,
                    "data_type": data_type,
                },
            )
            if artifact_id > 0:
                from study_query_llm.db.models_v2 import CallArtifact

                artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
                if artifact:
                    run_metadata["artifact_id"] = artifact_id
                    run_metadata["artifact_uri"] = str(artifact.uri)
                    run_group.metadata_json = dict(run_metadata)
            session.flush()

            # Link this execution into the unified provenanced_run contract.
            method_service = MethodService(repo)
            provenanced_run_service = ProvenancedRunService(repo)
            method_name = str(run_metadata.get("algorithm") or "clustering_method")
            method_version = str(metadata.get("method_version") or "1.0")
            method = method_service.get_method(method_name, version=method_version)
            if method is None:
                method_id = method_service.register_method(
                    name=method_name,
                    version=method_version,
                    description="Auto-registered clustering execution method definition",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "k_min": {"type": "integer"},
                            "k_max": {"type": "integer"},
                            "n_restarts": {"type": "integer"},
                        },
                    },
                )
            else:
                method_id = int(method.id)

            request_group_id_raw = metadata.get("request_group_id")
            if request_group_id_raw is None:
                # Compatibility fallback: infer request_group_id by expected run_key.
                request_groups = session.query(Group).filter(
                    Group.group_type.in_(["clustering_sweep_request", "mcq_sweep_request"])
                ).all()
                for req_group in request_groups:
                    req_meta = dict(req_group.metadata_json or {})
                    expected = set(str(x) for x in (req_meta.get("expected_run_keys") or []))
                    if run_key in expected:
                        request_group_id_raw = int(req_group.id)
                        break
            request_group_id = (
                int(request_group_id_raw) if request_group_id_raw is not None else None
            )
            if request_group_id is not None:
                provenanced_run_service.record_method_execution(
                    request_group_id=request_group_id,
                    run_key=run_key,
                    source_group_id=int(run_id),
                    result_group_id=int(run_id),
                    input_snapshot_group_id=primary_snapshot_id,
                    method_definition_id=int(method_id),
                    determinism_class=str(
                        metadata.get("determinism_class") or "pseudo_deterministic"
                    ),
                    config_json={
                        "sweep_config": dict(metadata.get("sweep_config") or {}),
                        "embedding_engine": engine,
                        "summarizer": summarizer,
                        "n_restarts": n_restarts,
                    },
                    result_ref=str(run_metadata.get("artifact_uri") or ""),
                    metadata_json={
                        "artifact_id": run_metadata.get("artifact_id"),
                        "source": str(run_metadata.get("source") or ""),
                    },
                    run_status="completed",
                )

            # Optional explicit run -> snapshot linkage.
            for snapshot_id in snapshot_ids:
                try:
                    provenance.link_run_to_dataset_snapshot(run_id, snapshot_id)
                except Exception:
                    # Keep run ingestion robust even when snapshot linkage is missing/invalid.
                    pass

            # Optional explicit run -> embedding_batch linkage and batch -> snapshot chain.
            embedding_batch_group_id = metadata.get("embedding_batch_group_id")
            if embedding_batch_group_id is not None:
                try:
                    emb_gid = int(embedding_batch_group_id)
                    provenance.link_run_to_embedding_batch(run_id, emb_gid)
                    for snapshot_id in snapshot_ids:
                        try:
                            provenance.link_embedding_batch_to_dataset_snapshot(emb_gid, snapshot_id)
                        except Exception:
                            pass
                except Exception:
                    pass

            by_k = _extract_by_k_metrics(result_dict, ground_truth_labels)
            for k in sorted(by_k.keys()):
                metrics_for_k = by_k[k]
                step_metadata: Dict[str, Any] = {
                    "k": k,
                    "n_samples": n_samples,
                }
                for m in _METRICS:
                    step_metadata[f"{m}s"] = [
                        v for v in metrics_for_k.get(m, []) if v is not None
                    ]

                step_id = provenance.create_step_group(
                    parent_run_id=run_id,
                    step_name=f"k={k}",
                    step_type="clustering",
                    metadata=step_metadata,
                )
                repo.create_group_link(
                    parent_group_id=run_id,
                    child_group_id=step_id,
                    link_type="clustering_step",
                    position=k,
                )

            print(f"      [DB] Saved run_key={run_key} -> group {run_id} ({len(by_k)} k-steps)")
            return run_id

    except Exception as exc:
        print(f"      [ERROR] DB ingestion failed for run_key={run_key}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
