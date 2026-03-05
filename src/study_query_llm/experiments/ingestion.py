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
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.experiments.result_metrics import (
    METRICS as _METRICS,
    extract_by_k_metrics as _extract_by_k_metrics,
)
from study_query_llm.experiments.sweep_io import serialize_sweep_result


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
    engine = metadata.get("embedding_engine", "?")
    summarizer = str(metadata.get("summarizer", "None"))
    n_restarts = metadata.get("n_restarts", 50)
    n_samples = metadata.get("actual_entry_count", 0)
    data_type = "50runs"

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

            # Optional explicit run -> snapshot linkage.
            for snapshot_id in snapshot_ids:
                try:
                    provenance.link_run_to_dataset_snapshot(run_id, snapshot_id)
                except Exception:
                    # Keep run ingestion robust even when snapshot linkage is missing/invalid.
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
