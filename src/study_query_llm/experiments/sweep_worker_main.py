#!/usr/bin/env python3
"""Request-mode sweep worker: run-key claims and orchestration job execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets import load_dataset

from study_query_llm.algorithms import SweepConfig, run_sweep
from study_query_llm.analysis.mcq_analyze_request import run_mcq_analyses_for_request
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import SweepRunClaim
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.ingestion import ingest_result_to_db, run_key_exists_in_db
from study_query_llm.experiments.mcq_run_persistence import mcq_run_key_exists_in_db
from study_query_llm.experiments.sweep_mcq_standalone import execute_mcq_standalone_run
from study_query_llm.experiments.sweep_request_types import SWEEP_TYPE_CLUSTERING, SWEEP_TYPE_MCQ
from study_query_llm.experiments.sweep_io import get_output_dir, serialize_sweep_result
from study_query_llm.providers.managed_tei_embedding_provider import ManagedTEIEmbeddingProvider
from study_query_llm.providers.managers.local_docker_tei import LocalDockerTEIManager
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.embeddings import (
    CACHE_KEY_VERSION,
    EmbeddingRequest,
    EmbeddingService,
)
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.services.paraphraser_factory import create_paraphraser_for_llm
from study_query_llm.services.jobs import (
    JobReducerService,
    JobRunContext,
    JobRunOutcome,
    create_job_runner,
)
from study_query_llm.services.sweep_request_service import SweepRequestService
from study_query_llm.utils.estela_loader import load_estela_dict
from study_query_llm.utils.text_utils import flatten_prompt_dict

DATABASE_URL = os.environ.get("DATABASE_URL")

# Populated by main() before run_worker / helper functions use it.
db: Optional[DatabaseConnectionV2] = None  # type: ignore[assignment]

ENTRY_MAX = int(os.environ.get("ENTRY_MAX", "300"))
N_RESTARTS = 50
K_MIN, K_MAX = 2, 20
OUT_PREFIX = "local_300_2d_14e_4s"

EMBEDDING_ENGINES = [
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "WhereIsAI/UAE-Large-V1",
    "BAAI/bge-m3",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",
    "sentence-transformers/all-mpnet-base-v2",
]

SUMMARIZERS = [None, "gpt-4o-mini", "gpt-4o", "gpt-5-chat"]

SWEEP_CONFIG = SweepConfig(
    skip_pca=True,
    k_min=K_MIN,
    k_max=K_MAX,
    max_iter=200,
    base_seed=0,
    n_restarts=N_RESTARTS,
    compute_stability=True,
    llm_interval=20,
    max_samples=10,
    distance_metric="cosine",
    normalize_vectors=True,
)

OUTPUT_DIR = get_output_dir()


def _safe_name(s: str) -> str:
    return s.replace("-", "_").replace("/", "_")


def _run_async_sync(coro_factory):
    def _runner():
        return asyncio.run(coro_factory())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_runner).result()


async def _fetch_embeddings(
    texts,
    engine,
    emb_provider,
    dataset_key: str,
    provider_name: str,
    db_conn: Optional[DatabaseConnectionV2] = None,
):
    conn = db_conn or db
    with conn.session_scope() as session:
        repo = RawCallRepository(session)
        artifact_service = ArtifactService(repository=repo)
        l3_hit = artifact_service.find_embedding_matrix_artifact(
            dataset_key=dataset_key,
            embedding_engine=engine,
            provider=provider_name,
            entry_max=len(texts),
            key_version=CACHE_KEY_VERSION,
        )
        if l3_hit:
            matrix = artifact_service.load_artifact(l3_hit["uri"], "embedding_matrix")
            return np.asarray(matrix, dtype=np.float64)

        service = EmbeddingService(repository=repo, provider=emb_provider)
        requests = [
            EmbeddingRequest(text=t, deployment=engine, provider=emb_provider.get_provider_name())
            for t in texts
        ]
        responses = await service.get_embeddings_batch(requests, chunk_size=32)
        matrix = np.asarray([r.vector for r in responses], dtype=np.float64)
        provenance = ProvenanceService(repo)
        embedding_batch_group_id = provenance.create_embedding_batch_group(
            deployment=engine,
            metadata={
                "dataset_key": dataset_key,
                "provider": provider_name,
                "entry_max": len(texts),
                "key_version": CACHE_KEY_VERSION,
            },
        )
        artifact_service.store_embedding_matrix(
            embedding_batch_group_id,
            matrix,
            dataset_key=dataset_key,
            embedding_engine=engine,
            provider=provider_name,
            entry_max=len(texts),
            key_version=CACHE_KEY_VERSION,
        )
        return matrix


class _SharedEndpointManager:
    """Minimal manager-like adapter for shared TEI endpoints.

    ManagedTEIEmbeddingProvider only needs endpoint_url/model_id/provider_label
    and ping(). In shared mode lifecycle is owned externally by a supervisor.
    """

    def __init__(self, endpoint_url: str, model_id: str, provider_label: str):
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.provider_label = provider_label

    def ping(self) -> None:
        return None


def _now_utc():
    return datetime.now(timezone.utc)


def _claim_run_target(request_id: int, run_key: str, worker_id: str, lease_seconds: int) -> bool:
    now = _now_utc()
    lease_expires = now + timedelta(seconds=lease_seconds)
    with db.session_scope() as session:
        claim = (
            session.query(SweepRunClaim)
            .filter(
                SweepRunClaim.request_group_id == request_id,
                SweepRunClaim.run_key == run_key,
            )
            .first()
        )
        if claim is None:
            claim = SweepRunClaim(
                request_group_id=request_id,
                run_key=run_key,
                claim_status="claimed",
                claimed_by=worker_id,
                claimed_at=now,
                lease_expires_at=lease_expires,
                heartbeat_at=now,
            )
            session.add(claim)
            session.flush()
            return True
        if claim.claim_status == "completed":
            return False
        if (
            claim.claim_status == "claimed"
            and claim.lease_expires_at is not None
            and claim.lease_expires_at > now
            and claim.claimed_by != worker_id
        ):
            return False
        claim.claim_status = "claimed"
        claim.claimed_by = worker_id
        claim.claimed_at = now
        claim.lease_expires_at = lease_expires
        claim.heartbeat_at = now
        session.flush()
        return True


def _complete_run_claim(request_id: int, run_key: str, run_id: int, worker_id: str) -> None:
    with db.session_scope() as session:
        claim = (
            session.query(SweepRunClaim)
            .filter(
                SweepRunClaim.request_group_id == request_id,
                SweepRunClaim.run_key == run_key,
            )
            .first()
        )
        if not claim:
            return
        claim.claim_status = "completed"
        claim.claimed_by = worker_id
        claim.run_group_id = run_id
        claim.heartbeat_at = _now_utc()
        claim.lease_expires_at = None
        session.flush()


def _fail_run_claim(request_id: int, run_key: str, worker_id: str, error_message: str) -> None:
    with db.session_scope() as session:
        claim = (
            session.query(SweepRunClaim)
            .filter(
                SweepRunClaim.request_group_id == request_id,
                SweepRunClaim.run_key == run_key,
            )
            .first()
        )
        if not claim:
            return
        metadata = dict(claim.metadata_json or {})
        metadata["last_error"] = error_message[:500]
        metadata["failed_at"] = _now_utc().isoformat()
        claim.metadata_json = metadata
        claim.claim_status = "failed"
        claim.claimed_by = worker_id
        claim.heartbeat_at = _now_utc()
        claim.lease_expires_at = None
        session.flush()


def load_dbpedia_full():
    dataset = load_dataset("dbpedia_14", split="train")
    texts, labels = [], []
    for item in dataset:
        text, label = item.get("content", ""), item.get("label", -1)
        if text and 10 < len(text) <= 1000 and label >= 0:
            texts.append(text)
            labels.append(label)
    cats = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "SportsTeam",
        "Information",
        "Animal",
        "Plant",
        "Album",
    ]
    return texts, np.asarray(labels), cats


def load_estela_full(repo_root: Path):
    pkl_path = str(repo_root / "notebooks" / "estela_prompt_data.pkl")
    data = load_estela_dict(pkl_path=pkl_path)
    flat = flatten_prompt_dict(data)
    texts = [t for t in flat.values() if isinstance(t, str)]
    texts = [t.replace("\x00", "").strip() for t in texts]
    texts = [t for t in texts if 10 < len(t) <= 1000]
    labels = np.zeros(len(texts), dtype=np.int64)
    return texts, labels, []


def _load_datasets(repo_root: Path):
    loaded = {}
    cfgs = [
        {"name": "dbpedia", "label_max": 14, "has_gt": True},
        {"name": "estela", "label_max": None, "has_gt": False},
    ]
    for cfg in cfgs:
        name = cfg["name"]
        if name == "dbpedia":
            texts_all, labels_all, category_names = load_dbpedia_full()
        else:
            texts_all, labels_all, category_names = load_estela_full(repo_root)
        label_max = cfg.get("label_max")
        has_gt = cfg.get("has_gt", True)
        if label_max is not None and has_gt:
            unique_labels = sorted(set(labels_all))
            lm = min(label_max, len(unique_labels))
            mask = np.isin(labels_all, unique_labels[:lm])
            idx = np.where(mask)[0]
        else:
            idx = np.arange(len(texts_all))
            lm = 0
        if len(idx) > ENTRY_MAX:
            np.random.seed(42)
            idx = np.random.choice(idx, size=ENTRY_MAX, replace=False)
        texts = [texts_all[i] for i in idx]
        labels = labels_all[idx]
        valid = [i for i, t in enumerate(texts) if 10 < len(t) <= 1000]
        if len(valid) < len(texts):
            texts = [texts[i] for i in valid]
            labels = labels[valid]
        loaded[name] = {
            "texts": texts,
            "labels": labels,
            "label_max": lm if label_max is not None else 0,
            "category_names": category_names,
            "has_gt": has_gt,
        }
    return loaded


def _missing_run_targets(request_id: int) -> Tuple[List[Tuple[str, Dict[str, Any]]], dict]:
    """Missing (run_key, target_dict) pairs for any sweep_type."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        progress = svc.compute_progress(request_id)
        missing_run_keys = progress.get("missing_run_keys") or []
        req = svc.get_request(request_id)
        run_key_to_target = (req or {}).get("run_key_to_target") or {}
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for rk in missing_run_keys:
        t = run_key_to_target.get(rk)
        if t:
            pairs.append((rk, t))
    return pairs, progress


def _save_shard_artifact(payload: Dict, job_key: str) -> str:
    shards_dir = OUTPUT_DIR / "job_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_name(job_key)
    out_path = shards_dir / f"{safe_key}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return str(out_path)


def run_one_run_k_try_job(
    job_snapshot: Dict[str, Any],
    datasets: Dict[str, Any],
    provider_cache: Dict[str, object],
    manager_cache: Dict[str, object],
    tei_endpoint: Optional[str],
    provider_label: str,
    embedding_provider_name: Optional[str],
    worker_slot: int,
    repo_root: Path,
    db: DatabaseConnectionV2,
    claim_wait_seconds: float = 0.0,
) -> Tuple[int, Optional[str], Optional[str]]:
    """Execute one run_k_try job. Does not call claim/complete/fail/promote.
    Returns (job_id, result_ref, None) on success, (job_id, None, error_string) on failure.
    """
    payload = job_snapshot.get("payload_json") or {}
    job_id = int(job_snapshot["id"])
    job_key = str(job_snapshot.get("job_key"))
    base_run_key = job_snapshot.get("base_run_key")
    seed_value = job_snapshot.get("seed_value")

    current_engine = payload.get("embedding_engine")
    dataset_name = payload.get("dataset")
    summarizer_key = payload.get("summarizer", "None")
    k_min = int(payload.get("k_min", K_MIN))
    k_max = int(payload.get("k_max", K_MAX))
    seed_value = int(seed_value or payload.get("seed_value", 0))

    if current_engine not in provider_cache:
        manager, provider = _build_provider_for_engine(
            embedding_engine=current_engine,
            shared_endpoint=tei_endpoint,
            provider_label=provider_label,
            worker_slot=worker_slot,
            embedding_provider_name=embedding_provider_name,
        )
        manager_cache[current_engine] = manager
        provider_cache[current_engine] = provider
    emb_provider = provider_cache[current_engine]

    dataset = datasets.get(dataset_name)
    if dataset is None:
        return (job_id, None, f"dataset_not_loaded: {dataset_name}")

    texts = dataset["texts"]
    labels = dataset["labels"]
    gt = labels if dataset["has_gt"] else None
    summarizer_name = None if summarizer_key == "None" else summarizer_key

    def _embed_sync(txts):
        dataset_key = f"{dataset_name}:entry_max={len(txts)}"
        return _run_async_sync(
            lambda: _fetch_embeddings(
                txts,
                current_engine,
                emb_provider,
                dataset_key=dataset_key,
                provider_name=provider_label,
                db_conn=db,
            )
        )

    cfg = SweepConfig(
        skip_pca=True,
        k_min=k_min,
        k_max=k_max,
        max_iter=200,
        base_seed=seed_value,
        n_restarts=1,
        compute_stability=True,
        llm_interval=20,
        max_samples=10,
        distance_metric="cosine",
        normalize_vectors=True,
    )
    base_embed_started = time.perf_counter()
    embeddings = _embed_sync(texts)
    base_embed_seconds = time.perf_counter() - base_embed_started
    paraphraser = create_paraphraser_for_llm(summarizer_name, db) if summarizer_name else None
    sweep_started = time.perf_counter()
    result = run_sweep(
        texts,
        embeddings,
        cfg,
        paraphraser=paraphraser,
        embedder=_embed_sync if paraphraser else None,
    )
    sweep_seconds = time.perf_counter() - sweep_started
    sweep_timing = dict((result.pca or {}).get("timing") or {})
    shard_payload = {
        "pca": result.pca,
        "by_k": serialize_sweep_result(result).get("by_k", {}),
        "job_key": job_key,
        "base_run_key": base_run_key,
        "k_min": k_min,
        "k_max": k_max,
        "try_idx": int(payload.get("try_idx", 0)),
        "seed_value": seed_value,
        "dataset": dataset_name,
        "embedding_engine": current_engine,
        "summarizer": summarizer_key,
        "ground_truth_labels": gt.tolist() if gt is not None else None,
        "n_texts": len(texts),
        "profiling": {
            "claim_wait_seconds": float(claim_wait_seconds),
            "base_embed_seconds": float(base_embed_seconds),
            "run_sweep_seconds": float(sweep_seconds),
            "llm_paraphrase_seconds": float(sweep_timing.get("llm_paraphrase_seconds", 0.0) or 0.0),
            "llm_reembed_seconds": float(sweep_timing.get("llm_reembed_seconds", 0.0) or 0.0),
            "llm_paraphrase_calls": int(sweep_timing.get("llm_paraphrase_calls", 0) or 0),
            "llm_reembed_calls": int(sweep_timing.get("llm_reembed_calls", 0) or 0),
        },
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    result_ref = _save_shard_artifact(shard_payload, job_key)
    return (job_id, result_ref, None)


def run_one_mcq_run_job(
    *,
    job_snapshot: Dict[str, Any],
    db: DatabaseConnectionV2,
    worker_label: str,
) -> Tuple[int, Optional[str], Optional[str]]:
    """Execute one mcq_run orchestration job."""
    payload = dict(job_snapshot.get("payload_json") or {})
    job_id = int(job_snapshot["id"])
    request_id = int(
        job_snapshot.get("request_group_id")
        or payload.get("request_id")
        or 0
    )
    run_key = str(
        payload.get("run_key")
        or job_snapshot.get("base_run_key")
        or job_snapshot.get("job_key")
        or ""
    )
    if request_id <= 0:
        return (job_id, None, "missing_request_group_id")
    if not run_key:
        return (job_id, None, "missing_run_key")

    run_id, err = execute_mcq_standalone_run(
        db=db,
        request_id=request_id,
        run_key=run_key,
        target=payload,
        worker_label=worker_label,
    )
    if err is not None:
        return (job_id, None, err)
    if run_id is None:
        return (job_id, None, "persist_mcq_failed")
    return (job_id, str(run_id), None)


def run_one_analysis_run_job(
    *,
    job_snapshot: Dict[str, Any],
    db: DatabaseConnectionV2,
    worker_label: str,
) -> Tuple[int, Optional[str], Optional[str]]:
    """Execute one analysis_run orchestration job."""
    del worker_label  # worker label is currently not needed by the analysis driver.
    payload = dict(job_snapshot.get("payload_json") or {})
    job_id = int(job_snapshot["id"])
    request_id = int(
        job_snapshot.get("request_group_id")
        or payload.get("request_id")
        or 0
    )
    analysis_key = str(payload.get("analysis_key") or "")
    sweep_type = str(payload.get("sweep_type") or "").strip().lower()
    if request_id <= 0:
        return (job_id, None, "missing_request_group_id")
    if not analysis_key:
        return (job_id, None, "missing_analysis_key")
    if sweep_type and sweep_type != SWEEP_TYPE_MCQ:
        return (job_id, None, f"unsupported_analysis_sweep_type:{sweep_type}")
    try:
        report = run_mcq_analyses_for_request(
            db,
            request_id,
            dry_run=False,
            analysis_keys=[analysis_key],
            orchestration_job_id=job_id,
            skip_completed=True,
        )
    except Exception as exc:
        return (job_id, None, str(exc)[:1000])
    return (job_id, f"analysis:{analysis_key}:{len(report.get('recorded') or [])}", None)


def worker_main_queued(
    request_id: int,
    worker_slot: int,
    job_queue: Any,
    result_queue: Any,
    tei_endpoint: Optional[str],
    provider_label: str,
    embedding_provider_name: Optional[str],
    repo_root: Path,
    idle_timeout_seconds: Optional[int],
    database_url: str,
) -> None:
    """Queue-based worker: receives job snapshots from job_queue, runs run_k_try, puts
    (job_id, result_ref, error) on result_queue. Creates own DB connection (Windows-safe).
    Does not call claim/complete/fail/promote."""
    db = DatabaseConnectionV2(database_url, enable_pgvector=True)
    db.init_db()

    datasets = _load_datasets(repo_root)
    provider_cache: Dict[str, object] = {}
    manager_cache: Dict[str, object] = {}

    while True:
        job = job_queue.get()
        if job is None:
            break
        job_id = int(job["id"])
        try:
            job_id_out, result_ref_out, error_out = run_one_run_k_try_job(
                job_snapshot=job,
                datasets=datasets,
                provider_cache=provider_cache,
                manager_cache=manager_cache,
                tei_endpoint=tei_endpoint,
                provider_label=provider_label,
                embedding_provider_name=embedding_provider_name,
                worker_slot=worker_slot,
                repo_root=repo_root,
                db=db,
                claim_wait_seconds=0.0,
            )
            result_queue.put((job_id_out, result_ref_out, error_out))
        except Exception as exc:
            result_queue.put((job_id, None, str(exc)[:1000]))

    for provider in provider_cache.values():
        try:
            _run_async_sync(lambda p=provider: p.close())
        except Exception:
            pass
    for manager in manager_cache.values():
        if manager is None:
            continue
        try:
            manager.stop()
        except Exception:
            pass


def _claim_next_sharded_job(
    *,
    request_id: int,
    worker_id: str,
    lease_seconds: int,
    job_types: List[str],
    filter_payload: Optional[Dict[str, Any]],
):
    t0 = time.perf_counter()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        job = repo.claim_next_orchestration_job(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            request_group_id=request_id,
            job_types=job_types,
            filter_payload=filter_payload,
        )
        if not job:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "_claim_next_sharded_job: no_job request=%s duration_ms=%.1f",
                request_id, elapsed_ms,
            )
            return None
        snapshot = {
            "id": int(job.id),
            "request_group_id": int(job.request_group_id),
            "job_type": str(job.job_type),
            "payload_json": dict(job.payload_json or {}),
            "seed_value": job.seed_value,
            "job_key": str(job.job_key),
            "base_run_key": job.base_run_key,
        }
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        "_claim_next_sharded_job: job_id=%s request=%s duration_ms=%.1f",
        snapshot["id"], request_id, elapsed_ms,
    )
    return snapshot


def _run_sharded_worker_loop(
    *,
    request_id: int,
    worker_id: str,
    worker_slot: int,
    embedding_engine: Optional[str],
    tei_endpoint: Optional[str],
    provider_label: str,
    embedding_provider_name: Optional[str],
    claim_lease_seconds: int,
    max_runs: Optional[int],
    idle_exit_seconds: int,
    force: bool,
    repo_root: Path,
) -> int:
    loaded = _load_datasets(repo_root)
    reducer = JobReducerService(db)
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req = SweepRequestService(repo).get_request(request_id)
    sweep_type = (req or {}).get("sweep_type") or SWEEP_TYPE_CLUSTERING
    job_types = (
        ["mcq_run", "analysis_run"]
        if sweep_type == SWEEP_TYPE_MCQ
        else ["run_k_try", "reduce_k", "finalize_run"]
    )
    filter_payload = (
        {"embedding_engine": embedding_engine}
        if embedding_engine and sweep_type == SWEEP_TYPE_CLUSTERING
        else None
    )
    started_at = time.time()
    last_work_at = started_at
    completed_leaf_jobs = 0

    provider_cache: Dict[str, object] = {}
    manager_cache: Dict[str, object] = {}

    while True:
        if max_runs is not None and completed_leaf_jobs >= max_runs:
            break

        claim_started = time.perf_counter()
        job = _claim_next_sharded_job(
            request_id=request_id,
            worker_id=worker_id,
            lease_seconds=claim_lease_seconds,
            job_types=job_types,
            filter_payload=filter_payload,
        )
        claim_wait_seconds = time.perf_counter() - claim_started
        if not job:
            idle_for = int(time.time() - last_work_at)
            if idle_for >= idle_exit_seconds:
                print(f"[{worker_id}] Idle exit after {idle_for}s in sharded mode.")
                break
            time.sleep(5)
            continue

        job_type = job.get("job_type")
        job_id = int(job["id"])
        job_key = str(job.get("job_key"))
        try:
            runner = create_job_runner(
                job_type,
                run_k_try_fn=run_one_run_k_try_job,
                mcq_run_fn=run_one_mcq_run_job,
                analysis_run_fn=run_one_analysis_run_job,
                reducer=reducer,
            )
            context = JobRunContext(
                datasets=loaded,
                provider_cache=provider_cache,
                manager_cache=manager_cache,
                tei_endpoint=tei_endpoint,
                provider_label=provider_label,
                embedding_provider_name=embedding_provider_name,
                worker_slot=worker_slot,
                repo_root=repo_root,
                claim_wait_seconds=claim_wait_seconds,
                reducer=reducer,
                db=db,
            )
            outcome = runner.run(job, context)
            if outcome.error is not None:
                if not outcome.db_updated_by_runner:
                    with db.session_scope() as session:
                        repo = RawCallRepository(session)
                        repo.fail_orchestration_job(
                            outcome.job_id, error_json={"error": outcome.error}
                        )
                print(f"[{worker_id}] JOB ERROR {job_key}: {outcome.error}")
            else:
                if not outcome.db_updated_by_runner:
                    with db.session_scope() as session:
                        repo = RawCallRepository(session)
                        repo.complete_orchestration_job(
                            outcome.job_id, result_ref=outcome.result_ref
                        )
                    completed_leaf_jobs += 1
                    print(f"[{worker_id}] DONE job {job_key} -> {outcome.result_ref}")
                elif job_type == "reduce_k":
                    print(f"[{worker_id}] REDUCED job {job_key} -> {outcome.result_ref}")
                elif job_type == "finalize_run":
                    print(f"[{worker_id}] FINALIZED job {job_key} -> run_id={outcome.result_ref}")
                last_work_at = time.time()
        except Exception as exc:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                repo.fail_orchestration_job(job_id, error_json={"error": str(exc)[:1000]})
            print(f"[{worker_id}] JOB ERROR {job_key}: {exc}")

    for provider in provider_cache.values():
        try:
            _run_async_sync(lambda p=provider: p.close())
        except Exception:
            pass
    for manager in manager_cache.values():
        if manager is None:
            continue
        try:
            manager.stop()
        except Exception:
            pass

    print(
        f"[{worker_id}] Sharded worker finished. "
        f"completed_leaf_jobs={completed_leaf_jobs}, elapsed={int(time.time()-started_at)}s"
    )
    return completed_leaf_jobs


def _iter_engine_order(target_engine: Optional[str]) -> list[str]:
    if target_engine:
        return [target_engine]
    return EMBEDDING_ENGINES


def _build_provider_for_engine(
    *,
    embedding_engine: str,
    shared_endpoint: Optional[str],
    provider_label: str,
    worker_slot: int,
    embedding_provider_name: Optional[str] = None,
):
    """Return (context_manager_or_None, provider) for one embedding engine."""
    if shared_endpoint:
        manager = _SharedEndpointManager(
            endpoint_url=shared_endpoint,
            model_id=embedding_engine,
            provider_label=provider_label,
        )
        provider = ManagedTEIEmbeddingProvider(manager)
        return None, provider

    if embedding_provider_name:
        factory = ProviderFactory()
        provider = factory.create_embedding_provider(embedding_provider_name)
        return None, provider

    model_safe = embedding_engine.replace("/", "-").replace(".", "-").lower()
    manager = LocalDockerTEIManager(
        model_id=embedding_engine,
        use_gpu=True,
        port=8080 + int(worker_slot),
        container_name=f"tei-{model_safe}-w{int(worker_slot)}",
    )
    manager.start()
    provider = ManagedTEIEmbeddingProvider(manager)
    return manager, provider


def _run_mcq_standalone_worker_loop(
    *,
    request_id: int,
    worker_id: str,
    claim_lease_seconds: int,
    max_runs: Optional[int],
    idle_exit_seconds: int,
    force: bool,
) -> int:
    """Standalone MCQ: claim each missing run_key, execute probe, persist mcq_run."""
    done = 0
    processed = 0
    started_at = time.time()
    last_work_at = started_at

    while True:
        pairs, _progress = _missing_run_targets(request_id)
        if not pairs:
            print(f"[{worker_id}] Nothing missing. Request already fulfilled.")
            break

        pairs.sort(key=lambda x: x[0])
        made_progress = False
        for run_key, target in pairs:
            if max_runs is not None and done >= max_runs:
                break

            if (not force) and mcq_run_key_exists_in_db(db, run_key):
                print(f"[{worker_id}] {run_key} SKIP (mcq_run in DB)")
                processed += 1
                continue

            if not _claim_run_target(
                request_id=request_id,
                run_key=run_key,
                worker_id=worker_id,
                lease_seconds=claim_lease_seconds,
            ):
                continue

            run_id, err = execute_mcq_standalone_run(
                db=db,
                request_id=request_id,
                run_key=run_key,
                target=target,
                worker_label=worker_id,
            )
            if err:
                _fail_run_claim(request_id, run_key, worker_id, err)
                print(f"[{worker_id}] {run_key} ERROR: {err}")
                continue
            if run_id is None:
                _fail_run_claim(request_id, run_key, worker_id, "persist_mcq_failed")
                continue

            _complete_run_claim(request_id, run_key, run_id, worker_id)
            done += 1
            processed += 1
            last_work_at = time.time()
            made_progress = True
            print(f"[{worker_id}] DONE {done}: {run_key} -> run_id={run_id}")

        if max_runs is not None and done >= max_runs:
            break
        if not made_progress:
            idle_for = int(time.time() - last_work_at)
            if idle_for >= idle_exit_seconds:
                print(f"[{worker_id}] Idle exit after {idle_for}s with no claimable work.")
                break
            time.sleep(5)

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        sweep_id = svc.finalize_if_fulfilled(
            request_id,
            sweep_name=f"mcq_sweep_{datetime.now().strftime('%Y%m%d')}",
        )
        if sweep_id:
            print(f"[{worker_id}] Request fulfilled -> sweep_id={sweep_id}")

    print(
        f"[{worker_id}] Finished (mcq). completed={done}, processed={processed}, "
        f"elapsed={int(time.time()-started_at)}s"
    )
    return done


def _run_standalone_worker_loop(
    *,
    request_id: int,
    worker_id: str,
    worker_slot: int,
    embedding_engine: Optional[str],
    tei_endpoint: Optional[str],
    provider_label: str,
    embedding_provider_name: Optional[str],
    claim_lease_seconds: int,
    max_runs: Optional[int],
    idle_exit_seconds: int,
    force: bool,
    repo_root: Path,
) -> int:
    """Standalone mode: dispatch clustering vs MCQ by request sweep_type."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req = svc.get_request(request_id)
        planned = svc.ensure_orchestration_jobs(request_id)
        job_count = len(repo.list_orchestration_jobs(request_group_id=request_id))
    sweep_type = (req or {}).get("sweep_type") or SWEEP_TYPE_CLUSTERING
    execution_mode = str((req or {}).get("execution_mode") or "standalone").lower()

    # Standalone is modeled as an orchestration profile when planned jobs exist.
    if job_count > 0:
        print(
            f"[{worker_id}] Routing standalone->orchestration jobs "
            f"(mode={execution_mode}, planned_now={planned}, jobs={job_count})"
        )
        return _run_sharded_worker_loop(
            request_id=request_id,
            worker_id=worker_id,
            worker_slot=worker_slot,
            embedding_engine=embedding_engine,
            tei_endpoint=tei_endpoint,
            provider_label=provider_label,
            embedding_provider_name=embedding_provider_name,
            claim_lease_seconds=claim_lease_seconds,
            max_runs=max_runs,
            idle_exit_seconds=idle_exit_seconds,
            force=force,
            repo_root=repo_root,
        )

    if sweep_type == SWEEP_TYPE_MCQ:
        return _run_mcq_standalone_worker_loop(
            request_id=request_id,
            worker_id=worker_id,
            claim_lease_seconds=claim_lease_seconds,
            max_runs=max_runs,
            idle_exit_seconds=idle_exit_seconds,
            force=force,
        )
    return _run_clustering_standalone_worker_loop(
        request_id=request_id,
        worker_id=worker_id,
        worker_slot=worker_slot,
        embedding_engine=embedding_engine,
        tei_endpoint=tei_endpoint,
        provider_label=provider_label,
        embedding_provider_name=embedding_provider_name,
        claim_lease_seconds=claim_lease_seconds,
        max_runs=max_runs,
        idle_exit_seconds=idle_exit_seconds,
        force=force,
        repo_root=repo_root,
    )


def _run_clustering_standalone_worker_loop(
    *,
    request_id: int,
    worker_id: str,
    worker_slot: int,
    embedding_engine: Optional[str],
    tei_endpoint: Optional[str],
    provider_label: str,
    embedding_provider_name: Optional[str],
    claim_lease_seconds: int,
    max_runs: Optional[int],
    idle_exit_seconds: int,
    force: bool,
    repo_root: Path,
) -> int:
    """Clustering-only standalone loop (datasets + embeddings + run_sweep)."""
    loaded = _load_datasets(repo_root)
    done = 0
    processed = 0
    started_at = time.time()
    last_work_at = started_at
    engines = _iter_engine_order(embedding_engine)
    provider_cache: Dict[str, object] = {}
    manager_cache: Dict[str, object] = {}

    while True:
        run_rows, progress = _missing_run_targets(request_id)
        if not run_rows:
            print(f"[{worker_id}] Nothing missing. Request already fulfilled.")
            break

        by_engine: Dict[str, list] = {}
        for run_key, target in run_rows:
            eng = target.get("embedding_engine")
            if eng:
                by_engine.setdefault(str(eng), []).append((run_key, target))

        made_progress = False
        for current_engine in engines:
            rows = by_engine.get(current_engine, [])
            if not rows:
                continue
            rows.sort(key=lambda x: x[0])

            if current_engine not in provider_cache:
                manager, provider = _build_provider_for_engine(
                    embedding_engine=current_engine,
                    shared_endpoint=tei_endpoint,
                    provider_label=provider_label,
                    worker_slot=worker_slot,
                    embedding_provider_name=embedding_provider_name,
                )
                manager_cache[current_engine] = manager
                provider_cache[current_engine] = provider

            emb_provider = provider_cache[current_engine]
            for run_key, target in rows:
                if max_runs is not None and done >= max_runs:
                    break

                dataset_name = str(target.get("dataset") or "")
                summarizer_key = target.get("summarizer", "None")

                if (not force) and run_key_exists_in_db(db, run_key):
                    print(f"[{worker_id}] {run_key} SKIP (run_key in DB)")
                    processed += 1
                    continue

                if not _claim_run_target(
                    request_id=request_id,
                    run_key=run_key,
                    worker_id=worker_id,
                    lease_seconds=claim_lease_seconds,
                ):
                    continue

                summarizer_name = None if summarizer_key == "None" else summarizer_key
                dataset = loaded.get(dataset_name)
                if dataset is None:
                    _fail_run_claim(request_id, run_key, worker_id, "dataset_not_loaded")
                    continue

                texts = dataset["texts"]
                labels = dataset["labels"]
                gt = labels if dataset["has_gt"] else None
                label_max = dataset["label_max"]

                def _embed_sync(txts):
                    dataset_key = f"{dataset_name}:entry_max={len(txts)}"
                    return _run_async_sync(
                        lambda: _fetch_embeddings(
                            txts,
                            current_engine,
                            emb_provider,
                            dataset_key=dataset_key,
                            provider_name=provider_label,
                        )
                    )

                try:
                    embeddings = _embed_sync(texts)
                    paraphraser = (
                        create_paraphraser_for_llm(summarizer_name, db) if summarizer_name else None
                    )
                    result = run_sweep(
                        texts,
                        embeddings,
                        SWEEP_CONFIG,
                        paraphraser=paraphraser,
                        embedder=_embed_sync if paraphraser else None,
                    )
                except Exception as exc:
                    _fail_run_claim(request_id, run_key, worker_id, str(exc))
                    print(f"[{worker_id}] {run_key} ERROR: {exc}")
                    continue

                metadata = {
                    "embedding_engine": current_engine,
                    "embedding_provider": provider_label,
                    "summarizer": str(summarizer_key),
                    "n_restarts": N_RESTARTS,
                    "request_group_id": int(request_id),
                    "determinism_class": "pseudo_deterministic",
                    "k_min": K_MIN,
                    "k_max": K_MAX,
                    "entry_max": ENTRY_MAX,
                    "dataset_name": dataset_name,
                    "label_max": label_max,
                    "n_texts": len(texts),
                    "distance_metric": "cosine",
                    "normalize_vectors": True,
                    "skip_pca": True,
                    "benchmark_source": dataset_name,
                    "actual_entry_count": len(texts),
                    "sweep_config": {"k_min": K_MIN, "k_max": K_MAX, "n_restarts": N_RESTARTS},
                }
                run_id = ingest_result_to_db(result, metadata, gt, db, run_key)
                if run_id is not None:
                    with db.session_scope() as session:
                        repo = RawCallRepository(session)
                        svc = SweepRequestService(repo)
                        svc.record_delivery(request_id, run_id, run_key)
                    _complete_run_claim(request_id, run_key, run_id, worker_id)
                    done += 1
                    processed += 1
                    last_work_at = time.time()
                    made_progress = True
                    print(f"[{worker_id}] DONE {done}: {run_key} -> run_id={run_id}")

            if max_runs is not None and done >= max_runs:
                break

        # Exit conditions
        if max_runs is not None and done >= max_runs:
            break
        if embedding_engine:
            refreshed_pairs, _ = _missing_run_targets(request_id)
            engine_missing = [
                p
                for p in refreshed_pairs
                if p[1].get("embedding_engine") == embedding_engine
            ]
            if not engine_missing:
                print(f"[{worker_id}] Engine exhausted: {embedding_engine}")
                break
        if not made_progress:
            idle_for = int(time.time() - last_work_at)
            if idle_for >= idle_exit_seconds:
                print(f"[{worker_id}] Idle exit after {idle_for}s with no claimable work.")
                break
            time.sleep(5)

    # Cleanup providers/managers
    for provider in provider_cache.values():
        try:
            _run_async_sync(lambda p=provider: p.close())
        except Exception:
            pass
    for manager in manager_cache.values():
        if manager is None:
            continue
        try:
            manager.stop()
        except Exception:
            pass

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        sweep_id = svc.finalize_if_fulfilled(
            request_id,
            sweep_name=f"{OUT_PREFIX}_sweep_{datetime.now().strftime('%Y%m%d')}",
        )
        if sweep_id:
            print(f"[{worker_id}] Request fulfilled -> sweep_id={sweep_id}")

    print(
        f"[{worker_id}] Finished. completed={done}, processed={processed}, "
        f"elapsed={int(time.time()-started_at)}s"
    )
    return done


def run_worker(
    request_id: int,
    worker_id: str,
    worker_slot: int,
    job_mode: str,
    embedding_engine: Optional[str],
    tei_endpoint: Optional[str],
    provider_label: str,
    embedding_provider_name: Optional[str],
    claim_lease_seconds: int,
    max_runs: Optional[int],
    idle_exit_seconds: int,
    force: bool,
    repo_root: Path,
) -> int:
    """Delegate to factory-created orchestrator based on job_mode."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        svc.ensure_orchestration_jobs(request_id)

    from study_query_llm.services.worker_orchestrator import create_worker_orchestrator

    orchestrator = create_worker_orchestrator(
        mode=job_mode,
        run_standalone_fn=_run_standalone_worker_loop,
        run_sharded_fn=_run_sharded_worker_loop,
        request_id=request_id,
        worker_id=worker_id,
        worker_slot=worker_slot,
        embedding_engine=embedding_engine,
        tei_endpoint=tei_endpoint,
        provider_label=provider_label,
        embedding_provider_name=embedding_provider_name,
        claim_lease_seconds=claim_lease_seconds,
        max_runs=max_runs,
        idle_exit_seconds=idle_exit_seconds,
        force=force,
        repo_root=repo_root,
    )
    return orchestrator.run()


def build_sweep_worker_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep request worker (clustering and MCQ, standalone or orchestration jobs)."
    )
    parser.add_argument("--request-id", type=int, required=True, help="Sweep request id")
    parser.add_argument("--worker-id", type=str, default=None, help="Worker identifier")
    parser.add_argument(
        "--worker-slot",
        type=int,
        default=0,
        help="Numeric worker slot used to isolate TEI port/container (port = 8080 + slot)",
    )
    parser.add_argument("--claim-lease-seconds", type=int, default=3600)
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after this many completed runs")
    parser.add_argument(
        "--job-mode",
        type=str,
        default="standalone",
        choices=["standalone", "sharded"],
        help="Execution mode: standalone run_key claims or sharded orchestration jobs.",
    )
    parser.add_argument(
        "--embedding-engine",
        type=str,
        default=None,
        help="Restrict worker to a single embedding engine (used by supervisor mode)",
    )
    parser.add_argument(
        "--tei-endpoint",
        type=str,
        default=None,
        help="Shared TEI endpoint URL (e.g. http://localhost:8080/v1). "
        "When set, worker will not manage Docker container lifecycle.",
    )
    parser.add_argument(
        "--provider-label",
        type=str,
        default="local_docker_tei",
        help="Provider label stored in metadata (use local_docker_tei_shared in shared mode).",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=None,
        help="Remote/shared embedding provider name (e.g. azure, openai, huggingface, local, ollama). "
        "When set and --tei-endpoint is not provided, worker will not start local TEI containers.",
    )
    parser.add_argument(
        "--idle-exit-seconds",
        type=int,
        default=90,
        help="Exit when no claimable work is found for this many seconds.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing pkl files")
    return parser


def main(argv: Optional[list] = None) -> None:
    """CLI entry: set global db, run worker loop."""
    global db
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")
    parser = build_sweep_worker_arg_parser()
    args = parser.parse_args(argv)

    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    root = Path(__file__).resolve().parents[3]
    worker_id = args.worker_id or f"sweep-worker-{os.getpid()}"

    run_worker(
        request_id=args.request_id,
        worker_id=worker_id,
        worker_slot=args.worker_slot,
        job_mode=args.job_mode,
        embedding_engine=args.embedding_engine,
        tei_endpoint=args.tei_endpoint,
        provider_label=args.provider_label,
        embedding_provider_name=args.embedding_provider,
        claim_lease_seconds=args.claim_lease_seconds,
        max_runs=args.max_runs,
        idle_exit_seconds=args.idle_exit_seconds,
        force=args.force,
        repo_root=root,
    )


if __name__ == "__main__":
    main()
