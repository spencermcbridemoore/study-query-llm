"""
300-sample bigrun sweep: 3 datasets × 3 embeddings × 5 summarizers, 50 restarts each.

Datasets  : dbpedia (14 cats), yahoo_answers (10 cats), estela (no GT labels)
Embeddings: embed-v-4-0, text-embedding-3-large, text-embedding-3-small
Summarizers: None, gpt-4o-mini, gpt-4o, gpt-5-chat, claude-opus-4-6
Restarts  : 50  |  Sample: 300  |  k: 2–20  |  cosine, no PCA

After each run:
  1. Pkl written to experimental_results/ (compute is safe on disk first).
  2. NeonDB ingestion from the in-memory result (Group/GroupLink via ProvenanceService).

Total sweeps: 3 × 3 × 5 = 45 (each 50-restart sweep over k=2..20).
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import SweepRunClaim
from study_query_llm.services.embedding_service import estimate_tokens, DEPLOYMENT_MAX_TOKENS
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.services.sweep_request_service import SweepRequestService
from study_query_llm.algorithms import SweepConfig

from study_query_llm.services.embedding_helpers import fetch_embeddings_async
from study_query_llm.services.paraphraser_factory import create_paraphraser_for_llm
from study_query_llm.experiments.sweep_io import save_single_sweep_result as save_pkl, get_output_dir
from study_query_llm.experiments.ingestion import ingest_result_to_db, run_key_exists_in_db

OUTPUT_DIR = get_output_dir()
from study_query_llm.utils.estela_loader import load_estela_dict
from study_query_llm.utils.text_utils import flatten_prompt_dict
from scripts.run_experimental_sweep import (
    load_dbpedia_full,
    load_yahoo_answers_full,
)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTRY_MAX = 300
N_RESTARTS = 50
K_MIN, K_MAX = 2, 20
OUT_PREFIX = "bigrun_300"

EMBEDDING_ENGINES = [
    "embed-v-4-0",
    "text-embedding-3-large",
    "text-embedding-3-small",
]

SUMMARIZERS = [
    None,
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-chat",
    "claude-opus-4-6",
]

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


# ---------------------------------------------------------------------------
# Worker claim helpers (request mode)
# ---------------------------------------------------------------------------

def _now_utc():
    return datetime.now(timezone.utc)


def _claim_run_target(
    db: DatabaseConnectionV2,
    request_id: int,
    run_key: str,
    worker_id: str,
    lease_seconds: int,
) -> bool:
    """Try to claim a run target for this worker.

    Returns True if this worker should execute the run, False otherwise.
    """
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

        # Take over expired or failed/released claims.
        claim.claim_status = "claimed"
        claim.claimed_by = worker_id
        claim.claimed_at = now
        claim.lease_expires_at = lease_expires
        claim.heartbeat_at = now
        session.flush()
        return True


def _complete_run_claim(
    db: DatabaseConnectionV2,
    request_id: int,
    run_key: str,
    run_id: int,
    worker_id: str,
) -> None:
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
            claim = SweepRunClaim(
                request_group_id=request_id,
                run_key=run_key,
            )
            session.add(claim)
        claim.claim_status = "completed"
        claim.claimed_by = worker_id
        claim.run_group_id = run_id
        claim.heartbeat_at = _now_utc()
        claim.lease_expires_at = None
        session.flush()


def _fail_run_claim(
    db: DatabaseConnectionV2,
    request_id: int,
    run_key: str,
    worker_id: str,
    error_message: str,
) -> None:
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
            claim = SweepRunClaim(
                request_group_id=request_id,
                run_key=run_key,
            )
            session.add(claim)
        metadata = dict(claim.metadata_json or {})
        metadata["last_error"] = error_message[:500]
        metadata["failed_at"] = _now_utc().isoformat()
        claim.metadata_json = metadata
        claim.claim_status = "failed"
        claim.claimed_by = worker_id
        claim.heartbeat_at = _now_utc()
        claim.lease_expires_at = None
        session.flush()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    return s.replace("-", "_").replace("/", "_")


def _load_estela_with_labels():
    """Load estela texts from the prompt dict. Returns (texts, labels, category_names).

    Labels are all zeros because estela has no categorical ground truth.
    Category names is an empty list for the same reason.
    """
    pkl_path = str(Path(__file__).parent.parent / "notebooks" / "estela_prompt_data.pkl")
    data = load_estela_dict(pkl_path=pkl_path)
    flat = flatten_prompt_dict(data)
    texts = [t for t in flat.values() if isinstance(t, str)]
    texts = [t.replace("\x00", "").strip() for t in texts]
    texts = [t for t in texts if 10 < len(t) <= 1000]
    labels = np.zeros(len(texts), dtype=np.int64)
    return texts, labels, []


DATASETS = [
    {
        "name": "dbpedia",
        "label_max": 14,
        "loader": load_dbpedia_full,
        "has_gt": True,
    },
    {
        "name": "yahoo_answers",
        "label_max": 10,
        "loader": load_yahoo_answers_full,
        "has_gt": True,
    },
    {
        "name": "estela",
        "label_max": None,   # no ground-truth categories
        "loader": _load_estela_with_labels,
        "has_gt": False,
    },
]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

async def _run_sweep(texts, embeddings, llm_deployment, db, embedding_engine):
    from study_query_llm.algorithms.sweep import run_sweep

    paraphraser = create_paraphraser_for_llm(llm_deployment, db)

    async def _embed_async(texts_list):
        return await fetch_embeddings_async(texts_list, embedding_engine, db)

    def _embed_sync(texts_list):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(_embed_async(texts_list))
        except RuntimeError:
            pass
        return asyncio.run(_embed_async(texts_list))

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: run_sweep(
                texts, embeddings, SWEEP_CONFIG,
                paraphraser=paraphraser,
                embedder=_embed_sync if paraphraser else None,
            ),
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    force: bool = False,
    save_local_pkl: bool = False,
    request_id: Optional[int] = None,
    create_request: bool = False,
    request_name: Optional[str] = None,
    worker_id: Optional[str] = None,
    claim_lease_seconds: int = 3600,
):
    worker_id = worker_id or f"{os.environ.get('COMPUTERNAME', 'worker')}-{os.getpid()}"

    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()

    # ------------------------------------------------------------------
    # --create-request: create request metadata and exit
    # ------------------------------------------------------------------
    if create_request:
        name = request_name or f"{OUT_PREFIX}_request_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            svc = SweepRequestService(repo)
            rid = svc.create_request(
                request_name=name,
                algorithm="cosine_kllmeans_no_pca",
                fixed_config={
                    "n_samples": ENTRY_MAX,
                    "n_restarts": N_RESTARTS,
                    "k_min": K_MIN,
                    "k_max": K_MAX,
                    "skip_pca": True,
                    "distance_metric": "cosine",
                    "normalize_vectors": True,
                    "llm_interval": 20,
                },
                parameter_axes={
                    "datasets": [d["name"] for d in DATASETS],
                    "embedding_engines": EMBEDDING_ENGINES,
                    "summarizers": [str(s) if s is not None else "None" for s in SUMMARIZERS],
                },
                entry_max=ENTRY_MAX,
                n_restarts_suffix="50runs",
                description=f"{OUT_PREFIX} sweep request: 3 datasets x 3 embeddings x 5 summarizers",
            )
        print(f"[OK] Created sweep request: id={rid}, name={name}")
        return

    total = len(DATASETS) * len(EMBEDDING_ENGINES) * len(SUMMARIZERS)

    print("=" * 80)
    print(f"300-sample Bigrun Sweep  ({total} sweeps, each {N_RESTARTS} restarts)")
    print("=" * 80)
    print(f"  entry_max   : {ENTRY_MAX}")
    print(f"  k_range     : {K_MIN}–{K_MAX}")
    print(f"  n_restarts  : {N_RESTARTS}")
    print(f"  datasets    : {[d['name'] for d in DATASETS]}")
    print(f"  embeddings  : {EMBEDDING_ENGINES}")
    print(f"  summarizers : {SUMMARIZERS}")
    print(f"  mode        : skip_pca=True, cosine, normalize")
    if request_id:
        print(f"  request_id  : {request_id} (missing-only mode)")
        print(f"  worker_id   : {worker_id}")
        print(f"  claim_lease : {claim_lease_seconds}s")

    print("\n[OK] Database initialised")

    # ------------------------------------------------------------------
    # Load all datasets up-front (texts + labels, sampled to ENTRY_MAX)
    # ------------------------------------------------------------------
    loaded: dict = {}
    for bench in DATASETS:
        name = bench["name"]
        print(f"\n[INFO] Loading {name} ...")
        try:
            texts_all, labels_all, category_names = bench["loader"]()
        except Exception as exc:
            print(f"  [ERROR] Failed to load {name}: {exc}")
            continue

        if bench["label_max"] is not None:
            unique_labels = sorted(set(labels_all))
            label_max = min(bench["label_max"], len(unique_labels))
            mask = np.isin(labels_all, unique_labels[:label_max])
            idx = np.where(mask)[0]
        else:
            # estela: take everything, no label filtering
            idx = np.arange(len(texts_all))
            label_max = 0

        if len(idx) > ENTRY_MAX:
            np.random.seed(42)
            idx = np.random.choice(idx, size=ENTRY_MAX, replace=False)

        texts = [texts_all[i] for i in idx]
        labels = labels_all[idx]

        # Character-length filter (consistent with other sweep scripts)
        valid = [i for i, t in enumerate(texts) if 10 < len(t) <= 1000]
        if len(valid) < len(texts):
            texts = [texts[i] for i in valid]
            labels = labels[valid]

        loaded[name] = {
            "texts": texts,
            "labels": labels,
            "label_max": label_max,
            "category_names": category_names,
            "has_gt": bench["has_gt"],
        }
        print(f"  {len(texts)} texts, {len(set(labels))} unique labels")

    # ------------------------------------------------------------------
    # Build task list: all combos (legacy) or missing-only (request mode)
    # ------------------------------------------------------------------
    tasks_to_run: list[tuple[str, str, str]] = []  # (dataset, engine, summarizer)
    if request_id:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            svc = SweepRequestService(repo)
            progress = svc.compute_progress(request_id)
            missing = progress.get("missing_run_keys") or []
            run_key_to_target = (svc.get_request(request_id) or {}).get("run_key_to_target") or {}
        for rk in missing:
            t = run_key_to_target.get(rk)
            if t:
                tasks_to_run.append((
                    t["dataset"],
                    t["embedding_engine"],
                    t["summarizer"],
                ))
        if not tasks_to_run:
            print(f"\n[OK] Request {request_id} already fulfilled. Nothing to run.")
            return
        total_to_run = len(tasks_to_run)
        print(f"\n[REQUEST] {total_to_run} missing runs to execute (of {progress['expected_count']} total)")
    else:
        for dataset_name in loaded:
            for embedding_engine in EMBEDDING_ENGINES:
                for llm in SUMMARIZERS:
                    summarizer_name = "None" if llm is None else str(llm)
                    tasks_to_run.append((dataset_name, embedding_engine, summarizer_name))
        total_to_run = len(tasks_to_run)

    # ------------------------------------------------------------------
    # Main sweep loop over tasks
    # ------------------------------------------------------------------
    run_ids_ingested: list[int] = []
    run_count = 0
    current_dataset = None
    current_engine = None
    embeddings_cache = None

    for dataset_name, embedding_engine, summarizer_name in tasks_to_run:
        run_count += 1
        engine_safe = _safe_name(embedding_engine)
        sum_safe = _safe_name(summarizer_name)
        llm = None if summarizer_name == "None" else summarizer_name

        if dataset_name not in loaded:
            print(f"  [{run_count}/{total_to_run}] {dataset_name} (SKIP – not loaded)")
            continue

        info = loaded[dataset_name]
        texts = info["texts"]
        labels = info["labels"]
        label_max = info["label_max"]
        has_gt = info["has_gt"]
        gt_labels = labels if has_gt else None

        # Token-length filter for this engine
        max_tokens = DEPLOYMENT_MAX_TOKENS.get(embedding_engine)
        if max_tokens:
            valid_idx = []
            for i, t in enumerate(texts):
                try:
                    if estimate_tokens(t, embedding_engine) <= max_tokens:
                        valid_idx.append(i)
                except Exception:
                    valid_idx.append(i)
            texts_eng = [texts[i] for i in valid_idx]
            labels_eng = labels[valid_idx] if has_gt else labels[valid_idx]
            gt_eng = labels_eng if has_gt else None
        else:
            texts_eng = texts
            gt_eng = gt_labels

        run_key = f"{dataset_name}_{engine_safe}_{sum_safe}_{ENTRY_MAX}_50runs"
        out_name = (
            f"{OUT_PREFIX}_entry{ENTRY_MAX}_{dataset_name}"
            f"_{engine_safe}_{sum_safe}_"
        )

        # Skip if run_key already in DB (idempotency) and not forcing
        if not force and run_key_exists_in_db(db, run_key):
            print(f"  [{run_count}/{total_to_run}] {summarizer_name} (SKIP – run_key in DB)")
            continue

        if request_id:
            claimed = _claim_run_target(
                db=db,
                request_id=request_id,
                run_key=run_key,
                worker_id=worker_id,
                lease_seconds=claim_lease_seconds,
            )
            if not claimed:
                print(
                    f"  [{run_count}/{total_to_run}] {dataset_name} / {embedding_engine} / "
                    f"{summarizer_name} (SKIP – claimed by another worker or already completed)"
                )
                continue

        print(f"\n  [{run_count}/{total_to_run}] {dataset_name} / {embedding_engine} / {summarizer_name}")

        # Fetch embeddings (reuse if same dataset+engine as previous)
        if (current_dataset, current_engine) != (dataset_name, embedding_engine):
            try:
                embeddings_cache = await fetch_embeddings_async(texts_eng, embedding_engine, db)
                current_dataset, current_engine = dataset_name, embedding_engine
            except Exception as exc:
                print(f"  [ERROR] Embedding fetch failed: {exc}")
                if request_id:
                    _fail_run_claim(db, request_id, run_key, worker_id, str(exc))
                continue
        embeddings = embeddings_cache

        try:
            result = await asyncio.wait_for(
                _run_sweep(texts_eng, embeddings, llm, db, embedding_engine),
                timeout=7200.0,
            )
        except asyncio.TimeoutError:
            print(f"  [ERROR] Timed out after 2 h – skipping")
            if request_id:
                _fail_run_claim(db, request_id, run_key, worker_id, "timeout after 2h")
            continue
        except Exception as exc:
            import traceback
            print(f"  [ERROR] Sweep failed: {exc}")
            traceback.print_exc()
            if request_id:
                _fail_run_claim(db, request_id, run_key, worker_id, str(exc))
            continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{out_name}{ts}.pkl"

        metadata = {
            "entry_max": ENTRY_MAX,
            "label_max": label_max,
            "actual_entry_count": len(texts_eng),
            "actual_label_count": len(set(gt_eng)) if gt_eng is not None else 0,
            "benchmark_source": dataset_name,
            "summarizer": summarizer_name,
            "embedding_engine": embedding_engine,
            "n_restarts": N_RESTARTS,
            "sweep_config": {
                "skip_pca": True,
                "k_min": K_MIN,
                "k_max": K_MAX,
                "n_restarts": N_RESTARTS,
                "compute_stability": True,
            },
            "note": "300-sample bigrun sweep: 3 datasets × 3 embeddings × 5 summarizers",
        }

        # 1. Ingest to DB (artifact persist via ArtifactService is primary)
        run_id = ingest_result_to_db(result, metadata, gt_eng, db, run_key)

        # 2. Optional local pickle backup (off by default)
        if save_local_pkl and run_id is not None:
            save_pkl(
                result,
                str(out_path),
                ground_truth_labels=gt_eng,
                dataset_name=dataset_name,
                metadata=metadata,
            )
            print(f"  [PKL] {out_path.name}")
        if run_id is not None:
            run_ids_ingested.append(run_id)
            if request_id:
                with db.session_scope() as session:
                    repo = RawCallRepository(session)
                    svc = SweepRequestService(repo)
                    svc.record_delivery(request_id, run_id, run_key)
                _complete_run_claim(db, request_id, run_key, run_id, worker_id)

    print(f"\n{'='*80}")
    print("[OK] All runs complete.")
    print(f"  Total sweeps executed: {run_count}/{total_to_run}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Request mode: finalize if fulfilled
    # ------------------------------------------------------------------
    if request_id and run_ids_ingested:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            svc = SweepRequestService(repo)
            sweep_id = svc.finalize_if_fulfilled(
                request_id,
                sweep_name=f"{OUT_PREFIX}_sweep_{datetime.now().strftime('%Y%m%d')}",
            )
            if sweep_id:
                print(f"  [REQUEST] Fulfilled -> clustering_sweep id={sweep_id}")

    # ------------------------------------------------------------------
    # Legacy mode: Register all newly ingested runs under a clustering_sweep group
    # ------------------------------------------------------------------
    if run_ids_ingested and not request_id:
        ts = datetime.now().strftime("%Y%m%d")
        sweep_name = f"{OUT_PREFIX}_sweep_{ts}"
        try:
            with db.session_scope() as session:
                from study_query_llm.db.models_v2 import Group, GroupLink
                provenance = ProvenanceService(RawCallRepository(session))

                existing = (
                    session.query(Group)
                    .filter(
                        Group.group_type == "clustering_sweep",
                        Group.name == sweep_name,
                    )
                    .first()
                )
                if existing:
                    sweep_id = existing.id
                    print(f"  [SWEEP] Using existing clustering_sweep '{sweep_name}' (id={sweep_id})")
                else:
                    sweep_id = provenance.create_clustering_sweep_group(
                        sweep_name=sweep_name,
                        algorithm="cosine_kllmeans_no_pca",
                        fixed_config={
                            "n_samples": ENTRY_MAX,
                            "n_restarts": N_RESTARTS,
                            "k_min": K_MIN,
                            "k_max": K_MAX,
                            **(SWEEP_CONFIG.__dict__ if hasattr(SWEEP_CONFIG, "__dict__") else {}),
                        },
                        parameter_axes={
                            "datasets": list(loaded.keys()),
                            "embedding_engines": EMBEDDING_ENGINES,
                            "summarizers": [str(s) if s is not None else "None" for s in SUMMARIZERS],
                        },
                        description=(
                            f"{OUT_PREFIX} sweep: {len(loaded)} datasets x "
                            f"{len(EMBEDDING_ENGINES)} embeddings x "
                            f"{len(SUMMARIZERS)} summarizers, "
                            f"{N_RESTARTS} restarts each."
                        ),
                    )
                    print(f"  [SWEEP] Created clustering_sweep '{sweep_name}' (id={sweep_id})")

                linked = 0
                for pos, run_id in enumerate(run_ids_ingested):
                    existing_link = (
                        session.query(GroupLink)
                        .filter_by(
                            parent_group_id=sweep_id,
                            child_group_id=run_id,
                            link_type="contains",
                        )
                        .first()
                    )
                    if not existing_link:
                        provenance.link_run_to_clustering_sweep(sweep_id, run_id, position=pos)
                        linked += 1

                print(f"  [SWEEP] Linked {linked} new runs to sweep '{sweep_name}'.")
        except Exception as exc:
            import traceback
            print(f"  [WARN] Failed to create/update clustering_sweep: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="300-sample bigrun sweep (3×3×5, 50 restarts)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if run_key already exists in DB",
    )
    parser.add_argument(
        "--save-local-pkl",
        action="store_true",
        help="Also write local pickle backup (off by default; blob-first is primary)",
    )
    parser.add_argument(
        "--request-id",
        type=int,
        default=None,
        help="Execute only missing runs for this sweep request ID",
    )
    parser.add_argument(
        "--create-request",
        action="store_true",
        help="Create sweep request metadata and exit (no runs)",
    )
    parser.add_argument(
        "--request-name",
        type=str,
        default=None,
        help="Name for new request when using --create-request",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker identifier for claim/lease tracking in request mode",
    )
    parser.add_argument(
        "--claim-lease-seconds",
        type=int,
        default=3600,
        help="Lease duration in seconds for request-mode run claims (default 3600)",
    )
    args = parser.parse_args()
    asyncio.run(main(
        force=args.force,
        save_local_pkl=args.save_local_pkl,
        request_id=args.request_id,
        create_request=args.create_request,
        request_name=args.request_name,
        worker_id=args.worker_id,
        claim_lease_seconds=args.claim_lease_seconds,
    ))
