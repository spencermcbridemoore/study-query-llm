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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.services.embedding_service import estimate_tokens, DEPLOYMENT_MAX_TOKENS
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.algorithms import SweepConfig

from study_query_llm.services.embedding_helpers import fetch_embeddings_async
from scripts.common.sweep_utils import (
    create_paraphraser_for_llm,
    save_single_sweep_result as save_pkl,
    ingest_result_to_db,
    OUTPUT_DIR,
)
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

async def main(force: bool = False):
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

    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
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
    # Main sweep loop: dataset → embedding → summarizer
    # ------------------------------------------------------------------
    run_ids_ingested: list[int] = []
    run_count = 0
    for dataset_name, info in loaded.items():
        texts = info["texts"]
        labels = info["labels"]
        label_max = info["label_max"]
        has_gt = info["has_gt"]
        gt_labels = labels if has_gt else None

        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print("=" * 80)

        for embedding_engine in EMBEDDING_ENGINES:
            engine_safe = _safe_name(embedding_engine)

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

            print(f"\n  EMBEDDING: {embedding_engine} ({len(texts_eng)} texts after token filter)")

            try:
                embeddings = await fetch_embeddings_async(texts_eng, embedding_engine, db)
            except Exception as exc:
                print(f"  [ERROR] Embedding fetch failed: {exc}")
                continue

            for llm in SUMMARIZERS:
                run_count += 1
                summarizer_name = "None" if llm is None else llm
                sum_safe = _safe_name(summarizer_name)

                run_key = f"{dataset_name}_{engine_safe}_{sum_safe}_{ENTRY_MAX}_50runs"
                out_name = (
                    f"{OUT_PREFIX}_entry{ENTRY_MAX}_{dataset_name}"
                    f"_{engine_safe}_{sum_safe}_"
                )

                # Skip if pkl backup already on disk AND not forcing
                if not force and list(OUTPUT_DIR.glob(out_name + "*.pkl")):
                    print(f"  [{run_count}/{total}] {summarizer_name} (SKIP – pkl exists)")
                    continue

                print(f"\n  [{run_count}/{total}] {dataset_name} / {embedding_engine} / {summarizer_name}")

                try:
                    result = await asyncio.wait_for(
                        _run_sweep(texts_eng, embeddings, llm, db, embedding_engine),
                        timeout=7200.0,
                    )
                except asyncio.TimeoutError:
                    print(f"  [ERROR] Timed out after 2 h – skipping")
                    continue
                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] Sweep failed: {exc}")
                    traceback.print_exc()
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

                # 1. Save pkl first (compute is safe on disk)
                save_pkl(
                    result,
                    str(out_path),
                    ground_truth_labels=gt_eng,
                    dataset_name=dataset_name,
                    metadata=metadata,
                )
                print(f"  [PKL] {out_path.name}")

                # 2. Ingest to NeonDB after pkl is confirmed written
                run_id = ingest_result_to_db(result, metadata, gt_eng, db, run_key)
                if run_id is not None:
                    run_ids_ingested.append(run_id)

    print(f"\n{'='*80}")
    print("[OK] All runs complete.")
    print(f"  Total sweeps executed: {run_count}/{total}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Register all newly ingested runs under a clustering_sweep group
    # ------------------------------------------------------------------
    if run_ids_ingested:
        ts = datetime.now().strftime("%Y%m%d")
        sweep_name = f"{OUT_PREFIX}_sweep_{ts}"
        try:
            with db.session_scope() as session:
                from study_query_llm.db.models_v2 import Group, GroupLink
                from study_query_llm.db.raw_call_repository import RawCallRepository
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
                            **SWEEP_CONFIG.__dict__ if hasattr(SWEEP_CONFIG, "__dict__") else {},
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
        help="Re-run even if pkl backup already exists on disk",
    )
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
