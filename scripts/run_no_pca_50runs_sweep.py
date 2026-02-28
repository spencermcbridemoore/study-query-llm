"""
No-PCA 50-runs sweep: 1000 samples, k=2-20, 50 restarts per (dataset, summarizer[, engine]).

Default: embed-v-4-0, dbpedia + yahoo_answers, None + gpt-5-chat (4 runs).
--dbpedia-large-small: dbpedia only, text-embedding-3-large + text-embedding-3-small, None + gpt-5-chat (4 runs).

Output: no_pca_50runs_entry1000_<dataset>_<engine_safe>_<summarizer>_<timestamp>.pkl
(engine_safe omitted for default single-engine runs for backward compatibility with existing filenames)
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

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.services.embedding_service import estimate_tokens, DEPLOYMENT_MAX_TOKENS
from study_query_llm.algorithms import SweepConfig

from study_query_llm.services.embedding_helpers import fetch_embeddings_async
from scripts.common.sweep_utils import (
    create_paraphraser_for_llm,
    save_single_sweep_result as save_results,
    OUTPUT_DIR,
)
from scripts.run_experimental_sweep import (
    load_dbpedia_full,
    load_yahoo_answers_full,
)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
import numpy as np

ENTRY_MAX = 1000
EMBEDDING_ENGINE_DEFAULT = "embed-v-4-0"
ENGINES_LARGE_SMALL = ["text-embedding-3-large", "text-embedding-3-small"]
SUMMARIZERS = [None, "gpt-5-chat"]
N_RESTARTS = 50
K_MIN, K_MAX = 2, 20

DATASETS_FULL = [
    {"name": "dbpedia", "type": "dbpedia", "categories": None, "label_max": 14, "loader": load_dbpedia_full},
    {"name": "yahoo_answers", "type": "yahoo_answers", "categories": None, "label_max": 10, "loader": load_yahoo_answers_full},
]
DATASETS_DBPEDIA_ONLY = [
    {"name": "dbpedia", "type": "dbpedia", "categories": None, "label_max": 14, "loader": load_dbpedia_full},
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

OUT_PREFIX = "no_pca_50runs"


def _engine_safe(engine: str) -> str:
    return engine.replace("-", "_")


async def run_single_sweep_no_pca(texts, embeddings, llm_deployment, db, sweep_config, embedding_engine: str):
    from study_query_llm.algorithms.sweep import run_sweep

    paraphraser = create_paraphraser_for_llm(llm_deployment, db)

    async def embedder_func(texts_list):
        return await fetch_embeddings_async(texts_list, embedding_engine, db)

    def embedder_sync(texts_list):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(embedder_func(texts_list))
        except RuntimeError:
            pass
        return asyncio.run(embedder_func(texts_list))

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: run_sweep(
                texts, embeddings, sweep_config,
                paraphraser=paraphraser,
                embedder=embedder_sync if paraphraser else None,
            ),
        )
    return result


async def main(force: bool = False, dbpedia_large_small: bool = False):
    engines = ENGINES_LARGE_SMALL if dbpedia_large_small else [EMBEDDING_ENGINE_DEFAULT]
    datasets = DATASETS_DBPEDIA_ONLY if dbpedia_large_small else DATASETS_FULL
    total = len(datasets) * len(engines) * len(SUMMARIZERS)

    print("=" * 80)
    if dbpedia_large_small:
        print("No-PCA 50-runs sweep (dbpedia, large/small engines, None + gpt-5-chat)")
    else:
        print("No-PCA 50-runs sweep (entry=1000, k=2-20, embed-v-4-0, None + gpt-5-chat)")
    print("=" * 80)
    print(f"  entry_max: {ENTRY_MAX}")
    print(f"  k_range: {K_MIN}-{K_MAX}")
    print(f"  n_restarts: {N_RESTARTS}")
    print(f"  summarizers: {SUMMARIZERS}")
    print(f"  embedding engines: {engines}")
    print(f"  datasets: {[d['name'] for d in datasets]}")

    if force and not dbpedia_large_small:
        removed = 0
        for p in OUTPUT_DIR.glob(f"{OUT_PREFIX}_*.pkl"):
            p.unlink()
            removed += 1
        if removed:
            print(f"\n[INFO] --force: removed {removed} existing pickle(s).")
    elif force and dbpedia_large_small:
        for eng in engines:
            es = _engine_safe(eng)
            for p in OUTPUT_DIR.glob(f"{OUT_PREFIX}_entry{ENTRY_MAX}_dbpedia_{es}_*.pkl"):
                p.unlink()
                print(f"[INFO] --force: removed {p.name}")

    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()

    loaded = {}
    for bench in datasets:
        name = bench["name"]
        print(f"\nLoading {name}...")
        texts, labels, category_names = bench["loader"]()
        unique_labels = sorted(set(labels))
        label_max = min(bench["label_max"], len(unique_labels))
        mask = np.isin(labels, unique_labels[:label_max])
        idx = np.where(mask)[0]
        if len(idx) > ENTRY_MAX:
            np.random.seed(42)
            idx = np.random.choice(idx, size=ENTRY_MAX, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]
        # Filter by length
        valid = [i for i, t in enumerate(texts) if 10 < len(t) <= 1000]
        if len(valid) < len(texts):
            texts = [texts[i] for i in valid]
            labels = labels[valid]
        loaded[name] = {"texts": texts, "labels": labels, "label_max": label_max, "category_names": category_names}
        print(f"  {len(texts)} texts, {len(set(labels))} labels")

    run_count = 0
    for dataset_name, info in loaded.items():
        texts = info["texts"]
        labels = info["labels"]
        label_max = info["label_max"]
        for embedding_engine in engines:
            max_tokens = DEPLOYMENT_MAX_TOKENS.get(embedding_engine)
            if max_tokens:
                valid = []
                valid_idx = []
                for i, t in enumerate(texts):
                    try:
                        if estimate_tokens(t, embedding_engine) <= max_tokens:
                            valid.append(t)
                            valid_idx.append(i)
                    except Exception:
                        valid.append(t)
                        valid_idx.append(i)
                if len(valid) < len(texts):
                    texts_eng = [texts[i] for i in valid]
                    labels_eng = labels[valid_idx]
                else:
                    texts_eng = texts
                    labels_eng = labels
            else:
                texts_eng = texts
                labels_eng = labels
            print(f"\n  Fetching embeddings for {dataset_name} / {embedding_engine} ({len(texts_eng)} texts)...")
            try:
                embeddings = await fetch_embeddings_async(texts_eng, embedding_engine, db)
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue
            engine_safe = _engine_safe(embedding_engine)
            for llm in SUMMARIZERS:
                run_count += 1
                summarizer_name = "None" if llm is None else llm
                sum_safe = summarizer_name.replace("-", "_")
                if dbpedia_large_small:
                    out_name = f"{OUT_PREFIX}_entry{ENTRY_MAX}_{dataset_name}_{engine_safe}_{sum_safe}_"
                else:
                    out_name = f"{OUT_PREFIX}_entry{ENTRY_MAX}_{dataset_name}_{sum_safe}_"
                if not force and list(OUTPUT_DIR.glob(out_name + "*.pkl")):
                    print(f"  [{run_count}/{total}] {embedding_engine} / {summarizer_name} (SKIP - exists)")
                    continue
                print(f"  [{run_count}/{total}] {dataset_name} / {embedding_engine} / {summarizer_name} ...")
                try:
                    result = await asyncio.wait_for(
                        run_single_sweep_no_pca(texts_eng, embeddings, llm, db, SWEEP_CONFIG, embedding_engine),
                        timeout=7200.0,
                    )
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = OUTPUT_DIR / f"{out_name}{ts}.pkl"
                metadata = {
                    "entry_max": ENTRY_MAX,
                    "label_max": label_max,
                    "actual_entry_count": len(texts_eng),
                    "actual_label_count": len(set(labels_eng)),
                    "benchmark_source": dataset_name,
                    "summarizer": summarizer_name,
                    "embedding_engine": embedding_engine,
                    "n_restarts": N_RESTARTS,
                    "sweep_config": {"skip_pca": True, "k_min": K_MIN, "k_max": K_MAX, "n_restarts": N_RESTARTS, "compute_stability": True},
                    "note": "No-PCA 50-runs sweep for mean/stdev and box stats",
                }
                save_results(result, str(out_path), ground_truth_labels=labels_eng, dataset_name=dataset_name, metadata=metadata)
                print(f"  Saved {out_path.name}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Overwrite existing 50runs pickles")
    p.add_argument("--dbpedia-large-small", action="store_true", help="Dbpedia only, text-embedding-3-large + text-embedding-3-small, None + gpt-5-chat (4 runs)")
    args = p.parse_args()
    asyncio.run(main(force=args.force, dbpedia_large_small=args.dbpedia_large_small))
