#!/usr/bin/env python3
"""
Build 30k embedding cache for DBPedia, Yahoo Answers, and Estela with all
available embedding engines.

For each (dataset, engine) pair the script:
  1. Checks whether the .npz cache file already exists — skips if so.
  2. Samples up to 30k texts with seed 42 (deterministic).
  3. Fetches embeddings in chunks of chunk_size, using the DB as a resumable
     cache (re-running after a crash picks up where it left off).
  4. Saves the .npz to data/embedding_cache/.

Errors for any single (dataset, engine) pair are caught and logged; the script
always moves on to the next pair so it can run overnight without supervision.

Usage:
    python scripts/build_embedding_cache_30k.py

    # Override engines (comma-separated) or cache dir:
    python scripts/build_embedding_cache_30k.py \\
        --deployment "text-embedding-3-small,embed-v-4-0" \\
        --cache-dir data/embedding_cache

    # Tune batching:
    python scripts/build_embedding_cache_30k.py --chunk-size 500 --timeout 14400

Requires DATABASE_URL to be set (loaded automatically from .env if present).
"""

import os
import sys
import asyncio
import argparse
import time
from pathlib import Path

# Ensure output is flushed immediately when running in background (non-TTY)
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so DATABASE_URL and other env vars are available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path if _env_path.exists() else None)
except ImportError:
    pass

# All 6 embedding engines expected to be available on Azure
ALL_ENGINES = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "Cohere-embed-v3-english",
    "Cohere-embed-v3-multilingual",
    "embed-v-4-0",
]

CACHE_SEED = 42
CACHE_N_SAMPLES = 30_000
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embedding_cache"
DEFAULT_CHUNK_SIZE = 500        # texts per embeddings.create call (conservative)
DEFAULT_TIMEOUT = 14400.0       # 4-hour wall-clock limit per (dataset, engine)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_estela(repo_root: Path):
    """Load estela texts from notebooks/estela_prompt_data.pkl or data/estela/estela_db.py."""
    import pickle

    pkl_path = repo_root / "notebooks" / "estela_prompt_data.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    else:
        import runpy, datetime as dt
        estela_py = repo_root / "data" / "estela" / "estela_db.py"
        if not estela_py.exists():
            raise FileNotFoundError(
                f"Estela data not found. Expected {pkl_path} or {estela_py}"
            )
        g = runpy.run_path(str(estela_py), init_globals={"datetime": dt})
        data = g["database_estela_dict"]

    # Flatten prompt keys the same way other scripts do
    def _is_prompt_key(k):
        return isinstance(k, str) and "prompt" in k.lower()

    def _flatten(d, path=()):
        flat = {}
        if isinstance(d, dict):
            for k, v in d.items():
                new_path = path + (k,)
                if _is_prompt_key(k) and isinstance(v, str):
                    flat[new_path] = v
                else:
                    flat.update(_flatten(v, new_path))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                flat.update(_flatten(v, path + (i,)))
        return flat

    flat = _flatten(data)
    import numpy as np
    raw_texts = [v for v in flat.values() if isinstance(v, str)]
    texts = [t.replace("\x00", "").strip() for t in raw_texts]
    texts = [t for t in texts if 10 < len(t) <= 1000]
    labels = np.zeros(len(texts), dtype=np.int64)
    return texts, labels


def _sample_dataset(texts, labels, seed: int = CACHE_SEED, n_samples: int = CACHE_N_SAMPLES):
    """Return (sampled_texts, sampled_labels) using deterministic seed."""
    import numpy as np
    n = len(texts)
    size = min(n_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=size, replace=False)
    return [texts[i] for i in indices], labels[indices]


# ---------------------------------------------------------------------------
# Per-(dataset, engine) build
# ---------------------------------------------------------------------------

async def _build_one(
    dataset_name: str,
    texts: list,
    labels,
    deployment: str,
    cache_dir: Path,
    db,
    fetch_embeddings_async_fn,
    save_embedding_cache_fn,
    get_cache_path_fn,
    chunk_size: int,
    timeout: float,
) -> bool:
    """Fetch + save cache for one (dataset, engine) pair. Returns True on success."""
    cache_path = get_cache_path_fn(cache_dir, dataset_name, deployment, CACHE_SEED, CACHE_N_SAMPLES)
    if cache_path.exists():
        print(f"    [SKIP] Cache exists: {cache_path.name}")
        return True

    sampled_texts, sampled_labels = _sample_dataset(texts, labels)
    n = len(sampled_texts)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(
        f"    Fetching {n} texts in {n_chunks} chunks of {chunk_size}..."
    )
    t0 = time.time()
    try:
        embeddings = await fetch_embeddings_async_fn(
            sampled_texts, deployment, db, timeout=timeout, chunk_size=chunk_size
        )
    except Exception as e:
        print(f"    [ERROR] Fetch failed for {dataset_name}/{deployment}: {e}")
        return False

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s, shape: {embeddings.shape}")

    try:
        save_embedding_cache_fn(
            cache_path,
            sampled_texts,
            embeddings,
            sampled_labels,
            dataset=dataset_name,
            deployment=deployment,
            seed=CACHE_SEED,
            n_samples=CACHE_N_SAMPLES,
        )
        print(f"    Saved: {cache_path.name}")
    except Exception as e:
        print(f"    [ERROR] Save failed for {dataset_name}/{deployment}: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Build 30k embedding cache for DBPedia, Yahoo Answers, and Estela "
                    "across all embedding engines."
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default="",
        help="Comma-separated engine names (default: all 6 engines).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Output directory (default: data/embedding_cache).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Texts per embeddings.create call (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-(dataset,engine) timeout in seconds (default: {DEFAULT_TIMEOUT:.0f}). "
             "Restart resumes via DB cache.",
    )
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        print("[ERROR] DATABASE_URL is required. Set it in .env or the environment.")
        sys.exit(1)

    from scripts.run_experimental_sweep import (
        DatabaseConnectionV2,
        DATABASE_URL,
        fetch_embeddings_async,
        load_dbpedia_full,
        load_yahoo_answers_full,
    )
    from study_query_llm.services import save_embedding_cache, get_cache_path

    engines = (
        [e.strip() for e in args.deployment.split(",") if e.strip()]
        if args.deployment
        else list(ALL_ENGINES)
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_size
    timeout = args.timeout

    print("=" * 70)
    print("Embedding cache builder")
    print("=" * 70)
    print(f"Cache dir  : {cache_dir}")
    print(f"Engines    : {engines}")
    print(f"Datasets   : dbpedia, yahoo_answers, estela")
    print(f"Seed       : {CACHE_SEED}  n_samples: {CACHE_N_SAMPLES}")
    print(f"Chunk size : {chunk_size}  Timeout/pair: {timeout:.0f}s")
    print()

    # --- Load datasets ---
    repo_root = Path(__file__).resolve().parent.parent
    print("Loading datasets...")

    datasets = {}

    try:
        texts_db, labels_db, _ = load_dbpedia_full(random_state=CACHE_SEED)
        datasets["dbpedia"] = (texts_db, labels_db)
        print(f"  dbpedia        : {len(texts_db)} texts")
    except Exception as e:
        print(f"  [WARN] Could not load dbpedia: {e}")

    try:
        texts_ya, labels_ya, _ = load_yahoo_answers_full(random_state=CACHE_SEED)
        datasets["yahoo_answers"] = (texts_ya, labels_ya)
        print(f"  yahoo_answers  : {len(texts_ya)} texts")
    except Exception as e:
        print(f"  [WARN] Could not load yahoo_answers: {e}")

    try:
        texts_es, labels_es = _load_estela(repo_root)
        datasets["estela"] = (texts_es, labels_es)
        print(f"  estela         : {len(texts_es)} texts")
    except Exception as e:
        print(f"  [WARN] Could not load estela: {e}")

    if not datasets:
        print("[ERROR] No datasets could be loaded. Exiting.")
        sys.exit(1)

    # --- Init DB ---
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()

    # --- Build all (dataset, engine) pairs ---
    total = len(datasets) * len(engines)
    done = 0
    succeeded = 0
    failed_pairs = []

    for engine in engines:
        print(f"\n{'=' * 70}")
        print(f"Engine: {engine}")
        print("=" * 70)

        for dataset_name, (texts, labels) in datasets.items():
            done += 1
            print(f"\n  [{done}/{total}] {dataset_name} x {engine}")
            try:
                ok = await _build_one(
                    dataset_name=dataset_name,
                    texts=texts,
                    labels=labels,
                    deployment=engine,
                    cache_dir=cache_dir,
                    db=db,
                    fetch_embeddings_async_fn=fetch_embeddings_async,
                    save_embedding_cache_fn=save_embedding_cache,
                    get_cache_path_fn=get_cache_path,
                    chunk_size=chunk_size,
                    timeout=timeout,
                )
                if ok:
                    succeeded += 1
                else:
                    failed_pairs.append((dataset_name, engine))
            except Exception as e:
                # Belt-and-suspenders: catch anything that escaped _build_one
                print(f"    [ERROR] Unexpected error for {dataset_name}/{engine}: {e}")
                failed_pairs.append((dataset_name, engine))

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    print(f"  Total pairs  : {total}")
    print(f"  Succeeded    : {succeeded}")
    print(f"  Failed/skipped errors: {len(failed_pairs)}")
    if failed_pairs:
        print("  Failed pairs:")
        for ds, eng in failed_pairs:
            print(f"    {ds} x {eng}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
