#!/usr/bin/env python3
"""
Build 30k embedding cache for DBPedia, Yahoo Answers, and Estela across all
available Azure embedding engines.

For each (dataset, engine) pair the script:
  1. Checks if the .npz file already exists — skips if so.
  2. Samples up to 30k texts with seed 42 (deterministic, reproducible).
  3. Fetches embeddings by calling the Azure API directly in sequential batches
     of batch_size texts per request.  Failed batches are retried with
     exponential back-off; permanently failed batches are filled with zeros and
     logged so the script never stops.
  4. Saves the .npz to data/embedding_cache/.

All errors are caught per (dataset, engine) — the script always continues to
the next pair so it can run overnight without supervision.

Usage:
    python scripts/build_embedding_cache_30k.py

    # Specific engines or cache dir:
    python scripts/build_embedding_cache_30k.py \\
        --deployment "text-embedding-3-small,embed-v-4-0" \\
        --cache-dir data/embedding_cache

    # Tuning:
    python scripts/build_embedding_cache_30k.py --batch-size 100 --max-retries 5

Requires: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (auto-loaded from .env).
"""

import os
import sys
import time
import asyncio
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure output is flushed immediately when running in the background (non-TTY)
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Load .env so AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, etc. are available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path if _env_path.exists() else None)
except ImportError:
    pass

# All 6 Azure embedding engines
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
DEFAULT_BATCH_SIZE = 100    # texts per embeddings.create call (safe for all models)
DEFAULT_MAX_RETRIES = 6     # exponential back-off: ~1+2+4+8+16+32 = ~63s total


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_estela(repo_root: Path):
    """Return (texts, labels) for the estela dataset."""
    import pickle
    import numpy as np

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
    raw_texts = [v for v in flat.values() if isinstance(v, str)]
    texts = [t.replace("\x00", "").strip() for t in raw_texts]
    texts = [t for t in texts if 10 < len(t) <= 1000]
    labels = np.zeros(len(texts), dtype=np.int64)
    return texts, labels


def _sample_dataset(texts, labels, seed=CACHE_SEED, n_samples=CACHE_N_SAMPLES):
    """Return deterministically sampled (texts, labels)."""
    import numpy as np
    n = len(texts)
    size = min(n_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=size, replace=False)
    return [texts[i] for i in indices], labels[indices]


def _cache_path(cache_dir: Path, dataset: str, engine: str, seed: int, n: int) -> Path:
    """Construct the .npz cache file path."""
    safe_engine = engine.replace("/", "_").replace("\\", "_").replace(":", "_")
    return cache_dir / f"{dataset}_{safe_engine}_seed{seed}_n{n}.npz"


# ---------------------------------------------------------------------------
# Azure embedding fetcher (no DB, no service layer overhead)
# ---------------------------------------------------------------------------

async def _fetch_batch_with_retry(
    client,
    model: str,
    texts: list,
    max_retries: int,
    batch_idx: int,
    total_batches: int,
) -> list:
    """
    Fetch embeddings for one batch via Azure API with exponential back-off retry.

    Returns a list of embedding vectors (list[float]), one per input text.
    On permanent failure, returns a list of None values so the caller can
    substitute zeros and continue.
    """
    wait = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.embeddings.create(model=model, input=texts)
            sorted_data = sorted(response.data, key=lambda e: e.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            is_retryable = is_rate_limit or any(
                kw in err_str.lower()
                for kw in ("timeout", "connection", "502", "503", "504", "internal server")
            )
            if attempt == max_retries or not is_retryable:
                print(
                    f"      [FAIL] batch {batch_idx+1}/{total_batches} "
                    f"after {attempt} attempt(s): {e}"
                )
                return [None] * len(texts)
            sleep_for = wait * (2 ** (attempt - 1))
            if is_rate_limit:
                # Back off more aggressively on 429
                sleep_for = max(sleep_for, 60.0)
            print(
                f"      [RETRY {attempt}/{max_retries}] batch {batch_idx+1}/{total_batches} "
                f"in {sleep_for:.0f}s: {e}"
            )
            await asyncio.sleep(sleep_for)
    return [None] * len(texts)


async def _build_one(
    dataset_name: str,
    texts: list,
    labels,
    engine: str,
    cache_dir: Path,
    client,
    batch_size: int,
    max_retries: int,
) -> bool:
    """Fetch and save embeddings for one (dataset, engine) pair. Returns True on success."""
    import numpy as np

    path = _cache_path(cache_dir, dataset_name, engine, CACHE_SEED, CACHE_N_SAMPLES)
    if path.exists():
        print(f"    [SKIP] Already cached: {path.name}")
        return True

    sampled_texts, sampled_labels = _sample_dataset(texts, labels)
    n = len(sampled_texts)
    batches = [sampled_texts[i:i + batch_size] for i in range(0, n, batch_size)]
    total_batches = len(batches)
    print(f"    {n} texts → {total_batches} batches of {batch_size}")

    t0 = time.time()
    all_vectors = []
    failed_count = 0

    for idx, batch in enumerate(batches):
        if idx % 20 == 0 or idx == total_batches - 1:
            elapsed = time.time() - t0
            print(
                f"    Batch {idx+1}/{total_batches}  "
                f"({elapsed:.0f}s elapsed, {failed_count} failed)"
            )
        vectors = await _fetch_batch_with_retry(client, engine, batch, max_retries, idx, total_batches)
        for v in vectors:
            all_vectors.append(v)

    elapsed = time.time() - t0

    # Convert to numpy; use zeros for failed (None) vectors
    dim = next((len(v) for v in all_vectors if v is not None), 1)
    embeddings = np.array(
        [v if v is not None else [0.0] * dim for v in all_vectors],
        dtype=np.float32,
    )
    failed_count = sum(1 for v in all_vectors if v is None)

    print(
        f"    Done in {elapsed:.0f}s — shape: {embeddings.shape}, "
        f"failed batches contributed zeros: {failed_count}"
    )

    # Save .npz
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            texts=np.array(sampled_texts, dtype=object),
            embeddings=embeddings,
            labels=sampled_labels,
            dataset=np.array(dataset_name),
            deployment=np.array(engine),
            seed=np.array(CACHE_SEED),
            n_samples=np.array(CACHE_N_SAMPLES),
        )
        print(f"    Saved: {path.name}")
        return True
    except Exception as e:
        print(f"    [ERROR] Save failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Build 30k embedding cache for DBPedia, Yahoo Answers, and Estela "
                    "across all Azure embedding engines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default="",
        help="Comma-separated engine names. Default: all 6 engines.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Texts per embeddings.create call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries per batch before filling with zeros and moving on.",
    )
    args = parser.parse_args()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

    missing = [name for name, val in [
        ("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", api_key)
    ] if not val]
    if missing:
        print(f"[ERROR] Missing required env vars: {', '.join(missing)}")
        sys.exit(1)

    from openai import AsyncAzureOpenAI

    engines = (
        [e.strip() for e in args.deployment.split(",") if e.strip()]
        if args.deployment
        else list(ALL_ENGINES)
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    max_retries = args.max_retries

    print("=" * 70)
    print("Embedding cache builder (direct Azure API, no DB overhead)")
    print("=" * 70)
    print(f"Cache dir   : {cache_dir}")
    print(f"Engines     : {engines}")
    print(f"Datasets    : dbpedia, yahoo_answers, estela")
    print(f"Seed        : {CACHE_SEED}  n_samples: {CACHE_N_SAMPLES}")
    print(f"Batch size  : {batch_size}  Max retries/batch: {max_retries}")
    print(f"API version : {api_version}")
    print()

    repo_root = Path(__file__).resolve().parent.parent
    print("Loading datasets...")

    datasets = {}

    try:
        from scripts.run_experimental_sweep import load_dbpedia_full
        texts_db, labels_db, _ = load_dbpedia_full(random_state=CACHE_SEED)
        datasets["dbpedia"] = (texts_db, labels_db)
        print(f"  dbpedia        : {len(texts_db)} texts")
    except Exception as e:
        print(f"  [WARN] Could not load dbpedia: {e}")

    try:
        from scripts.run_experimental_sweep import load_yahoo_answers_full
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
        print("[ERROR] No datasets could be loaded.")
        sys.exit(1)

    total_pairs = len(datasets) * len(engines)
    done = 0
    succeeded = 0
    failed_pairs = []

    for engine in engines:
        print(f"\n{'=' * 70}")
        print(f"Engine: {engine}")
        print("=" * 70)

        # Create a fresh async client for each engine
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        try:
            for dataset_name, (texts, labels) in datasets.items():
                done += 1
                print(f"\n  [{done}/{total_pairs}] {dataset_name} × {engine}")
                try:
                    ok = await _build_one(
                        dataset_name=dataset_name,
                        texts=texts,
                        labels=labels,
                        engine=engine,
                        cache_dir=cache_dir,
                        client=client,
                        batch_size=batch_size,
                        max_retries=max_retries,
                    )
                    if ok:
                        succeeded += 1
                    else:
                        failed_pairs.append((dataset_name, engine))
                except Exception as e:
                    print(f"    [ERROR] Unexpected: {e}")
                    traceback.print_exc()
                    failed_pairs.append((dataset_name, engine))
        finally:
            await client.close()

    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    print(f"  Total pairs  : {total_pairs}")
    print(f"  Succeeded    : {succeeded}")
    print(f"  Errors       : {len(failed_pairs)}")
    if failed_pairs:
        for ds, eng in failed_pairs:
            print(f"    ✗ {ds} × {eng}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
