#!/usr/bin/env python3
"""
Build 30k embedding cache for DBPedia and Yahoo Answers with seed 42.

Loads full datasets (same 10–1000 char filter as sweeps), samples 30k with
seed 42 per dataset, fetches embeddings for the given deployment(s), and
writes cache files to data/embedding_cache/ so that sweep scripts can use
them when running with 30k and seed 42.

Usage:
    # Default: one deployment from env or text-embedding-3-small, cache in data/embedding_cache
    python scripts/build_embedding_cache_30k.py

    # Custom deployment and cache dir
    python scripts/build_embedding_cache_30k.py --deployment text-embedding-3-small --cache-dir data/embedding_cache

    # Multiple deployments (comma-separated)
    python scripts/build_embedding_cache_30k.py --deployment "text-embedding-3-small,embed-v-4-0"

Requires DATABASE_URL so embeddings can be fetched (and optionally persisted to DB).
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_SEED = 42
CACHE_N_SAMPLES = 30_000
DEFAULT_DEPLOYMENT = "text-embedding-3-small"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embedding_cache"


def _sample_30k(texts, labels, seed: int = CACHE_SEED):
    """Sample up to 30k indices with seed; return (sampled_texts, sampled_labels)."""
    import numpy as np

    n = len(texts)
    size = min(CACHE_N_SAMPLES, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=size, replace=False)
    sampled_texts = [texts[i] for i in indices]
    sampled_labels = labels[indices]
    return sampled_texts, sampled_labels


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
) -> None:
    """Sample 30k, fetch embeddings, save cache for one (dataset, deployment)."""
    sampled_texts, sampled_labels = _sample_30k(texts, labels)
    print(f"  [{dataset_name}] Fetching embeddings for {len(sampled_texts)} texts ({deployment})...")
    embeddings = await fetch_embeddings_async_fn(sampled_texts, deployment, db)
    cache_path = get_cache_path_fn(cache_dir, dataset_name, deployment, CACHE_SEED, CACHE_N_SAMPLES)
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
    print(f"  [{dataset_name}] Saved {cache_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Build 30k embedding cache (seed 42) for DBPedia and Yahoo Answers."
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", DEFAULT_DEPLOYMENT),
        help="Embedding deployment name (default: env AZURE_OPENAI_EMBEDDING_DEPLOYMENT or text-embedding-3-small). Use comma for multiple.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory (default: data/embedding_cache)",
    )
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL"):
        print("[ERROR] DATABASE_URL environment variable is required to fetch embeddings.")
        sys.exit(1)

    from scripts.run_experimental_sweep import (
        DatabaseConnectionV2,
        DATABASE_URL,
        fetch_embeddings_async,
        load_dbpedia_full,
        load_yahoo_answers_full,
    )
    from study_query_llm.services import save_embedding_cache, get_cache_path

    deployments = [d.strip() for d in args.deployment.split(",") if d.strip()]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    print(f"Deployments: {deployments}")
    print(f"Seed: {CACHE_SEED}, n_samples: {CACHE_N_SAMPLES}")

    print("\nLoading datasets (10 < length <= 1000 chars)...")
    texts_dbpedia, labels_dbpedia, _ = load_dbpedia_full(random_state=CACHE_SEED)
    texts_yahoo, labels_yahoo, _ = load_yahoo_answers_full(random_state=CACHE_SEED)
    print(f"  DBPedia: {len(texts_dbpedia)} texts")
    print(f"  Yahoo Answers: {len(texts_yahoo)} texts")

    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()

    for deployment in deployments:
        print(f"\nDeployment: {deployment}")
        await _build_one(
            "dbpedia",
            texts_dbpedia,
            labels_dbpedia,
            deployment,
            cache_dir,
            db,
            fetch_embeddings_async,
            save_embedding_cache,
            get_cache_path,
        )
        await _build_one(
            "yahoo_answers",
            texts_yahoo,
            labels_yahoo,
            deployment,
            cache_dir,
            db,
            fetch_embeddings_async,
            save_embedding_cache,
            get_cache_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
