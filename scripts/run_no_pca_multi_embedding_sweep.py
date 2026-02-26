"""
No-PCA Multi-Embedding Sweep: Test Multiple Embedding Engines Without Dimensionality Reduction

Configuration:
- Datasets: dbpedia (14 categories), yahoo_answers (10 categories)
- Embedding Engines: 6 (embed-v-4-0, Cohere v3 variants, OpenAI variants)
- label_max: FIXED at maximum (14 for dbpedia, 10 for yahoo_answers)
- entry_max: 500 (single run, no [100,200,300,400])
- k_range: 1 to 20
- summarizers: All (None, gpt-4o-mini, gpt-4o, gpt-5-chat)
- skip_pca: True (use full embedding dimensions)
- n_restarts: 1 (no stability analysis, single run per configuration)

This sweep tests:
1. Full-dimensional embeddings (no PCA reduction)
2. Multiple embedding engines (different embedding spaces)
3. Extended k range (1-20)
4. All available categories (not subsampled)

Total runs: 2 datasets × 6 embedding engines × 4 summarizers = 48 runs
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add parent directory and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Core library imports
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.services.embedding_service import estimate_tokens, DEPLOYMENT_MAX_TOKENS
from study_query_llm.algorithms import SweepConfig

# Shared script utilities
from scripts.common.embedding_utils import fetch_embeddings_async
from scripts.common.sweep_utils import (
    create_paraphraser_for_llm,
    save_single_sweep_result as save_results,
    OUTPUT_DIR,
)
from scripts.run_experimental_sweep import (
    load_dbpedia_full,
    load_yahoo_answers_full,
    LLM_SUMMARIZERS,
)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
from study_query_llm.services import get_embeddings_with_file_cache
import numpy as np
import time
from datetime import datetime
import glob
from concurrent.futures import ThreadPoolExecutor

# Embedding engines to test.
# Each entry is a (model, provider) tuple.
# provider can be "azure", "openai", "huggingface", "local", "ollama", etc.
# Example local entries: ("BAAI/bge-m3", "ollama"), ("nomic-embed-text", "ollama")
EMBEDDING_ENGINES = [
    ("embed-v-4-0", "azure"),             # Hyperbolic via Azure
    ("Cohere-embed-v3-multilingual", "azure"),
    ("Cohere-embed-v3-english", "azure"),
    ("text-embedding-ada-002", "azure"),  # OpenAI legacy via Azure
    ("text-embedding-3-large", "azure"),  # OpenAI via Azure
    ("text-embedding-3-small", "azure"),  # OpenAI via Azure
]

# Single entry_max value
ENTRY_MAX = 500

# Optional file cache for 30k seed-42 runs (no errors if missing)
EMBEDDING_CACHE_DIR = os.environ.get("EMBEDDING_CACHE_DIR") or str(Path(__file__).parent.parent / "data" / "embedding_cache")

# Datasets configuration
DATASETS = [
    {
        "name": "dbpedia",
        "type": "dbpedia",
        "categories": None,  # Use all 14 categories
        "label_max": 14,     # FIXED: use all categories
        "loader": load_dbpedia_full,
    },
    {
        "name": "yahoo_answers",
        "type": "yahoo_answers",
        "categories": None,  # Use all 10 categories
        "label_max": 10,     # FIXED: use all categories
        "loader": load_yahoo_answers_full,
    },
]

# No-PCA sweep configuration
NO_PCA_SWEEP_CONFIG = SweepConfig(
    skip_pca=True,           # NEW: Use full-dimensional embeddings (no PCA)
    k_min=1,                 # Start at 1 instead of 2
    k_max=20,                # Go up to 20 instead of 10
    max_iter=200,
    base_seed=0,
    n_restarts=1,            # Single run (no stability analysis)
    compute_stability=False,  # Not needed with single run
    llm_interval=20,
    max_samples=10,
    distance_metric="cosine",
    normalize_vectors=True,
)


async def run_single_sweep_no_pca(
    texts, embeddings, llm_deployment, db, embedding_engine, sweep_config,
    llm_provider: str = "azure", embedding_provider: str = "azure",
):
    """
    Run sweep with no-PCA config and custom embedding engine.
    
    Args:
        texts: List of text strings
        embeddings: Pre-computed embeddings (full-dimensional)
        llm_deployment: LLM model name to use for summarization (or None)
        db: Database connection
        embedding_engine: Name of embedding engine / model used
        sweep_config: SweepConfig with skip_pca=True
        llm_provider: Provider for the paraphraser (default "azure")
        embedding_provider: Provider for the embedder used in centroid updates (default "azure")
    """
    from study_query_llm.algorithms.sweep import run_sweep
    
    paraphraser = create_paraphraser_for_llm(llm_deployment, db, provider=llm_provider)
    
    async def embedder_func(texts_list):
        """Embed texts using the specified embedding engine."""
        return await fetch_embeddings_async(texts_list, embedding_engine, db, provider_name=embedding_provider)
    
    def embedder_sync(texts_list):
        """Synchronous wrapper for embedder."""
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
                embedder=embedder_sync if paraphraser else None
            )
        )
    
    return result


async def main(force: bool = False):
    """Main execution function."""
    print("=" * 80)
    print("No-PCA Multi-Embedding Sweep")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Datasets: {[s['name'] for s in DATASETS]}")
    print(f"  label_max: FIXED at max categories per dataset")
    print(f"    - dbpedia: 14 categories")
    print(f"    - yahoo_answers: 10 categories")
    print(f"  entry_max: {ENTRY_MAX} (single run)")
    print(f"  embedding_engines: {len(EMBEDDING_ENGINES)}")
    for engine in EMBEDDING_ENGINES:
        print(f"    - {engine}")
    print(f"  k_range: {NO_PCA_SWEEP_CONFIG.k_min} to {NO_PCA_SWEEP_CONFIG.k_max}")
    print(f"  summarizers: {LLM_SUMMARIZERS}")
    print(f"  n_restarts: {NO_PCA_SWEEP_CONFIG.n_restarts} (single run, no stability)")
    print(f"  skip_pca: {NO_PCA_SWEEP_CONFIG.skip_pca} (full-dimensional embeddings)")
    
    total_runs = len(DATASETS) * len(EMBEDDING_ENGINES) * len(LLM_SUMMARIZERS)
    print(f"\n[INFO] Total experimental runs: {total_runs}")
    print(f"   datasets: {len(DATASETS)}")
    print(f"   embedding engines: {len(EMBEDDING_ENGINES)}")
    print(f"   summarizers: {len(LLM_SUMMARIZERS)}")
    print(f"\n[WARN] No-PCA mode: Expect ~20-24x slower runtime than PCA mode")
    
    # Initialize database
    print("\n[INFO] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("[OK] Database initialized")
    
    # Load datasets
    print("\n[INFO] Loading full benchmark datasets...")
    print("   Filtering: 10 < length <= 1000 characters")
    
    loaded_datasets = {}
    for benchmark in DATASETS:
        name = benchmark['name']
        print(f"   Loading {name}...")
        
        try:
            texts, labels, category_names = benchmark['loader']()
            loaded_datasets[name] = {
                'texts': texts,
                'labels': labels,
                'category_names': category_names,
                'label_max': benchmark['label_max'],
            }
            print(f"      [OK] Loaded {len(texts)} texts, {len(set(labels))} unique labels")
        except Exception as e:
            print(f"      [ERROR] Failed to load {name}: {e}")
            continue
    
    if not loaded_datasets:
        print("\n[ERROR] No datasets loaded successfully. Exiting.")
        return

    # If --force, remove existing no-PCA pickles so we re-run and save Z
    if force:
        removed = 0
        for pattern in (
            f"experimental_sweep_entry{ENTRY_MAX}_dbpedia_*.pkl",
            f"experimental_sweep_entry{ENTRY_MAX}_yahoo_answers_*.pkl",
        ):
            for p in OUTPUT_DIR.glob(pattern):
                p.unlink()
                removed += 1
        if removed:
            print(f"\n[INFO] --force: removed {removed} existing pickle(s). Re-running all runs.")
    
    # Run sweep
    print("\n" + "=" * 80)
    start_time = time.time()
    run_count = 0
    
    for dataset_name, dataset_info in loaded_datasets.items():
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print("="*80)
        
        texts = dataset_info['texts']
        labels = dataset_info['labels']
        label_max = dataset_info['label_max']
        
        # Sample entries (with label_max fixed at maximum)
        unique_labels = sorted(set(labels))
        actual_label_count = len(unique_labels)
        
        if actual_label_count < label_max:
            print(f"    [WARN] Dataset has only {actual_label_count} labels, less than requested {label_max}")
            label_max = actual_label_count
        
        # Select labels
        selected_labels = unique_labels[:label_max]
        
        # Get indices for selected labels
        mask = np.isin(labels, selected_labels)
        candidate_indices = np.where(mask)[0]
        
        # Sample up to ENTRY_MAX
        if len(candidate_indices) > ENTRY_MAX:
            np.random.seed(42)  # Reproducible sampling
            sampled_indices = np.random.choice(candidate_indices, size=ENTRY_MAX, replace=False)
        else:
            sampled_indices = candidate_indices
        
        sampled_texts = [texts[i] for i in sampled_indices]
        sampled_labels = labels[sampled_indices]
        
        actual_entry_count = len(sampled_texts)
        actual_label_count = len(set(sampled_labels))
        
        print(f"  Sampled: {actual_entry_count} entries, {actual_label_count} unique labels (all categories)")
        
        # Loop over embedding engines
        for engine_entry in EMBEDDING_ENGINES:
            # Unpack (model, provider) tuple
            embedding_engine, embedding_provider = engine_entry
            print(f"\n  EMBEDDING ENGINE: {embedding_engine} (provider: {embedding_provider})")
            
            # Filter texts that exceed token limit for this embedding engine
            max_tokens = DEPLOYMENT_MAX_TOKENS.get(embedding_engine)
            filtered_texts = sampled_texts
            filtered_labels = sampled_labels
            
            if max_tokens:
                valid_texts = []
                valid_indices = []
                for i, text in enumerate(sampled_texts):
                    try:
                        estimated = estimate_tokens(text, embedding_engine)
                        if estimated <= max_tokens:
                            valid_texts.append(text)
                            valid_indices.append(i)
                    except Exception:
                        # If estimation fails, include text
                        valid_texts.append(text)
                        valid_indices.append(i)
                
                if len(valid_texts) < len(sampled_texts):
                    print(f"    Filtered: {len(valid_texts)} valid, {len(sampled_texts) - len(valid_texts)} filtered out")
                    filtered_texts = valid_texts
                    filtered_labels = sampled_labels[valid_indices]
            
            # ===================================================================
            # Embed texts ONCE per embedding engine (shared across all summarizers)
            # ===================================================================
            print(f"    Fetching embeddings for {len(filtered_texts)} texts with {embedding_engine}...")
            try:
                if (
                    EMBEDDING_CACHE_DIR
                    and len(filtered_texts) == 30000
                    and dataset_name in ("dbpedia", "yahoo_answers")
                ):
                    shared_embeddings = await get_embeddings_with_file_cache(
                        filtered_texts,
                        embedding_engine,
                        db,
                        Path(EMBEDDING_CACHE_DIR),
                        dataset_name,
                        42,
                        30000,
                        fetch_embeddings_async,
                    )
                else:
                    shared_embeddings = await fetch_embeddings_async(
                        filtered_texts, embedding_engine, db,
                        provider_name=embedding_provider,
                    )
                print(f"    [OK] Fetched {len(shared_embeddings)} embeddings (shape: {shared_embeddings.shape})")
                print(f"    [INFO] Embedding dimension: {shared_embeddings.shape[1]} (full, no PCA)")
            except Exception as e:
                print(f"    [ERROR] Embedding fetch failed: {e}")
                print(f"    [INFO] Skipping all runs for this embedding engine...")
                import traceback
                traceback.print_exc()
                continue
            
            # Run sweep for each LLM summarizer
            for summarizer_entry in LLM_SUMMARIZERS:
                # Unpack (model, provider) tuple or handle None
                if summarizer_entry is None:
                    llm_deployment = None
                    llm_provider = "azure"
                else:
                    llm_deployment, llm_provider = summarizer_entry
                run_count += 1
                summarizer_name = "None" if llm_deployment is None else llm_deployment
                
                # Check if this run already exists
                # Normalize embedding engine name for filename (replace special chars)
                engine_safe = embedding_engine.replace('-', '_').replace('/', '_')
                existing_files = list(glob.glob(
                    str(OUTPUT_DIR / (
                        f"experimental_sweep_"
                        f"entry{ENTRY_MAX}_"
                        f"{dataset_name}_"
                        f"labelmax{label_max}_"
                        f"{engine_safe}_"
                        f"{summarizer_name.replace('-', '_')}_*.pkl"
                    ))
                ))
                if existing_files:
                    print(f"\n      [{run_count}/{total_runs}] {summarizer_name} (SKIP - already exists)")
                    continue
                
                print(f"\n      [{run_count}/{total_runs}] Summarizer: {summarizer_name} (provider: {llm_provider})")
                
                # Use the SHARED embeddings from this embedding engine
                embeddings = shared_embeddings
                
                # Run sweep with no-PCA config
                result = None
                try:
                    result = await asyncio.wait_for(
                        run_single_sweep_no_pca(
                            filtered_texts, embeddings, llm_deployment, db,
                            embedding_engine, NO_PCA_SWEEP_CONFIG,
                            llm_provider=llm_provider,
                            embedding_provider=embedding_provider,
                        ),
                        timeout=3600.0  # 1 hour timeout (longer for no-PCA)
                    )
                except asyncio.TimeoutError:
                    print(f"      [ERROR] Sweep execution timed out after 1 hour")
                    print(f"      [INFO] Skipping this run and continuing...")
                    continue
                except Exception as e:
                    print(f"      [WARN] Sweep execution failed: {e}")
                    print(f"      [INFO] Will attempt to continue...")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(OUTPUT_DIR / (
                    f"experimental_sweep_"
                    f"entry{ENTRY_MAX}_"
                    f"{dataset_name}_"
                    f"labelmax{label_max}_"
                    f"{engine_safe}_"
                    f"{summarizer_name.replace('-', '_')}_"
                    f"{timestamp}.pkl"
                ))
                
                # Save results with metadata
                metadata = {
                    "entry_max": ENTRY_MAX,
                    "label_max": label_max,
                    "actual_entry_count": len(filtered_texts),
                    "actual_label_count": len(set(filtered_labels)),
                    "benchmark_source": dataset_name,
                    "categories": dataset_info['category_names'],
                    "summarizer": summarizer_name,
                    "llm_provider": llm_provider,
                    "embedding_engine": embedding_engine,
                    "embedding_provider": embedding_provider,
                    "embedding_dimension": shared_embeddings.shape[1],
                    "sweep_config": {
                        "skip_pca": NO_PCA_SWEEP_CONFIG.skip_pca,
                        "k_min": NO_PCA_SWEEP_CONFIG.k_min,
                        "k_max": NO_PCA_SWEEP_CONFIG.k_max,
                        "max_iter": NO_PCA_SWEEP_CONFIG.max_iter,
                        "n_restarts": NO_PCA_SWEEP_CONFIG.n_restarts,
                        "compute_stability": NO_PCA_SWEEP_CONFIG.compute_stability,
                    },
                    "note": "No-PCA multi-embedding sweep: full dimensions, extended k range (1-20), single run",
                }
                
                try:
                    saved_file = save_results(
                        result,
                        output_file,
                        ground_truth_labels=filtered_labels,
                        dataset_name=dataset_name,
                        metadata=metadata,
                    )
                    print(f"      [OK] Saved to: {Path(saved_file).name}")
                except Exception as e:
                    print(f"      [ERROR] Failed to save results file: {e}")
                    print(f"      [WARN] Results were computed but could not be saved!")
                    import traceback
                    traceback.print_exc()
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / run_count
                remaining = (total_runs - run_count) * avg_time
                print(f"      Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
                print(f"      Elapsed: {elapsed/3600:.2f}h | ETA: {remaining/3600:.1f}h")
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"[OK] All experimental runs completed!")
    print(f"   Total runs: {run_count}")
    print(f"   Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-PCA multi-embedding sweep")
    parser.add_argument("--force", action="store_true", help="Remove existing no-PCA pickles and re-run all (to regenerate with Z for silhouette)")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
