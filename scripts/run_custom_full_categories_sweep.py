"""
Custom Sweep: Full Categories, Variable K and Entry Count

Configuration:
- Datasets: dbpedia (14 categories), yahoo_answers (10 categories)
- label_max: FIXED at maximum (14 for dbpedia, 10 for yahoo_answers)
- entry_max: [100, 200, 300, 400, 500]
- k_range: 1 to 20 (extended from default 2-10)
- summarizers: All (None, gpt-4o-mini, gpt-4o, gpt-5-chat)

This sweep tests how clustering performance changes with:
1. Different numbers of datapoints (100-500)
2. Extended k range (1-20) 
3. Using ALL available categories (not subsampled)
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import base sweep functionality
from scripts.run_experimental_sweep import (
    DatabaseConnectionV2, DATABASE_URL,
    fetch_embeddings_async, run_single_sweep,
    create_paraphraser_for_llm, save_results,
    load_dbpedia_full, load_yahoo_answers_full,
    estimate_tokens, DEPLOYMENT_MAX_TOKENS, EMBEDDING_DEPLOYMENT,
    OUTPUT_DIR, LLM_SUMMARIZERS,
    SweepConfig
)
import numpy as np
import time
from datetime import datetime
import glob

# Custom configuration
ENTRY_MAX_VALUES = [100, 200, 300, 400, 500]

CUSTOM_BENCHMARKS = [
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

# Extended k range: 1 to 20
CUSTOM_SWEEP_CONFIG = SweepConfig(
    pca_dim=64,
    k_min=1,        # Start at 1 instead of 2
    k_max=20,       # Go up to 20 instead of 10
    max_iter=200,
    base_seed=0,
    n_restarts=20,  # Keep full restarts for stability
    compute_stability=True,
    coverage_threshold=0.2,
    llm_interval=20,
    max_samples=10,
)


async def main():
    """Main execution function."""
    print("=" * 80)
    print("Custom PCA KLLMeans Sweep - Full Categories, Extended K Range")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Datasets: {[s['name'] for s in CUSTOM_BENCHMARKS]}")
    print(f"  label_max: FIXED at max categories per dataset")
    print(f"    - dbpedia: 14 categories")
    print(f"    - yahoo_answers: 10 categories")
    print(f"  entry_max: {ENTRY_MAX_VALUES}")
    print(f"  k_range: {CUSTOM_SWEEP_CONFIG.k_min} to {CUSTOM_SWEEP_CONFIG.k_max}")
    print(f"  summarizers: {LLM_SUMMARIZERS}")
    print(f"  n_restarts: {CUSTOM_SWEEP_CONFIG.n_restarts}")
    
    total_runs = len(ENTRY_MAX_VALUES) * len(CUSTOM_BENCHMARKS) * len(LLM_SUMMARIZERS)
    print(f"\n[INFO] Total experimental runs: {total_runs}")
    print(f"   entry_max values: {len(ENTRY_MAX_VALUES)}")
    print(f"   benchmark sources: {len(CUSTOM_BENCHMARKS)}")
    print(f"   summarizers: {len(LLM_SUMMARIZERS)}")
    
    # Initialize database
    print("\n[INFO] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("[OK] Database initialized")
    
    # Load datasets
    print("\n[INFO] Loading full benchmark datasets...")
    print("   Filtering: 10 < length <= 1000 characters")
    
    loaded_datasets = {}
    for benchmark in CUSTOM_BENCHMARKS:
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
    
    # Run sweep
    print("\n" + "=" * 80)
    start_time = time.time()
    run_count = 0
    
    for entry_max in ENTRY_MAX_VALUES:
        print(f"\n{'='*80}")
        print(f"ENTRY_MAX: {entry_max}")
        print("="*80)
        
        for dataset_name, dataset_info in loaded_datasets.items():
            print(f"\n  BENCHMARK: {dataset_name}")
            
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
            
            # Sample up to entry_max
            if len(candidate_indices) > entry_max:
                np.random.seed(42)  # Reproducible sampling
                sampled_indices = np.random.choice(candidate_indices, size=entry_max, replace=False)
            else:
                sampled_indices = candidate_indices
            
            sampled_texts = [texts[i] for i in sampled_indices]
            sampled_labels = labels[sampled_indices]
            
            actual_entry_count = len(sampled_texts)
            actual_label_count = len(set(sampled_labels))
            
            print(f"    Sampled: {actual_entry_count} entries, {actual_label_count} unique labels (all categories)")
            
            # Filter texts that exceed token limit
            max_tokens = DEPLOYMENT_MAX_TOKENS.get(EMBEDDING_DEPLOYMENT)
            if max_tokens:
                valid_texts = []
                valid_indices = []
                for i, text in enumerate(sampled_texts):
                    try:
                        estimated = estimate_tokens(text, EMBEDDING_DEPLOYMENT)
                        if estimated <= max_tokens:
                            valid_texts.append(text)
                            valid_indices.append(i)
                    except Exception:
                        valid_texts.append(text)
                        valid_indices.append(i)
                
                if len(valid_texts) < len(sampled_texts):
                    print(f"      Filtered: {len(valid_texts)} valid, {len(sampled_texts) - len(valid_texts)} filtered out")
                    sampled_texts = valid_texts
                    sampled_labels = sampled_labels[valid_indices]
            
            # ===================================================================
            # CRITICAL: Embed original texts ONCE (shared across all summarizers)
            # ===================================================================
            print(f"      Fetching embeddings for {len(sampled_texts)} original texts...")
            try:
                shared_embeddings = await fetch_embeddings_async(sampled_texts, EMBEDDING_DEPLOYMENT, db)
                print(f"      [OK] Fetched {len(shared_embeddings)} embeddings (shared across all summarizers)")
            except Exception as e:
                print(f"      [ERROR] Embedding fetch failed: {e}")
                print(f"      [INFO] Skipping all runs for this dataset/entry combo...")
                import traceback
                traceback.print_exc()
                continue
            
            # Run sweep for each LLM summarizer
            for llm_deployment in LLM_SUMMARIZERS:
                run_count += 1
                summarizer_name = "None" if llm_deployment is None else llm_deployment
                
                # Check if this run already exists
                existing_files = list(glob.glob(
                    str(OUTPUT_DIR / (
                        f"experimental_sweep_"
                        f"entry{entry_max}_"
                        f"{dataset_name}_"
                        f"labelmax{label_max}_"
                        f"{summarizer_name.replace('-', '_')}_*.pkl"
                    ))
                ))
                if existing_files:
                    print(f"\n        [{run_count}/{total_runs}] Summarizer: {summarizer_name} (SKIP - already exists)")
                    continue
                
                print(f"\n        [{run_count}/{total_runs}] Summarizer: {summarizer_name}")
                
                # Use the SHARED embeddings from original texts
                embeddings = shared_embeddings
                
                # Run sweep
                result = None
                try:
                    result = await asyncio.wait_for(
                        run_single_sweep(sampled_texts, embeddings, llm_deployment, db, CUSTOM_SWEEP_CONFIG),
                        timeout=1800.0
                    )
                except asyncio.TimeoutError:
                    print(f"        [ERROR] Sweep execution timed out after 30 minutes")
                    print(f"        [INFO] Skipping this run and continuing...")
                    continue
                except Exception as e:
                    print(f"        [WARN] Sweep execution failed: {e}")
                    print(f"        [INFO] Will attempt to save any partial results...")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(OUTPUT_DIR / (
                    f"experimental_sweep_"
                    f"entry{entry_max}_"
                    f"{dataset_name}_"
                    f"labelmax{label_max}_"
                    f"{summarizer_name.replace('-', '_')}_"
                    f"{timestamp}.pkl"
                ))
                
                # Save results with metadata
                metadata = {
                    "entry_max": entry_max,
                    "label_max": label_max,
                    "actual_entry_count": actual_entry_count,
                    "actual_label_count": actual_label_count,
                    "benchmark_source": dataset_name,
                    "categories": dataset_info['category_names'],
                    "summarizer": summarizer_name,
                    "embedding_deployment": EMBEDDING_DEPLOYMENT,
                    "sweep_config": {
                        "pca_dim": CUSTOM_SWEEP_CONFIG.pca_dim,
                        "k_min": CUSTOM_SWEEP_CONFIG.k_min,
                        "k_max": CUSTOM_SWEEP_CONFIG.k_max,
                        "max_iter": CUSTOM_SWEEP_CONFIG.max_iter,
                        "n_restarts": CUSTOM_SWEEP_CONFIG.n_restarts,
                    },
                    "note": "Custom sweep: full categories, extended k range (1-20)",
                }
                
                try:
                    saved_file = save_results(
                        result,
                        output_file,
                        ground_truth_labels=sampled_labels,
                        dataset_name=dataset_name,
                        metadata=metadata,
                    )
                    print(f"        [OK] Saved to: {saved_file}")
                except Exception as e:
                    print(f"        [ERROR] Failed to save results file: {e}")
                    print(f"        [WARN] Results were computed but could not be saved!")
                    import traceback
                    traceback.print_exc()
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / run_count
                remaining = (total_runs - run_count) * avg_time
                print(f"        Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
                print(f"        ETA: {remaining/3600:.1f} hours")
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"[OK] All experimental runs completed!")
    print(f"   Total runs: {run_count}")
    print(f"   Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*80}")


# Need to update run_single_sweep to accept custom config
async def run_single_sweep_custom(
    texts, embeddings, llm_deployment, db, sweep_config
):
    """Run sweep with custom config."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.run_experimental_sweep import run_sweep
    
    paraphraser = create_paraphraser_for_llm(llm_deployment, db)
    
    async def embedder_func(texts_list):
        return await fetch_embeddings_async(texts_list, EMBEDDING_DEPLOYMENT, db)
    
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
                embedder=embedder_sync if paraphraser else None
            )
        )
    
    return result


# Monkey patch to pass custom config
async def run_single_sweep(texts, embeddings, llm_deployment, db, sweep_config=None):
    if sweep_config is None:
        from scripts.run_experimental_sweep import SWEEP_CONFIG as default_config
        sweep_config = default_config
    return await run_single_sweep_custom(texts, embeddings, llm_deployment, db, sweep_config)


if __name__ == "__main__":
    asyncio.run(main())
