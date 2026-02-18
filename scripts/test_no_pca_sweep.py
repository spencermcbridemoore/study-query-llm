"""
Test script for no-PCA multi-embedding sweep.

Tests with:
- 1 dataset (dbpedia)
- 1 embedding engine (text-embedding-3-small)
- 2 summarizers (None, gpt-4o-mini)
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the main sweep script
from scripts.run_no_pca_multi_embedding_sweep import (
    NO_PCA_SWEEP_CONFIG, DATASETS, LLM_SUMMARIZERS, ENTRY_MAX,
    DatabaseConnectionV2, DATABASE_URL, OUTPUT_DIR,
    fetch_embeddings_async, run_single_sweep_no_pca,
    save_results, estimate_tokens, DEPLOYMENT_MAX_TOKENS
)
import numpy as np
from datetime import datetime
import time

# Test configuration
TEST_EMBEDDING_ENGINE = "text-embedding-3-small"
TEST_SUMMARIZERS = [None, "gpt-4o-mini"]  # Just 2 for testing
TEST_DATASET = DATASETS[0]  # Just dbpedia


async def test_run():
    """Test the no-PCA sweep with minimal configuration."""
    print("=" * 80)
    print("TEST: No-PCA Multi-Embedding Sweep")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Dataset: {TEST_DATASET['name']}")
    print(f"  Embedding engine: {TEST_EMBEDDING_ENGINE}")
    print(f"  Summarizers: {['None' if s is None else s for s in TEST_SUMMARIZERS]}")
    print(f"  Entry max: {ENTRY_MAX}")
    print(f"  Skip PCA: {NO_PCA_SWEEP_CONFIG.skip_pca}")
    print(f"  K range: {NO_PCA_SWEEP_CONFIG.k_min} to {NO_PCA_SWEEP_CONFIG.k_max}")
    
    # Initialize database
    print("\n[INFO] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("[OK] Database initialized")
    
    # Load dataset
    print(f"\n[INFO] Loading {TEST_DATASET['name']}...")
    texts, labels, category_names = TEST_DATASET['loader']()
    print(f"[OK] Loaded {len(texts)} texts, {len(set(labels))} unique labels")
    
    label_max = TEST_DATASET['label_max']
    
    # Sample entries
    unique_labels = sorted(set(labels))
    selected_labels = unique_labels[:label_max]
    mask = np.isin(labels, selected_labels)
    candidate_indices = np.where(mask)[0]
    
    if len(candidate_indices) > ENTRY_MAX:
        np.random.seed(42)
        sampled_indices = np.random.choice(candidate_indices, size=ENTRY_MAX, replace=False)
    else:
        sampled_indices = candidate_indices
    
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    print(f"[INFO] Sampled: {len(sampled_texts)} entries, {len(set(sampled_labels))} unique labels")
    
    # Filter by token limit
    max_tokens = DEPLOYMENT_MAX_TOKENS.get(TEST_EMBEDDING_ENGINE)
    if max_tokens:
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(sampled_texts):
            try:
                estimated = estimate_tokens(text, TEST_EMBEDDING_ENGINE)
                if estimated <= max_tokens:
                    valid_texts.append(text)
                    valid_indices.append(i)
            except Exception:
                valid_texts.append(text)
                valid_indices.append(i)
        
        if len(valid_texts) < len(sampled_texts):
            print(f"[INFO] Filtered: {len(valid_texts)} valid texts")
            sampled_texts = valid_texts
            sampled_labels = sampled_labels[valid_indices]
    
    # Embed texts
    print(f"\n[INFO] Fetching embeddings with {TEST_EMBEDDING_ENGINE}...")
    start_embed = time.time()
    try:
        embeddings = await fetch_embeddings_async(sampled_texts, TEST_EMBEDDING_ENGINE, db)
        embed_time = time.time() - start_embed
        print(f"[OK] Fetched embeddings (shape: {embeddings.shape}) in {embed_time:.1f}s")
        print(f"[INFO] Embedding dimension: {embeddings.shape[1]} (full, no PCA)")
    except Exception as e:
        print(f"[ERROR] Embedding fetch failed: {e}")
        return False
    
    # Run sweep for each test summarizer
    for summarizer in TEST_SUMMARIZERS:
        summarizer_name = "None" if summarizer is None else summarizer
        print(f"\n{'='*80}")
        print(f"Testing Summarizer: {summarizer_name}")
        print("="*80)
        
        start_sweep = time.time()
        try:
            result = await run_single_sweep_no_pca(
                sampled_texts, embeddings, summarizer, db,
                TEST_EMBEDDING_ENGINE, NO_PCA_SWEEP_CONFIG
            )
            sweep_time = time.time() - start_sweep
            print(f"[OK] Sweep completed in {sweep_time:.1f}s ({sweep_time/60:.1f}min)")
            
            # Check results
            print(f"\n[INFO] Results:")
            print(f"  PCA metadata: {result.pca}")
            print(f"  K values tested: {sorted(result.by_k.keys())}")
            print(f"  Skip PCA: {result.pca.get('skip_pca', 'not set')}")
            
            # Verify no-PCA was used
            if result.pca.get('skip_pca') != True:
                print(f"[ERROR] skip_pca should be True but got {result.pca.get('skip_pca')}")
                return False
            
            if result.pca.get('pca_dim_used') != embeddings.shape[1]:
                print(f"[ERROR] Expected dimension {embeddings.shape[1]} but got {result.pca.get('pca_dim_used')}")
                return False
            
            print(f"[PASS] No-PCA mode verified: using full {embeddings.shape[1]}-dimensional space")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            engine_safe = TEST_EMBEDDING_ENGINE.replace('-', '_')
            output_file = str(OUTPUT_DIR / (
                f"test_experimental_sweep_"
                f"entry{ENTRY_MAX}_"
                f"{TEST_DATASET['name']}_"
                f"labelmax{label_max}_"
                f"{engine_safe}_"
                f"{summarizer_name.replace('-', '_')}_"
                f"{timestamp}.pkl"
            ))
            
            metadata = {
                "entry_max": ENTRY_MAX,
                "label_max": label_max,
                "benchmark_source": TEST_DATASET['name'],
                "summarizer": summarizer_name,
                "embedding_engine": TEST_EMBEDDING_ENGINE,
                "embedding_dimension": embeddings.shape[1],
                "sweep_config": {
                    "skip_pca": NO_PCA_SWEEP_CONFIG.skip_pca,
                    "k_min": NO_PCA_SWEEP_CONFIG.k_min,
                    "k_max": NO_PCA_SWEEP_CONFIG.k_max,
                },
                "note": "TEST run for no-PCA multi-embedding sweep",
            }
            
            saved_file = save_results(
                result, output_file,
                ground_truth_labels=sampled_labels,
                dataset_name=TEST_DATASET['name'],
                metadata=metadata
            )
            print(f"[OK] Saved test file: {Path(saved_file).name}")
            
        except Exception as e:
            print(f"[ERROR] Sweep failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*80}")
    print("[SUCCESS] All test runs completed successfully!")
    print("="*80)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_run())
    sys.exit(0 if success else 1)
