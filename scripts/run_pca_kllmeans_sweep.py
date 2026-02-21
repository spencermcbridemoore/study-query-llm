#!/usr/bin/env python3
"""
PCA KLLMeans Sweep - Standalone Python Script

This script runs the PCA KLLMeans sweep analysis using the algorithms library
and service layer. Can be run directly in Python or Jupyter environments.

Usage:
    python scripts/run_pca_kllmeans_sweep.py

Or in Jupyter:
    %run scripts/run_pca_kllmeans_sweep.py
"""

import os
import sys
import asyncio
import time
import pickle
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add src and repo root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.embedding_service import EmbeddingService, EmbeddingRequest
from study_query_llm.services.summarization_service import SummarizationService, SummarizationRequest
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.algorithms import SweepConfig, run_sweep

from scripts.common.data_utils import flatten_prompt_dict, clean_texts, is_prompt_key
from scripts.common.embedding_utils import fetch_embeddings_async
from scripts.common.sweep_utils import (
    create_paraphraser_for_llm,
    save_batch_sweep_results as save_results,
)

# Try to apply nest_asyncio for Jupyter compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Not in Jupyter, no need for nest_asyncio


# ============================================================================
# Configuration
# ============================================================================

# Embedding deployment
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# LLM deployments for summarization (3 LLMs + None = 4 runs)
LLM_SUMMARIZERS = [
    None,  # Non-LLM summaries (just use original representatives)
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-chat-2025-08-07",  # Change this to your preferred third LLM
]

# Sweep configuration
SWEEP_CONFIG = SweepConfig(
    pca_dim=64,
    rank_r=2,
    k_min=2,
    k_max=10,
    max_iter=200,
    base_seed=0,
    n_restarts=20,  # Multiple restarts for stability analysis
    compute_stability=True,  # Enable stability metrics
    coverage_threshold=0.2,
)

# Database URL (required)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable must be set. "
        "Example: postgresql://user:pass@host/db?sslmode=require"
    )


# ============================================================================
# Helper Functions
# ============================================================================

_clean_texts = clean_texts  # backward-compat alias


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution function."""
    print("=" * 60)
    print("PCA KLLMeans Sweep Analysis")
    print("=" * 60)

    # Initialize database
    print("\n[INFO] Initializing database...")
    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    print("[OK] Database initialized")

    # Load prompt dictionary
    # Option 1: Load from pickle file
    # Option 2: Load from JSON file
    # Option 3: Define directly in script (see below)
    print("\n[INFO] Loading prompts...")
    
    database_estela_dict = None
    
    # Try to load from pickle file first
    prompt_dict_file = os.environ.get("PROMPT_DICT_FILE")
    if prompt_dict_file and os.path.exists(prompt_dict_file):
        print(f"   Loading from pickle file: {prompt_dict_file}")
        with open(prompt_dict_file, "rb") as f:
            database_estela_dict = pickle.load(f)
    else:
        # Try to load from JSON file
        prompt_dict_json = os.environ.get("PROMPT_DICT_JSON")
        if prompt_dict_json and os.path.exists(prompt_dict_json):
            print(f"   Loading from JSON file: {prompt_dict_json}")
            import json
            with open(prompt_dict_json, "r", encoding="utf-8") as f:
                database_estela_dict = json.load(f)
        else:
            # Option 3: Define directly in script
            # Uncomment and modify this section to define your dictionary:
            # database_estela_dict = {
            #     'path/to/file.yaml': {
            #         'generation prompts': [
            #             {'prompt 1': 'Your prompt text here...'},
            #             ...
            #         ],
            #         ...
            #     },
            #     ...
            # }
            
            # For now, create empty dict - user must define it
            database_estela_dict = {}
            
            if not database_estela_dict:
                print("\n[ERROR] database_estela_dict is empty.")
                print("   Please do one of the following:")
                print("   1. Set PROMPT_DICT_FILE environment variable to a pickle file path")
                print("   2. Set PROMPT_DICT_JSON environment variable to a JSON file path")
                print("   3. Define database_estela_dict directly in the script (see comments)")
                return

    # Flatten prompts and extract texts
    flat_prompts = flatten_prompt_dict(database_estela_dict)
    texts = list(flat_prompts.values())
    texts = _clean_texts(texts)
    print(f"[OK] Loaded {len(texts)} valid prompts")

    # Show samples
    print("\n[INFO] Sample prompts:")
    for i, (k, v) in enumerate(list(flat_prompts.items())[:3]):
        print(f"  {i+1}. Key: {k}")
        print(f"     Text: {v[:100]}{'...' if len(v) > 100 else ''}")

    # Fetch embeddings
    print(f"\n[INFO] Fetching embeddings using {EMBEDDING_DEPLOYMENT}...")
    embeddings = await fetch_embeddings_async(texts, EMBEDDING_DEPLOYMENT, db)
    print(f"[OK] Got embeddings: shape {embeddings.shape}")

    # Create run group for provenance tracking
    print("\n[INFO] Creating run group for provenance tracking...")
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repository=repo)

        run_group_id = provenance.create_run_group(
            algorithm="pca_kllmeans_sweep",
            name=f"pca_kllmeans_sweep_{EMBEDDING_DEPLOYMENT}",
            config={
                "embedding_deployment": EMBEDDING_DEPLOYMENT,
                "n_texts": len(texts),
                "k_range": f"{SWEEP_CONFIG.k_min}-{SWEEP_CONFIG.k_max}",
                "llm_summarizers": [
                    s if s else "None" for s in LLM_SUMMARIZERS
                ],
            },
        )
        print(f"[OK] Created run group: id={run_group_id}")

    # Run sweep for each LLM summarizer concurrently
    print("\n[INFO] Running sweeps concurrently...")
    
    async def run_single_sweep(llm_deployment: str) -> tuple[str, Any]:
        """Run a single sweep for a given LLM deployment."""
        summarizer_name = "None" if llm_deployment is None else llm_deployment
        
        # Create paraphraser
        paraphraser = create_paraphraser_for_llm(llm_deployment, db)
        
        # Run sweep in thread pool executor (CPU-bound work)
        # The paraphraser will handle async internally
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: run_sweep(texts, embeddings, SWEEP_CONFIG, paraphraser=paraphraser)
            )
        
        print(f"[OK] Completed {summarizer_name}. Ks: {sorted([int(k) for k in result.by_k.keys()])}")
        return summarizer_name, result
    
    # Create tasks for all sweeps
    tasks = [
        run_single_sweep(llm_deployment)
        for llm_deployment in LLM_SUMMARIZERS
    ]
    
    # Run all sweeps concurrently with progress tracking
    # Note: async_tqdm.gather() doesn't support return_exceptions parameter
    # Use asyncio.gather() with return_exceptions=True
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and check for errors
    all_results = {}
    for result_item in results_list:
        if isinstance(result_item, Exception):
            print(f"[ERROR] Error in sweep: {result_item}")
            raise result_item
        summarizer_name, result = result_item
        all_results[summarizer_name] = result

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Embedding deployment: {EMBEDDING_DEPLOYMENT}")
    print(f"Number of texts: {len(texts)}")
    print(f"K range: {SWEEP_CONFIG.k_min} to {SWEEP_CONFIG.k_max}")
    print(
        f"Summarizers tested: {len(LLM_SUMMARIZERS)} ({', '.join([s if s else 'None' for s in LLM_SUMMARIZERS])})"
    )
    print(f"\nResults structure: all_results[summarizer_name]['by_k'][k_value]")

    # Save results
    print("\n[INFO] Saving results...")
    output_file = save_results(all_results)
    print(f"[OK] Results saved to: {output_file}")
    print(f"   Includes: representatives, labels, objectives, stability metrics, and distance matrices")
    print(f"\n   Example access:")
    print(f"     results = pickle.load(open('{output_file}', 'rb'))")
    print(f"     results['None']['by_k']['5']['stability']['silhouette']['mean']")
    print(f"     results['None']['by_k']['5']['representatives']")

    # Display results
    print(f"\n{'=' * 60}")
    print("RESULTS PREVIEW")
    print(f"{'=' * 60}")
    for summarizer_name, result in all_results.items():
        print(f"\n{summarizer_name}:")
        for k in sorted([int(k) for k in result.by_k.keys()])[:3]:  # Show first 3 K values
            k_data = result.by_k[str(k)]
            reps = k_data.get("representatives", [])
            print(f"  K={k}: {len(reps)} representatives")
            if k_data.get("stability"):
                stab = k_data["stability"]
                print(f"    Silhouette: {stab['silhouette']['mean']:.3f} ± {stab['silhouette']['std']:.3f}")
                print(f"    Stability ARI: {stab['stability_ari']['mean']:.3f} ± {stab['stability_ari']['std']:.3f}")

    print("\n[OK] Analysis complete!")
    return all_results, output_file


if __name__ == "__main__":
    # Run the main function
    results, output_file = asyncio.run(main())
    print(f"\n[INFO] Results saved to: {output_file}")
