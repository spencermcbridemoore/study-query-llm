"""
Limited Test Sweep - Verify Experimental Design Fix

Tests the corrected experimental design with a small subset:
- entry_max: [100]
- label_max: [1, 2, 3]
- datasets: [dbpedia]  # Using dbpedia since it had no content filter issues
- summarizers: All (None, gpt-4o-mini, gpt-4o, gpt-5-chat)

This will verify:
1. Embeddings are computed once and shared across all summarizers
2. LLMs only influence in-loop centroid updates
3. No pre-summarization happens
4. Results are saved correctly
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and override configurations from run_experimental_sweep
from scripts.run_experimental_sweep import *

# Override configurations for limited test
ENTRY_MAX_VALUES = [100]
LABEL_MAX_VALUES = [1, 2, 3]
BENCHMARK_SOURCES = [
    {
        "name": "dbpedia",
        "type": "dbpedia",
        "categories": None,  # 14 categories
    },
]

# Update sweep config for faster testing (fewer restarts)
SWEEP_CONFIG = SweepConfig(
    pca_dim=64,
    k_min=2,
    k_max=10,
    max_iter=200,
    base_seed=0,
    n_restarts=5,  # Reduced from 20 for faster testing
    compute_stability=True,
    coverage_threshold=0.2,
    llm_interval=20,
    max_samples=10,
)

if __name__ == "__main__":
    print("="*80)
    print("LIMITED TEST SWEEP - Verify Experimental Design Fix")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  entry_max: {ENTRY_MAX_VALUES}")
    print(f"  label_max: {LABEL_MAX_VALUES}")
    print(f"  datasets: {[s['name'] for s in BENCHMARK_SOURCES]}")
    print(f"  summarizers: {LLM_SUMMARIZERS}")
    print(f"  n_restarts: {SWEEP_CONFIG.n_restarts} (reduced for testing)")
    print(f"\nExpected runs: {len(ENTRY_MAX_VALUES)} × {len(LABEL_MAX_VALUES)} × {len(BENCHMARK_SOURCES)} × {len(LLM_SUMMARIZERS)} = {len(ENTRY_MAX_VALUES) * len(LABEL_MAX_VALUES) * len(BENCHMARK_SOURCES) * len(LLM_SUMMARIZERS)}")
    print("="*80)
    
    asyncio.run(main())
