#!/usr/bin/env python3
"""
Limited Test Sweep - Validation Run

Runs a small subset of the experimental sweep to validate the fix:
- entry_max: [100]
- dataset: 20newsgroups_6cat only
- label_max: [1, 2, 3]
- All summarizers: [None, gpt-4o-mini, gpt-4o, gpt-5-chat]

This will produce 12 pickle files (1 entry_max × 1 dataset × 3 label_max × 4 summarizers)

Usage:
    python scripts/run_limited_test_sweep.py
"""

import os
import sys

# Temporarily modify the sweep configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import and modify the main sweep module
import run_experimental_sweep

# Override configurations for limited test
run_experimental_sweep.ENTRY_MAX_VALUES = [100]
run_experimental_sweep.LABEL_MAX_VALUES = [1, 2, 3]
run_experimental_sweep.BENCHMARK_SOURCES = [
    {
        "name": "20newsgroups_6cat",
        "type": "20newsgroups",
        "categories": ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 
                      'rec.sport.hockey', 'sci.space', 'talk.politics.misc'],
    }
]

print("=" * 80)
print("LIMITED TEST SWEEP")
print("=" * 80)
print("\nConfiguration:")
print(f"  entry_max: {run_experimental_sweep.ENTRY_MAX_VALUES}")
print(f"  label_max: {run_experimental_sweep.LABEL_MAX_VALUES}")
print(f"  datasets: {[s['name'] for s in run_experimental_sweep.BENCHMARK_SOURCES]}")
print(f"  summarizers: {run_experimental_sweep.LLM_SUMMARIZERS}")
print(f"\nExpected output: {1 * 1 * 3 * 4} = 12 pickle files")
print("=" * 80)
print()

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_experimental_sweep.main())
