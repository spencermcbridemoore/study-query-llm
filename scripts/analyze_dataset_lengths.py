#!/usr/bin/env python3
"""
Analyze character length distributions for benchmark datasets.

This script loads each benchmark dataset and computes character length
statistics including percentiles.
"""

import os
import sys
import pickle
import json
import numpy as np
from typing import List, Tuple, Dict, Any

# Add src and repo root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_experimental_sweep import (
    BENCHMARK_SOURCES,
    load_benchmark_dataset,
)
from study_query_llm.utils.text_utils import flatten_prompt_dict, clean_texts

_clean_texts = clean_texts  # backward-compat alias


def load_estela_dict() -> Tuple[str, List[str]]:
    """
    Load estela prompt dictionary.
    
    Returns:
        (dataset_name, texts)
    """
    database_estela_dict = None
    
    # Try to load from pickle file first
    default_file = "notebooks/estela_prompt_data.pkl"
    if os.path.exists(default_file):
        with open(default_file, "rb") as f:
            database_estela_dict = pickle.load(f)
    else:
        # Try environment variable
        prompt_dict_file = os.environ.get("PROMPT_DICT_FILE")
        if prompt_dict_file and os.path.exists(prompt_dict_file):
            with open(prompt_dict_file, "rb") as f:
                database_estela_dict = pickle.load(f)
        else:
            prompt_dict_json = os.environ.get("PROMPT_DICT_JSON")
            if prompt_dict_json and os.path.exists(prompt_dict_json):
                with open(prompt_dict_json, "r", encoding="utf-8") as f:
                    database_estela_dict = json.load(f)
            else:
                raise ValueError(
                    "No estela dictionary file found. "
                    "Expected: notebooks/estela_prompt_data.pkl or set PROMPT_DICT_FILE/PROMPT_DICT_JSON"
                )
    
    # Flatten prompts and extract texts
    flat_prompts = flatten_prompt_dict(database_estela_dict)
    texts = list(flat_prompts.values())
    texts = _clean_texts(texts)
    
    return "estela_prompt_dict", texts


def analyze_dataset_lengths(source_config: Dict[str, Any]) -> Tuple[str, Dict[str, float], int]:
    """
    Analyze character lengths for a dataset.
    
    Args:
        source_config: Dataset configuration dictionary
        
    Returns:
        (dataset_name, percentiles_dict, total_count)
    """
    try:
        texts, labels, category_names = load_benchmark_dataset(source_config, random_state=42)
        
        # Calculate character lengths
        lengths = [len(text) for text in texts]
        
        # Calculate percentiles
        percentiles = {
            '1st': np.percentile(lengths, 1),
            '5th': np.percentile(lengths, 5),
            '10th': np.percentile(lengths, 10),
            '25th': np.percentile(lengths, 25),
            '50th': np.percentile(lengths, 50),
            '75th': np.percentile(lengths, 75),
            '90th': np.percentile(lengths, 90),
            '95th': np.percentile(lengths, 95),
            '99th': np.percentile(lengths, 99),
        }
        
        return source_config['name'], percentiles, len(texts)
    except Exception as e:
        return source_config['name'], None, 0


def analyze_estela_lengths() -> Tuple[str, Dict[str, float], int]:
    """
    Analyze character lengths for estela dictionary.
    
    Returns:
        (dataset_name, percentiles_dict, total_count)
    """
    try:
        name, texts = load_estela_dict()
        
        # Calculate character lengths
        lengths = [len(text) for text in texts]
        
        # Calculate percentiles
        percentiles = {
            '1st': np.percentile(lengths, 1),
            '5th': np.percentile(lengths, 5),
            '10th': np.percentile(lengths, 10),
            '25th': np.percentile(lengths, 25),
            '50th': np.percentile(lengths, 50),
            '75th': np.percentile(lengths, 75),
            '90th': np.percentile(lengths, 90),
            '95th': np.percentile(lengths, 95),
            '99th': np.percentile(lengths, 99),
        }
        
        return name, percentiles, len(texts)
    except Exception as e:
        return "estela_prompt_dict", None, 0


def main():
    """Main execution function."""
    print("=" * 80)
    print("Dataset Character Length Analysis")
    print("=" * 80)
    print()
    
    results = []
    
    # Analyze benchmark datasets
    for source in BENCHMARK_SOURCES:
        print(f"Analyzing {source['name']}...", end=" ")
        name, percentiles, count = analyze_dataset_lengths(source)
        
        if percentiles is not None:
            results.append((name, percentiles, count))
            print(f"OK ({count} datapoints)")
        else:
            print(f"FAILED")
    
    # Analyze estela dictionary
    print(f"Analyzing estela_prompt_dict...", end=" ")
    name, percentiles, count = analyze_estela_lengths()
    
    if percentiles is not None:
        results.append((name, percentiles, count))
        print(f"OK ({count} datapoints)")
    else:
        print(f"FAILED (file not found or error)")
    
    print()
    print("=" * 80)
    print("Results Table")
    print("=" * 80)
    print()
    
    # Print table header
    header = f"{'Dataset':<25} {'Count':<8} {'1st':<8} {'5th':<8} {'10th':<8} {'25th':<8} {'50th':<8} {'75th':<8} {'90th':<8} {'95th':<8} {'99th':<8}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for name, percentiles, count in results:
        row = (
            f"{name:<25} "
            f"{count:<8} "
            f"{percentiles['1st']:<8.0f} "
            f"{percentiles['5th']:<8.0f} "
            f"{percentiles['10th']:<8.0f} "
            f"{percentiles['25th']:<8.0f} "
            f"{percentiles['50th']:<8.0f} "
            f"{percentiles['75th']:<8.0f} "
            f"{percentiles['90th']:<8.0f} "
            f"{percentiles['95th']:<8.0f} "
            f"{percentiles['99th']:<8.0f}"
        )
        print(row)
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total datasets analyzed: {len(results)}")
    print(f"Total datasets failed: {len(BENCHMARK_SOURCES) - len(results)}")
    
    # Also print as markdown table for easy copy
    print()
    print("Markdown Table:")
    print()
    print("| Dataset | Count | 1st | 5th | 10th | 25th | 50th | 75th | 90th | 95th | 99th |")
    print("|---------|-------|-----|-----|------|------|------|------|------|------|------|")
    for name, percentiles, count in results:
        print(
            f"| {name} | {count} | {percentiles['1st']:.0f} | {percentiles['5th']:.0f} | "
            f"{percentiles['10th']:.0f} | {percentiles['25th']:.0f} | {percentiles['50th']:.0f} | "
            f"{percentiles['75th']:.0f} | {percentiles['90th']:.0f} | {percentiles['95th']:.0f} | "
            f"{percentiles['99th']:.0f} |"
        )


if __name__ == "__main__":
    main()
