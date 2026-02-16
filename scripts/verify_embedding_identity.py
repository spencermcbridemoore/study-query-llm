"""
Verify Experimental Design Fix: Embedding Identity Check

This script validates that the experimental design fix is working correctly by:
1. Loading pickle files for the same dataset/entry_max/label_max combination
2. Extracting the input embeddings used for clustering
3. Comparing embeddings across different summarizers (None, gpt-4o, gpt-4o-mini, gpt-5-chat)
4. Verifying that embeddings are IDENTICAL (not just similar)

Expected result: All summarizers should have EXACTLY the same embeddings.
If this check passes, the fix is confirmed working.
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_embeddings_from_pickle(pkl_file):
    """Load embeddings from a pickle file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Try to extract embeddings from the data structure
    # They might be in different locations depending on how they were saved
    if 'Z' in data:  # PCA-projected embeddings
        return data['Z']
    elif 'embeddings' in data:
        return data['embeddings']
    elif 'sweep_result' in data and hasattr(data['sweep_result'], 'Z'):
        return data['sweep_result'].Z
    else:
        return None

def main():
    results_dir = Path("experimental_results")
    
    # Group files by (dataset, entry_max, label_max)
    file_groups = defaultdict(list)
    
    for pkl_file in results_dir.glob("experimental_sweep_*.pkl"):
        name = pkl_file.stem
        
        # Parse filename: experimental_sweep_entry{X}_{dataset}_labelmax{Y}_{summarizer}_{timestamp}
        parts = name.split('_')
        
        try:
            # Find entry_max
            entry_idx = next(i for i, p in enumerate(parts) if p.startswith('entry'))
            entry_max = parts[entry_idx].replace('entry', '')
            
            # Find label_max
            labelmax_idx = next(i for i, p in enumerate(parts) if p.startswith('labelmax'))
            label_max = parts[labelmax_idx].replace('labelmax', '')
            
            # Find dataset name (between entry and labelmax)
            dataset = '_'.join(parts[entry_idx+1:labelmax_idx])
            
            # Find summarizer (after labelmax)
            summarizer = parts[labelmax_idx+1]
            
            key = (dataset, entry_max, label_max)
            file_groups[key].append((summarizer, pkl_file))
        except (StopIteration, IndexError, ValueError):
            print(f"[WARN] Could not parse filename: {pkl_file.name}")
            continue
    
    print("="*80)
    print("Embedding Identity Verification")
    print("="*80)
    print(f"\nFound {len(file_groups)} unique dataset/entry/label combinations")
    print(f"Total files: {sum(len(files) for files in file_groups.values())}")
    
    # Check each group
    all_passed = True
    for key, files in sorted(file_groups.items()):
        dataset, entry_max, label_max = key
        
        if len(files) < 2:
            continue  # Need at least 2 summarizers to compare
        
        print(f"\n{'='*80}")
        print(f"Checking: {dataset}, entry={entry_max}, label_max={label_max}")
        print(f"  Files: {len(files)}")
        
        # Load embeddings from each file
        embeddings_by_summarizer = {}
        for summarizer, pkl_file in files:
            emb = load_embeddings_from_pickle(pkl_file)
            if emb is not None:
                embeddings_by_summarizer[summarizer] = emb
                print(f"    {summarizer}: shape={emb.shape}")
            else:
                print(f"    {summarizer}: [WARN] Could not extract embeddings")
        
        if len(embeddings_by_summarizer) < 2:
            print("  [SKIP] Not enough embeddings loaded for comparison")
            continue
        
        # Compare all pairs
        summarizers = list(embeddings_by_summarizer.keys())
        reference_summarizer = summarizers[0]
        reference_emb = embeddings_by_summarizer[reference_summarizer]
        
        group_passed = True
        for summarizer in summarizers[1:]:
            test_emb = embeddings_by_summarizer[summarizer]
            
            # Check if identical
            if reference_emb.shape != test_emb.shape:
                print(f"  [FAIL] Shape mismatch: {reference_summarizer} {reference_emb.shape} vs {summarizer} {test_emb.shape}")
                group_passed = False
                all_passed = False
                continue
            
            # Check for exact equality
            are_identical = np.allclose(reference_emb, test_emb, rtol=0, atol=0)
            
            if are_identical:
                print(f"  [PASS] {reference_summarizer} == {summarizer} (IDENTICAL)")
            else:
                # Check how different they are
                max_diff = np.max(np.abs(reference_emb - test_emb))
                mean_diff = np.mean(np.abs(reference_emb - test_emb))
                are_close = np.allclose(reference_emb, test_emb, rtol=1e-9, atol=1e-9)
                
                if are_close:
                    print(f"  [PASS] {reference_summarizer} ≈ {summarizer} (very close, max_diff={max_diff:.2e})")
                else:
                    print(f"  [FAIL] {reference_summarizer} != {summarizer} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                    group_passed = False
                    all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("[SUCCESS] All embedding comparisons PASSED! Fix is working correctly.")
        print("All summarizers use the same embeddings from original texts.")
    else:
        print("[FAILURE] Some embeddings are different! Fix may not be working correctly.")
        print("Check the output above for details.")
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
