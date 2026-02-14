#!/usr/bin/env python3
"""Find and extract estela dictionary from notebook."""

import json

with open('notebooks/pca_kllmeans_sweep.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if 'source' in cell:
        source = ''.join(cell['source'])
        if 'database_estela_dict' in source:
            # Check if it's a definition (commented or not)
            if 'database_estela_dict =' in source or '#database_estela_dict =' in source:
                print(f"\n=== Cell {i} ===")
                print(f"Length: {len(source)} characters")
                print(f"First 200 chars: {source[:200]}")
                print(f"Last 200 chars: {source[-200:]}")
                # Check if it's commented
                lines = cell['source']
                commented = sum(1 for line in lines if line.strip().startswith('#'))
                total = len([l for l in lines if l.strip()])
                print(f"Commented lines: {commented}/{total}")
