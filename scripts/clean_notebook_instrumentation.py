#!/usr/bin/env python3
"""Remove instrumentation from notebook while keeping fixes."""

import json
import os
import re
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent.parent
os.chdir(script_dir)

notebook_path = Path("notebooks/pca_kllmeans_sweep.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove instrumentation blocks from all cells
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        if isinstance(source_lines, str):
            source_lines = source_lines.split('\n')
        
        new_lines = []
        skip_block = False
        i = 0
        while i < len(source_lines):
            line = source_lines[i]
            
            # Check for start of instrumentation block
            if "# #region agent log" in line or "#region agent log" in line:
                skip_block = True
                i += 1
                continue
            
            # Check for end of instrumentation block
            if "# #endregion" in line or "#endregion" in line:
                skip_block = False
                i += 1
                continue
            
            # Skip lines inside instrumentation block
            if skip_block:
                i += 1
                continue
            
            # Keep all other lines
            new_lines.append(line)
            i += 1
        
        # Update cell source
        if isinstance(cell["source"], str):
            cell["source"] = '\n'.join(new_lines)
        else:
            cell["source"] = new_lines

# Save the notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook instrumentation cleaned successfully")
