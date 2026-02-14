#!/usr/bin/env python3
"""Extract estela dictionary from notebook and save to pickle."""

import json
import pickle

# Load notebook
with open('notebooks/pca_kllmeans_sweep.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find cell with estela dictionary (commented or uncommented)
database_estela_dict = None
for cell in notebook['cells']:
    if 'source' in cell:
        source_text = ''.join(cell['source'])
        # Look for database_estela_dict definition
        if 'database_estela_dict' in source_text and ('=' in source_text or '{' in source_text):
            # Try to execute the cell (uncomment if needed)
            try:
                # Remove comment markers and execute
                code = source_text
                if code.strip().startswith('#'):
                    # It's commented, uncomment it
                    code = '\n'.join([line[1:] if line.strip().startswith('#') else line 
                                     for line in code.split('\n')])
                
                # Execute in a safe namespace
                namespace = {}
                exec(code, namespace)
                if 'database_estela_dict' in namespace:
                    database_estela_dict = namespace['database_estela_dict']
                    print(f"Found estela dictionary with {len(database_estela_dict)} entries")
                    break
            except Exception as e:
                continue

if database_estela_dict:
    # Save to pickle
    with open('notebooks/estela_prompt_data.pkl', 'wb') as f:
        pickle.dump(database_estela_dict, f)
    print(f"Saved estela dictionary to notebooks/estela_prompt_data.pkl")
else:
    print("Could not find estela dictionary in notebook")
