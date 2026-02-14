#!/usr/bin/env python3
"""
Analyze character length distributions for estela prompt dictionary.

This script loads the estela dictionary and computes character length
statistics including percentiles.
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Tuple, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _is_prompt_key(key: str) -> bool:
    """Check if a key represents a prompt."""
    key_lower = key.lower()
    return "prompt" in key_lower


def flatten_prompt_dict(data, path=()):
    """
    Flatten nested prompt dictionary into a flat map of key tuples -> prompt strings.
    
    This function is copied from notebooks/pca_kllmeans_sweep.ipynb
    """
    flat = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + (key,)
            if isinstance(key, str) and _is_prompt_key(key) and isinstance(value, str):
                flat[new_path] = value
            else:
                flat.update(flatten_prompt_dict(value, new_path))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_path = path + (f"[{i}]",)
            flat.update(flatten_prompt_dict(value, new_path))
    return flat


def load_estela_dict() -> List[str]:
    """
    Load estela prompt dictionary and flatten to list of strings.
    
    Tries to load from notebook cell first, then falls back to pickle/JSON files.
    
    Returns:
        List of prompt strings
    """
    database_estela_dict = None
    
    # First, try to load from notebook cell 13 (commented out definition)
    try:
        import json as json_lib
        print("Attempting to load estela dictionary from notebook cell 13...")
        with open("notebooks/pca_kllmeans_sweep.ipynb", "r", encoding="utf-8") as f:
            notebook = json_lib.load(f)
        
        # Cell 13 contains the large estela dictionary definition
        if len(notebook['cells']) > 13:
            cell = notebook['cells'][13]
            if 'source' in cell and cell.get('cell_type') == 'code':
                source_lines = cell['source']
                
                # Find the uncommented dictionary definition line (the one with the literal dict, not pickle.load)
                dict_line_idx = None
                for i, line in enumerate(source_lines):
                    if ('database_estela_dict =' in line and 
                        '{' in line and  # Has dictionary literal
                        'pickle.load' not in line and  # Not loading from pickle
                        not line.lstrip().startswith('#')):
                        dict_line_idx = i
                        break
                
                if dict_line_idx is not None:
                    # Extract the dictionary assignment and everything after it
                    # Build code starting from the assignment
                    assignment_line = source_lines[dict_line_idx]
                    
                    # Check if assignment is indented
                    leading_spaces_in_assignment = len(assignment_line) - len(assignment_line.lstrip())
                    
                    # Collect all lines from assignment to end of cell, fixing indentation
                    # The dictionary definition spans to the end of the cell
                    code_parts = []
                    
                    # Start with just the assignment (remove leading space to make it top-level)
                    if leading_spaces_in_assignment > 0:
                        # Remove leading spaces from assignment line
                        clean_assignment = assignment_line[leading_spaces_in_assignment:]
                        code_parts.append(clean_assignment)
                    else:
                        code_parts.append(assignment_line)
                    
                    # For subsequent lines, maintain relative indentation
                    # Continue until end of cell (the dict definition is the last thing)
                    for line in source_lines[dict_line_idx + 1:]:
                        if line.strip():  # Non-empty line
                            current_indent = len(line) - len(line.lstrip())
                            if current_indent >= leading_spaces_in_assignment:
                                # Reduce indentation by the amount from assignment line
                                adjusted = line[leading_spaces_in_assignment:]
                                code_parts.append(adjusted)
                            else:
                                # Less indented - might be end of dict
                                # Include if it's just a closing brace or similar
                                stripped = line.strip()
                                if stripped in ['}', '},', '}']:
                                    code_parts.append(stripped + '\n')
                                # Otherwise stop (end of dict definition)
                                break
                        else:
                            # Empty line - include as-is
                            code_parts.append(line)
                    
                    code = ''.join(code_parts)
                    
                    # Write to temp file and execute
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                        f.write(code)
                        temp_file = f.name
                    
                    try:
                        # Include necessary imports in namespace
                        namespace = {
                            'os': __import__('os'),
                            'pickle': __import__('pickle'),
                            'json': __import__('json'),
                            'datetime': __import__('datetime'),
                        }
                        with open(temp_file, 'r', encoding='utf-8') as f:
                            exec(compile(f.read(), temp_file, 'exec'), namespace)
                        
                        if 'database_estela_dict' in namespace:
                            database_estela_dict = namespace['database_estela_dict']
                            print(f"[OK] Loaded estela dictionary from notebook cell 13 ({len(database_estela_dict)} entries)")
                    except Exception as e:
                        print(f"[WARN] Failed to execute extracted dictionary: {e}")
                        # Continue to file loading
                        pass
                    finally:
                        import os
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
    except Exception as e:
        print(f"[WARN] Failed to load from notebook: {e}")
        # Continue to file loading
        pass
    
    # Fall back to file loading
    if database_estela_dict is None:
        # Try to load from pickle file
        default_file = "notebooks/estela_prompt_data.pkl"
        if os.path.exists(default_file):
            print(f"Loading from pickle file: {default_file}")
            with open(default_file, "rb") as f:
                database_estela_dict = pickle.load(f)
        else:
            # Try environment variable
            prompt_dict_file = os.environ.get("PROMPT_DICT_FILE")
            if prompt_dict_file and os.path.exists(prompt_dict_file):
                print(f"Loading from pickle file (env): {prompt_dict_file}")
                with open(prompt_dict_file, "rb") as f:
                    database_estela_dict = pickle.load(f)
            else:
                prompt_dict_json = os.environ.get("PROMPT_DICT_JSON")
                if prompt_dict_json and os.path.exists(prompt_dict_json):
                    import json
                    print(f"Loading from JSON file (env): {prompt_dict_json}")
                    with open(prompt_dict_json, "r", encoding="utf-8") as f:
                        database_estela_dict = json.load(f)
                else:
                    raise ValueError(
                        f"No estela dictionary found. "
                        f"Tried: notebook cell, {default_file}, PROMPT_DICT_FILE, PROMPT_DICT_JSON"
                    )
    
    # Flatten prompts and extract texts (same as notebook)
    flat_prompts = flatten_prompt_dict(database_estela_dict)
    texts = list(flat_prompts.values())
    
    # Clean texts (remove None, convert to string, remove null bytes)
    cleaned_texts = []
    for text in texts:
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\x00", "").strip()
        if text:  # Only keep non-empty strings
            cleaned_texts.append(text)
    
    return cleaned_texts


def analyze_lengths(texts: List[str]) -> Tuple[Dict[str, float], int]:
    """
    Analyze character lengths for texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        (percentiles_dict, total_count)
    """
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
    
    return percentiles, len(texts)


def main():
    """Main execution function."""
    print("=" * 80)
    print("Estela Prompt Dictionary Character Length Analysis")
    print("=" * 80)
    print()
    
    try:
        # Load estela data
        print("Loading estela dictionary...")
        texts = load_estela_dict()
        print(f"[OK] Loaded {len(texts)} prompts")
        
        # Analyze lengths
        print("\nAnalyzing character lengths...")
        percentiles, count = analyze_lengths(texts)
        
        # Print results
        print()
        print("=" * 80)
        print("Results")
        print("=" * 80)
        print()
        print(f"Total prompts: {count}")
        print()
        print("Character Length Percentiles:")
        print(f"  1st percentile:  {percentiles['1st']:.0f} characters")
        print(f"  5th percentile:  {percentiles['5th']:.0f} characters")
        print(f"  10th percentile: {percentiles['10th']:.0f} characters")
        print(f"  25th percentile: {percentiles['25th']:.0f} characters")
        print(f"  50th percentile: {percentiles['50th']:.0f} characters (median)")
        print(f"  75th percentile: {percentiles['75th']:.0f} characters")
        print(f"  90th percentile: {percentiles['90th']:.0f} characters")
        print(f"  95th percentile: {percentiles['95th']:.0f} characters")
        print(f"  99th percentile: {percentiles['99th']:.0f} characters")
        
        # Print table format
        print()
        print("=" * 80)
        print("Table Format")
        print("=" * 80)
        print()
        header = f"{'Dataset':<25} {'Count':<8} {'1st':<8} {'5th':<8} {'10th':<8} {'25th':<8} {'50th':<8} {'75th':<8} {'90th':<8} {'95th':<8} {'99th':<8}"
        print(header)
        print("-" * len(header))
        row = (
            f"{'estela_prompt_dict':<25} "
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
        
        # Markdown table
        print()
        print("Markdown Table:")
        print()
        print("| Dataset | Count | 1st | 5th | 10th | 25th | 50th | 75th | 90th | 95th | 99th |")
        print("|---------|-------|-----|-----|------|------|------|------|------|------|------|")
        print(
            f"| estela_prompt_dict | {count} | {percentiles['1st']:.0f} | {percentiles['5th']:.0f} | "
            f"{percentiles['10th']:.0f} | {percentiles['25th']:.0f} | {percentiles['50th']:.0f} | "
            f"{percentiles['75th']:.0f} | {percentiles['90th']:.0f} | {percentiles['95th']:.0f} | "
            f"{percentiles['99th']:.0f} |"
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to analyze estela data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
