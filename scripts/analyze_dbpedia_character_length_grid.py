#!/usr/bin/env python3
"""
DBPedia character and token length percentile grids.

Loads DBPedia (same filter: 10 < len <= 1000), computes character lengths,
and prints:
  1) Character grid: rows = length categories, columns = percentiles (min .. max).
  2) Token grid: same structure, using stratified sampling at various string lengths
     + tiktoken (cl100k_base) to estimate token percentiles per band and overall.
"""

import argparse
import json
import os
import pickle
import random
import sys
from typing import List, Optional, Tuple

import numpy as np

# Add path so we can import from run_experimental_sweep and study_query_llm
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from scripts.run_experimental_sweep import load_dbpedia_full, load_yahoo_answers_full
from study_query_llm.services.embedding_service import estimate_tokens

from study_query_llm.utils.text_utils import flatten_prompt_dict as _flatten_prompt_dict, clean_texts as _clean_texts

PCT_LEVELS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
PCT_NAMES = ["min", "1st", "5th", "10th", "25th", "50th", "75th", "90th", "95th", "99th", "max"]

BANDS = [
    ("All lengths", None, None),
    ("11-100", 11, 100),
    ("100-200", 100, 200),
    ("200-300", 200, 300),
    ("300-400", 300, 400),
    ("400-500", 400, 500),
    ("500-600", 500, 600),
    ("600-700", 600, 700),
    ("700-800", 700, 800),
    ("800-900", 800, 900),
    ("900-1000", 900, 1000),
]


def load_estela_dict() -> Tuple[str, List[str]]:
    """Load estela prompt dictionary. Returns (dataset_name, texts)."""
    default_file = os.path.join(_REPO_ROOT, "notebooks", "estela_prompt_data.pkl")
    if os.path.exists(default_file):
        with open(default_file, "rb") as f:
            data = pickle.load(f)
    else:
        prompt_dict_file = os.environ.get("PROMPT_DICT_FILE")
        if prompt_dict_file and os.path.exists(prompt_dict_file):
            with open(prompt_dict_file, "rb") as f:
                data = pickle.load(f)
        else:
            prompt_dict_json = os.environ.get("PROMPT_DICT_JSON")
            if prompt_dict_json and os.path.exists(prompt_dict_json):
                with open(prompt_dict_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise FileNotFoundError(
                    "Estela file not found. Expected notebooks/estela_prompt_data.pkl or PROMPT_DICT_FILE/PROMPT_DICT_JSON"
                )
    flat = _flatten_prompt_dict(data)
    texts = _clean_texts(list(flat.values()))
    return "estela", texts


def all_lengths_row_char(texts: List[str], name: str) -> List:
    """Single 'All lengths' row: [name, N, min, 1st, ..., max] for character length."""
    lengths = np.array([len(t) for t in texts], dtype=np.float64)
    if len(lengths) == 0:
        return [name, 0] + [np.nan] * len(PCT_LEVELS)
    pct_vals = [float(np.percentile(lengths, p)) for p in PCT_LEVELS]
    return [name, len(texts)] + pct_vals


def all_lengths_row_token(texts: List[str], name: str, max_sample: int = 5000, random_state: int = 42) -> List:
    """Single 'All lengths' row for token length; samples if len(texts) > max_sample."""
    rng = random.Random(random_state)
    if len(texts) > max_sample:
        indices = rng.sample(range(len(texts)), max_sample)
        subset = [texts[i] for i in indices]
        n = len(texts)
    else:
        subset = texts
        n = len(texts)
    token_lengths = []
    for t in subset:
        try:
            token_lengths.append(estimate_tokens(t))
        except Exception:
            token_lengths.append(len(t) // 4)
    if not token_lengths:
        return [name, n] + [np.nan] * len(PCT_LEVELS)
    arr = np.array(token_lengths, dtype=np.float64)
    pct_vals = [float(np.percentile(arr, p)) for p in PCT_LEVELS]
    return [name, n] + pct_vals


def _band_mask(lengths: np.ndarray, low: Optional[int], high: Optional[int]) -> np.ndarray:
    if low is None:
        return np.ones(len(lengths), dtype=bool)
    if high == 1000:
        return (lengths >= low) & (lengths <= high)
    return (lengths >= low) & (lengths < high)


def _print_grid(title: str, rows: list, col_headers: list, col_widths: list) -> None:
    print(title)
    print("=" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    header_str = "".join(h.ljust(col_widths[i]) for i, h in enumerate(col_headers))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        cells = [str(row[0])[:18].ljust(18), str(row[1]).rjust(6)]
        for i, v in enumerate(row[2:], start=2):
            if np.isnan(v):
                cells.append("—".rjust(col_widths[i]))
            else:
                cells.append(f"{v:.0f}".rjust(col_widths[i]))
        print("".join(cells))
    print()

    md_header = "| " + " | ".join(col_headers) + " |"
    md_sep = "|" + "|".join(["---"] * len(col_headers)) + "|"
    print("Markdown:")
    print(md_header)
    print(md_sep)
    for row in rows:
        cells = [str(row[0]), str(int(row[1]))]
        for v in row[2:]:
            cells.append("—" if np.isnan(v) else f"{v:.0f}")
        print("| " + " | ".join(cells) + " |")
    print()


def run_character_grid(texts: list, lengths: np.ndarray) -> list:
    pct_levels = PCT_LEVELS
    rows = []
    for label, low, high in BANDS:
        mask = _band_mask(lengths, low, high)
        subset = lengths[mask]
        if len(subset) == 0:
            row = [label, 0] + [np.nan] * len(pct_levels)
            rows.append(row)
            continue
        pct_vals = [float(np.percentile(subset, p)) for p in pct_levels]
        rows.append([label, len(subset)] + pct_vals)
    return rows


def run_token_grid(texts: list, lengths: np.ndarray, sample_per_band: int, random_state: int) -> list:
    """Stratified sample by char-length band, tokenize, then compute token percentiles per band and overall."""
    rng = random.Random(random_state)
    indices = np.arange(len(texts), dtype=np.intp)
    pct_levels = PCT_LEVELS

    # Collect (band_index, list of token lengths, band_count) for bands 1..10 (skip "All" for now)
    band_token_lists = []
    band_counts = []

    for idx, (label, low, high) in enumerate(BANDS):
        if low is None:
            continue
        mask = _band_mask(lengths, low, high)
        count = int(np.sum(mask))
        band_counts.append(count)
        band_inds = indices[mask]
        n_sample = min(sample_per_band, len(band_inds))
        if n_sample == 0:
            band_token_lists.append([])
            continue
        chosen = rng.sample(list(band_inds), n_sample)
        token_lengths = []
        for i in chosen:
            try:
                token_lengths.append(estimate_tokens(texts[i]))
            except Exception:
                token_lengths.append(len(texts[i]) // 4)
        band_token_lists.append(token_lengths)

    # Per-band token percentiles
    rows = []
    for idx, (label, low, high) in enumerate(BANDS):
        if low is None:
            # "All lengths": weighted combination of band samples
            expanded = []
            for b_idx, toks in enumerate(band_token_lists):
                n_band = band_counts[b_idx]
                if not toks:
                    continue
                repeat = max(1, round(n_band / len(toks)))
                for t in toks:
                    expanded.extend([t] * repeat)
            if not expanded:
                row = ["All lengths", 0] + [np.nan] * len(pct_levels)
            else:
                arr = np.array(expanded, dtype=np.float64)
                pct_vals = [float(np.percentile(arr, p)) for p in pct_levels]
                row = ["All lengths", sum(band_counts)] + pct_vals
            rows.append(row)
            continue
        b_idx = idx - 1
        toks = band_token_lists[b_idx]
        count = band_counts[b_idx]
        if not toks:
            row = [label, 0] + [np.nan] * len(pct_levels)
        else:
            arr = np.array(toks, dtype=np.float64)
            pct_vals = [float(np.percentile(arr, p)) for p in pct_levels]
            row = [label, count] + pct_vals
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="DBPedia character and token length percentile grids.")
    parser.add_argument("--chars-only", action="store_true", help="Only print character-length grid")
    parser.add_argument("--sample", type=int, default=800, help="Max documents to sample per char-length band for token grid (default: 800)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    print("Loading DBPedia (10 < length <= 1000 chars)...")
    texts, labels, category_names = load_dbpedia_full(random_state=args.seed)
    lengths = np.array([len(t) for t in texts], dtype=np.float64)
    n = len(lengths)
    print(f"Loaded {n} documents. 14 categories: {category_names}\n")

    col_headers = ["Length category", "N"] + PCT_NAMES
    col_widths = [18, 6] + [8] * len(PCT_NAMES)

    # 1) Character grid
    char_rows = run_character_grid(texts, lengths)
    _print_grid("Character length percentiles by length category (DBPedia)", char_rows, col_headers, col_widths)
    all_row = next(r for r in char_rows if r[0] == "All lengths")
    print("Character summary: N =", int(all_row[1]), "| min =", int(all_row[2]),
          "| 50th =", int(all_row[7]), "| max =", int(all_row[-1]))
    print()

    if not args.chars_only:
        # 2) Token grid (stratified sampling at various string lengths, then tiktoken)
        print("Token grid: stratified sampling", args.sample, "docs per char-length band, tiktoken (cl100k_base)...")
        token_rows = run_token_grid(texts, lengths, sample_per_band=args.sample, random_state=args.seed)
        _print_grid("Token length percentiles by character-length category (DBPedia, estimated from sample)", token_rows, col_headers, col_widths)
        all_token = next(r for r in token_rows if r[0] == "All lengths")
        print("Token summary (weighted): N =", int(all_token[1]), "| min =", int(all_token[2]),
              "| 50th =", int(all_token[7]), "| max =", int(all_token[-1]))

    # Estela and Yahoo: "All lengths" row only (two rows each for char and token)
    print()
    print("=" * 80)
    print("Estela and Yahoo — All lengths row (character and token)")
    print("=" * 80)
    col_headers = ["Dataset", "N"] + PCT_NAMES
    col_widths = [22, 8] + [8] * len(PCT_NAMES)
    extra_rows_char = []
    extra_rows_tok = []
    # Yahoo
    try:
        y_texts, _, _ = load_yahoo_answers_full(random_state=args.seed)
        extra_rows_char.append(all_lengths_row_char(y_texts, "yahoo"))
        extra_rows_tok.append(all_lengths_row_token(y_texts, "yahoo", max_sample=5000, random_state=args.seed))
    except Exception as e:
        print(f"Yahoo: load failed — {e}")
    # Estela
    try:
        estela_name, estela_texts = load_estela_dict()
        extra_rows_char.append(all_lengths_row_char(estela_texts, estela_name))
        extra_rows_tok.append(all_lengths_row_token(estela_texts, estela_name, max_sample=5000, random_state=args.seed))
    except FileNotFoundError as e:
        print(f"Estela: skipped ({e})")
    except Exception as e:
        print(f"Estela: load failed — {e}")

    if extra_rows_char:
        _print_grid("All lengths (character) — estela & yahoo", extra_rows_char, col_headers, col_widths)
    if extra_rows_tok:
        _print_grid("All lengths (token) — estela & yahoo", extra_rows_tok, col_headers, col_widths)


if __name__ == "__main__":
    main()
