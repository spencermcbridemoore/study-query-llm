"""Centralized Estela prompt dictionary loading."""

import json
import os
import pickle
from typing import Any, Dict, Optional


def load_estela_dict(
    pkl_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load an Estela prompt dictionary from pickle or JSON.

    Resolution order:
    1. Explicit ``pkl_path`` / ``json_path`` arguments
    2. ``PROMPT_DICT_FILE`` environment variable (pickle)
    3. ``PROMPT_DICT_JSON`` environment variable (JSON)

    Returns an empty dict and prints an error message if no source is found.
    """
    pkl_path = pkl_path or os.environ.get("PROMPT_DICT_FILE")
    json_path = json_path or os.environ.get("PROMPT_DICT_JSON")

    if pkl_path and os.path.exists(pkl_path):
        print(f"   Loading from pickle file: {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if json_path and os.path.exists(json_path):
        print(f"   Loading from JSON file: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n[ERROR] No prompt dictionary source found.")
    print("   Set PROMPT_DICT_FILE (pickle) or PROMPT_DICT_JSON (JSON),")
    print("   or pass pkl_path / json_path explicitly.")
    return {}
