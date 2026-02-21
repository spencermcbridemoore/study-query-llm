"""Text data utilities shared across sweep and analysis scripts."""

from typing import Dict, List, Tuple


def is_prompt_key(key: str) -> bool:
    """Check if a dictionary key represents a prompt field."""
    return "prompt" in key.lower()


def flatten_prompt_dict(
    data, path: Tuple[str, ...] = ()
) -> Dict[Tuple[str, ...], str]:
    """Flatten a nested prompt dictionary into a flat map of key-tuples to prompt strings."""
    flat: Dict[Tuple[str, ...], str] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + (key,)
            if isinstance(key, str) and is_prompt_key(key) and isinstance(value, str):
                flat[new_path] = value
            else:
                flat.update(flatten_prompt_dict(value, new_path))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_path = path + (f"[{i}]",)
            flat.update(flatten_prompt_dict(value, new_path))

    return flat


def clean_texts(texts_list: List[str]) -> List[str]:
    """Clean and filter texts: strip nulls, null bytes, and empty strings."""
    cleaned = []
    for text in texts_list:
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\x00", "").strip()
        if text:
            cleaned.append(text)
    return cleaned
