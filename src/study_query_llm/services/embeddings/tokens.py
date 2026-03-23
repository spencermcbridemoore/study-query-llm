"""Token estimation for embedding inputs."""

from typing import Optional


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses tiktoken if available (most accurate), otherwise falls back to approximation.
    """
    try:
        import tiktoken

        if model and model.startswith("text-embedding-3"):
            encoding_name = "cl100k_base"
        elif model and "ada-002" in model:
            encoding_name = "cl100k_base"
        else:
            encoding_name = "cl100k_base"

        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    except ImportError:
        return len(text) // 4
