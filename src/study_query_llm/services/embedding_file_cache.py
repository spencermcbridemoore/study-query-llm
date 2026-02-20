"""
File-based embedding cache: load/save (texts, embeddings, labels) to disk.

Cache is optional and best-effort. Missing or invalid cache never raises;
callers fall back to the embedding API.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Callable, Any, Awaitable
import hashlib
import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def _deployment_safe(deployment: str) -> str:
    """Make deployment string safe for filenames."""
    return deployment.replace("-", "_").replace("/", "_")


def get_cache_path(
    cache_dir: Path, dataset_name: str, deployment: str, seed: int, n_samples: int
) -> Path:
    """Build cache file path from (dataset_name, deployment, seed, n_samples)."""
    safe = _deployment_safe(deployment)
    return cache_dir / f"{dataset_name}_30k_seed{seed}_{safe}.npz"


def _cache_path(cache_dir: Path, dataset_name: str, deployment: str, seed: int, n_samples: int) -> Path:
    return get_cache_path(cache_dir, dataset_name, deployment, seed, n_samples)


def _text_list_signature(texts: List[str]) -> Tuple[int, str, str]:
    """(length, hash of first, hash of last) for fast match without full scan."""
    if not texts:
        return 0, "", ""
    first = hashlib.sha256(texts[0].encode("utf-8")).hexdigest()
    last = hashlib.sha256(texts[-1].encode("utf-8")).hexdigest()
    return len(texts), first, last


def load_embedding_cache(cache_path: Path) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
    """
    Load (texts, embeddings, labels) from a .npz cache file.

    Returns None if the file is missing, unreadable, or invalid (wrong keys/shape).
    Does not raise for file-not-found or invalid cache.
    """
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            if "texts" not in data or "embeddings" not in data or "labels" not in data:
                logger.debug("Embedding cache missing required keys: %s", list(data.keys()))
                return None
            texts = data["texts"]
            embeddings = data["embeddings"]
            labels = data["labels"]
            # npz may store 0-dim array for scalar; ensure 1D for texts/labels
            if texts.ndim == 0:
                texts = np.atleast_1d(texts)
            texts_list = texts.tolist()
            if labels.ndim == 0:
                labels = np.atleast_1d(labels)
            n = len(texts_list)
            if embeddings.shape[0] != n or labels.shape[0] != n:
                logger.debug(
                    "Embedding cache shape mismatch: texts=%s embeddings=%s labels=%s",
                    n,
                    embeddings.shape,
                    labels.shape,
                )
                return None
            return texts_list, np.asarray(embeddings, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    except Exception as e:
        logger.debug("Embedding cache load failed for %s: %s", cache_path, e)
        return None


def save_embedding_cache(
    cache_path: Path,
    texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    dataset: Optional[str] = None,
    deployment: Optional[str] = None,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
) -> None:
    """
    Save (texts, embeddings, labels) to a .npz file. Creates parent directory if needed.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.asarray(labels, dtype=np.int64)
    data = {
        "texts": texts_arr,
        "embeddings": np.asarray(embeddings, dtype=np.float64),
        "labels": labels_arr,
    }
    if dataset is not None:
        data["dataset"] = np.array(dataset)
    if deployment is not None:
        data["deployment"] = np.array(deployment)
    if seed is not None:
        data["seed"] = np.array(seed)
    if n_samples is not None:
        data["n_samples"] = np.array(n_samples)
    np.savez_compressed(cache_path, **data)


async def get_embeddings_with_file_cache(
    texts: List[str],
    deployment: str,
    db: Any,
    cache_dir: Optional[Path],
    dataset_name: Optional[str],
    seed: Optional[int],
    n_samples: Optional[int],
    fetch_embeddings_async: Callable[[List[str], str, Any], Awaitable[np.ndarray]],
) -> np.ndarray:
    """
    Return embeddings for `texts`, using file cache when available and matching.

    If cache_dir is None/empty, or dataset_name/seed/n_samples not provided, or
    cache file is missing/invalid, or cached texts do not match requested texts,
    falls back to fetch_embeddings_async(texts, deployment, db). Never raises
    due to missing or invalid cache.
    """
    if not cache_dir or not dataset_name or seed is None or n_samples is None:
        return await fetch_embeddings_async(texts, deployment, db)

    path = _cache_path(Path(cache_dir), dataset_name, deployment, seed, n_samples)
    if not path.exists():
        return await fetch_embeddings_async(texts, deployment, db)

    loaded = load_embedding_cache(path)
    if loaded is None:
        return await fetch_embeddings_async(texts, deployment, db)

    cached_texts, cached_embeddings, _ = loaded
    if len(cached_texts) != len(texts):
        return await fetch_embeddings_async(texts, deployment, db)
    sig = _text_list_signature(texts)
    cached_sig = _text_list_signature(cached_texts)
    if sig != cached_sig:
        return await fetch_embeddings_async(texts, deployment, db)

    return cached_embeddings
