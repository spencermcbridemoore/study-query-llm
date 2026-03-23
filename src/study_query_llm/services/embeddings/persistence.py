"""DB persistence and failure logging for embedding calls."""

from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from ...utils.logging_config import get_logger
from .._shared import handle_db_persistence_error
from .constants import CACHE_KEY_VERSION
from .hashing import compute_raw_text_sha256
from .models import EmbeddingRequest

if TYPE_CHECKING:
    from ...db.raw_call_repository import RawCallRepository
    from ...providers.base_embedding import EmbeddingResult

logger = get_logger(__name__)

L1PutFn = Optional[Callable[[str, List[float], Optional[int]], None]]


def log_embedding_failure(
    repository: Optional["RawCallRepository"],
    request: EmbeddingRequest,
    request_hash: str,
    error: Exception,
    latency_ms: Optional[float] = None,
) -> Optional[int]:
    """Log failed embedding request to database."""
    if not repository:
        return None

    try:
        error_json = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        metadata_json = {
            "request_hash": request_hash,
        }
        if request.group_id:
            metadata_json["group_id"] = request.group_id
        metadata_json.update(request.metadata)

        call_id = repository.insert_raw_call(
            provider=f"{request.provider}_openai_{request.deployment}",
            request_json={"input": request.text, "model": request.deployment},
            model=request.deployment,
            modality="embedding",
            status="failed",
            response_json=None,
            error_json=error_json,
            latency_ms=latency_ms,
            tokens_json=None,
            metadata_json=metadata_json,
        )

        logger.warning(
            "Logged failed embedding: id=%s, deployment=%s, error=%s",
            call_id,
            request.deployment,
            str(error),
        )

        return call_id

    except Exception as db_error:
        logger.error("Failed to log embedding failure: %s", str(db_error), exc_info=True)
        return None


def persist_embedding(
    repository: Optional["RawCallRepository"],
    require_db_persistence: bool,
    request: EmbeddingRequest,
    vector: List[float],
    dimension: int,
    deployment: str,
    latency_ms: float,
    request_hash: str,
    *,
    l1_put: L1PutFn = None,
) -> Optional[int]:
    """Persist single embedding to RawCall, EmbeddingVector, and cache table."""
    if not repository:
        return None

    try:
        from ...db.models_v2 import EmbeddingVector

        request_json = {"input": request.text, "model": deployment}
        if request.dimensions:
            request_json["dimensions"] = request.dimensions

        response_json = {
            "model": deployment,
            "embedding_dim": dimension,
        }

        metadata_json = {
            "request_hash": request_hash,
        }
        if request.group_id:
            metadata_json["group_id"] = request.group_id
        metadata_json.update(request.metadata)

        raw_call_id = repository.insert_raw_call(
            provider=f"{request.provider}_openai_{deployment}",
            request_json=request_json,
            model=deployment,
            modality="embedding",
            status="success",
            response_json=response_json,
            latency_ms=latency_ms,
            tokens_json=None,
            metadata_json=metadata_json,
        )

        vector_norm = float(np.linalg.norm(vector))

        embedding_vector = EmbeddingVector(
            call_id=raw_call_id,
            vector=vector,
            dimension=dimension,
            norm=vector_norm,
            metadata_json={"model": deployment},
        )

        repository.session.add(embedding_vector)
        repository.session.flush()
        repository.upsert_embedding_cache_entry(
            cache_key=request_hash,
            key_version=CACHE_KEY_VERSION,
            provider=request.provider,
            deployment=deployment,
            dimensions=request.dimensions,
            encoding_format=request.encoding_format,
            input_text_raw=request.text,
            input_text_sha256_raw=compute_raw_text_sha256(request.text),
            vector=vector,
            dimension=dimension,
            norm=vector_norm,
            source_raw_call_id=raw_call_id,
        )
        if l1_put is not None:
            l1_put(request_hash, list(vector), raw_call_id)

        logger.info(
            "Stored embedding: id=%s, deployment=%s, dimension=%s",
            raw_call_id,
            deployment,
            dimension,
        )

        return raw_call_id

    except Exception as db_error:
        handle_db_persistence_error(
            logger, db_error, require_db_persistence, "persist embedding"
        )
        return None


def persist_embedding_batch(
    repository: Optional["RawCallRepository"],
    requests: List[EmbeddingRequest],
    embeddings: List["EmbeddingResult"],
    request_hashes: List[str],
    latency_ms: float,
    *,
    l1_put: L1PutFn = None,
) -> List[int]:
    """Persist batch of embeddings; returns raw_call_ids (0 on per-row failure)."""
    from ...db.models_v2 import EmbeddingVector

    if not repository:
        return [0] * len(requests)

    raw_call_ids: List[int] = []
    for req, emb_obj, req_hash in zip(requests, embeddings, request_hashes):
        vector = emb_obj.vector
        dimension = len(vector)
        vector_norm = float(np.linalg.norm(vector))

        request_json = {"input": req.text, "model": req.deployment}
        if req.dimensions:
            request_json["dimensions"] = req.dimensions

        response_json = {
            "model": req.deployment,
            "embedding_dim": dimension,
        }

        metadata_json = {"request_hash": req_hash}
        if req.group_id:
            metadata_json["group_id"] = req.group_id
        metadata_json.update(req.metadata)

        try:
            raw_call_id = repository.insert_raw_call(
                provider=f"{req.provider}_openai_{req.deployment}",
                request_json=request_json,
                model=req.deployment,
                modality="embedding",
                status="success",
                response_json=response_json,
                latency_ms=latency_ms,
                tokens_json=None,
                metadata_json=metadata_json,
            )

            ev = EmbeddingVector(
                call_id=raw_call_id,
                vector=vector,
                dimension=dimension,
                norm=vector_norm,
                metadata_json={"model": req.deployment},
            )
            repository.session.add(ev)
            repository.session.flush()
            repository.upsert_embedding_cache_entry(
                cache_key=req_hash,
                key_version=CACHE_KEY_VERSION,
                provider=req.provider,
                deployment=req.deployment,
                dimensions=req.dimensions,
                encoding_format=req.encoding_format,
                input_text_raw=req.text,
                input_text_sha256_raw=compute_raw_text_sha256(req.text),
                vector=vector,
                dimension=dimension,
                norm=vector_norm,
                source_raw_call_id=raw_call_id,
            )
            if l1_put is not None:
                l1_put(req_hash, list(vector), raw_call_id)
            raw_call_ids.append(raw_call_id)

        except Exception as e:
            logger.warning("Failed to persist embedding for hash %s: %s", req_hash, e)
            raw_call_ids.append(0)

    return raw_call_ids
