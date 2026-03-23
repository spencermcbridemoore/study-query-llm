"""Dataclasses for embedding requests and responses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingRequest:
    """Request parameters for embedding generation."""

    text: str
    deployment: str
    provider: str = "azure"
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    group_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    vector: List[float]
    model: str
    dimension: int
    request_hash: str
    cached: bool
    raw_call_id: Optional[int] = None
    latency_ms: Optional[float] = None
