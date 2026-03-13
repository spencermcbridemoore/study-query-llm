"""
ExecutionBackend Protocol -- shared interface for running containerized jobs.

Represents "run a containerized job on some compute and get a result back".
Broader than ModelManager (which is model-lifecycle-specific).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class JobState(Enum):
    """State of a submitted job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceSpec:
    """Resource requirements for a job."""

    cpu: float = 1.0
    memory_gb: float = 4.0
    gpu_count: int = 0
    gpu_type: Optional[str] = None  # e.g. "RTX_4090", "A100"


@dataclass
class JobSpec:
    """Specification for a containerized job."""

    image: str
    command: list[str]
    env: Dict[str, str] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    name: Optional[str] = None
    working_dir: Optional[str] = None


@dataclass
class JobStatus:
    """Status of a submitted job."""

    state: JobState
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Structural interface for execution backends."""

    backend_type: str

    def submit(self, spec: JobSpec) -> str: ...
    def poll(self, job_ref: str) -> JobStatus: ...
    def cancel(self, job_ref: str) -> None: ...
    def logs(self, job_ref: str, tail: int = 100) -> str: ...
