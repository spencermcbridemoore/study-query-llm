"""
Execution backends for running containerized jobs on local Docker, SSH hosts, or cloud providers.

Provides the ExecutionBackend protocol and factory for creating backend instances.
"""

from .protocol import (
    ExecutionBackend,
    JobSpec,
    JobState,
    JobStatus,
    ResourceSpec,
)
from .factory import ExecutionBackendFactory

__all__ = [
    "ExecutionBackend",
    "ExecutionBackendFactory",
    "JobSpec",
    "JobState",
    "JobStatus",
    "ResourceSpec",
]
