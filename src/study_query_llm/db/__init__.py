"""
Database Layer - Models, connections, and repositories.

This package provides database persistence for inference runs using SQLAlchemy.
Supports both SQLite (development) and PostgreSQL (production).

V2 schema (Postgres) provides immutable raw capture with grouping support.
"""

from .models import Base, InferenceRun
from .connection import DatabaseConnection
from .inference_repository import InferenceRepository

# V2 schema exports
from .models_v2 import (
    BaseV2,
    RawCall,
    Group,
    GroupMember,
    CallArtifact,
    EmbeddingVector,
)
from .connection_v2 import DatabaseConnectionV2
from .raw_call_repository import RawCallRepository

__all__ = [
    # V1 schema
    "Base",
    "InferenceRun",
    "DatabaseConnection",
    "InferenceRepository",
    # V2 schema
    "BaseV2",
    "RawCall",
    "Group",
    "GroupMember",
    "CallArtifact",
    "EmbeddingVector",
    "DatabaseConnectionV2",
    "RawCallRepository",
]

