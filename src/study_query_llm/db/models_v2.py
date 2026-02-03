"""
Database Models V2 - Immutable capture schema for Postgres.

This module defines the v2 schema with immutable raw capture tables,
mutable grouping tables, and support for multimodal/embedding data.

Designed for PostgreSQL with optional pgvector support.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, DateTime, JSON, Float, Text, ForeignKey,
    Index, CheckConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY

BaseV2 = declarative_base()


class RawCall(BaseV2):
    """
    Immutable raw capture table for all LLM/provider calls.
    
    Stores request/response as JSON to support any modality (text, embeddings,
    multimodal, etc.). Status field allows logging both successes and failures.
    
    Attributes:
        id: Primary key
        provider: Provider name (e.g., 'azure_openai_gpt-4')
        model: Model/deployment name
        modality: Type of call ('text', 'embedding', 'multimodal', etc.)
        status: Call status ('success', 'failed', 'timeout', etc.)
        request_json: Full request payload as JSON
        response_json: Full response payload as JSON (null if failed)
        error_json: Error details as JSON (null if success)
        latency_ms: Response latency in milliseconds
        tokens_json: Token usage breakdown as JSON
        metadata_json: Additional metadata as JSON
        created_at: Timestamp when call was made
    """
    __tablename__ = 'raw_calls'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=True, index=True)
    modality = Column(String(50), nullable=False, default='text', index=True)
    status = Column(String(20), nullable=False, default='success', index=True)
    request_json = Column(JSON, nullable=False)
    response_json = Column(JSON, nullable=True)
    error_json = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)
    tokens_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    # Relationships
    group_members = relationship("GroupMember", back_populates="raw_call", cascade="all, delete-orphan")
    artifacts = relationship("CallArtifact", back_populates="raw_call", cascade="all, delete-orphan")
    embedding_vectors = relationship("EmbeddingVector", back_populates="raw_call", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("status IN ('success', 'failed', 'timeout', 'cancelled')", name="check_status"),
        Index('idx_raw_calls_provider_status', 'provider', 'status'),
        Index('idx_raw_calls_created_at', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<RawCall(id={self.id}, provider={self.provider}, "
            f"modality={self.modality}, status={self.status}, "
            f"created_at={self.created_at})>"
        )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'provider': self.provider,
            'model': self.model,
            'modality': self.modality,
            'status': self.status,
            'request_json': self.request_json,
            'response_json': self.response_json,
            'error_json': self.error_json,
            'latency_ms': self.latency_ms,
            'tokens_json': self.tokens_json,
            'metadata_json': self.metadata_json,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Group(BaseV2):
    """
    Mutable grouping table for batches, experiments, labels, etc.
    
    Groups can be created/modified/deleted without affecting immutable RawCall records.
    Multiple groups can reference the same RawCall via GroupMember.
    
    Standard Group Types (see ProvenanceService for conventions):
    - `dataset`: Input data collection (links to embedding RawCalls)
    - `embedding_batch`: Batch of embeddings created together
    - `run`: Complete algorithm execution (e.g., PCA+KLLMeans sweep)
    - `step`: Individual step within a run (e.g., "pca_projection", "clustering_k=5")
    - `metrics`: Computed metrics/analysis results
    - `summarization_batch`: Batch of LLM summarization calls
    - `batch`: Generic batch (legacy, use specific types above when possible)
    - `experiment`: Generic experiment (legacy, use 'run' for algorithm executions)
    - `label`: Generic label (legacy)
    - `custom`: Custom group type
    
    Attributes:
        id: Primary key
        group_type: Type of group (see standard types above)
        name: Group name/identifier
        description: Optional description
        created_at: When group was created
        metadata_json: Additional metadata as JSON (algorithm config, parent_run_id, etc.)
    """
    __tablename__ = 'groups'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    group_type = Column(String(50), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    metadata_json = Column(JSON, nullable=True)
    
    # Relationships
    members = relationship("GroupMember", back_populates="group", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_groups_type_name', 'group_type', 'name'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<Group(id={self.id}, group_type={self.group_type}, "
            f"name={self.name}, created_at={self.created_at})>"
        )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'group_type': self.group_type,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata_json': self.metadata_json,
        }


class GroupMember(BaseV2):
    """
    Join table linking Groups to RawCalls.
    
    Allows many-to-many relationships: one RawCall can belong to multiple groups,
    and one Group can contain multiple RawCalls.
    
    Attributes:
        id: Primary key
        group_id: Foreign key to Group
        call_id: Foreign key to RawCall
        added_at: When this membership was created
        position: Optional ordering within the group
        role: Optional role/label for this member in the group
    """
    __tablename__ = 'group_members'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey('groups.id', ondelete='CASCADE'), nullable=False, index=True)
    call_id = Column(Integer, ForeignKey('raw_calls.id', ondelete='CASCADE'), nullable=False, index=True)
    added_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    position = Column(Integer, nullable=True)
    role = Column(String(100), nullable=True)
    
    # Relationships
    group = relationship("Group", back_populates="members")
    raw_call = relationship("RawCall", back_populates="group_members")
    
    __table_args__ = (
        Index('idx_group_members_group_call', 'group_id', 'call_id', unique=True),
        Index('idx_group_members_call', 'call_id'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<GroupMember(id={self.id}, group_id={self.group_id}, "
            f"call_id={self.call_id}, added_at={self.added_at})>"
        )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'group_id': self.group_id,
            'call_id': self.call_id,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'position': self.position,
            'role': self.role,
        }


class CallArtifact(BaseV2):
    """
    Table for storing references to multimodal artifacts (images, audio, etc.).
    
    Stores URIs/metadata rather than binary data. Binary data should be stored
    in object storage or filesystem, with URIs referenced here.
    
    Attributes:
        id: Primary key
        call_id: Foreign key to RawCall
        artifact_type: Type of artifact ('image', 'audio', 'video', etc.)
        uri: URI/path to the artifact
        content_type: MIME type
        byte_size: Size in bytes
        metadata_json: Additional metadata as JSON
    """
    __tablename__ = 'call_artifacts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey('raw_calls.id', ondelete='CASCADE'), nullable=False, index=True)
    artifact_type = Column(String(50), nullable=False, index=True)
    uri = Column(String(1000), nullable=False)
    content_type = Column(String(100), nullable=True)
    byte_size = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    # Relationships
    raw_call = relationship("RawCall", back_populates="artifacts")
    
    __table_args__ = (
        Index('idx_call_artifacts_call_type', 'call_id', 'artifact_type'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<CallArtifact(id={self.id}, call_id={self.call_id}, "
            f"artifact_type={self.artifact_type}, uri={self.uri})>"
        )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'call_id': self.call_id,
            'artifact_type': self.artifact_type,
            'uri': self.uri,
            'content_type': self.content_type,
            'byte_size': self.byte_size,
            'metadata_json': self.metadata_json,
        }


class GroupLink(BaseV2):
    """
    Table for explicitly modeling relationships between groups.
    
    Supports parent-child relationships, step sequences, dependencies, and generation chains.
    This allows tracking how groups relate to each other (e.g., run steps, data flow).
    
    Link Types:
    - `step`: Step groups within a run (ordered by position)
    - `contains`: One group contains another
    - `depends_on`: One group depends on another (data flow)
    - `generates`: One group generates another (e.g., embedding batch generates run)
    
    Attributes:
        id: Primary key
        parent_group_id: Foreign key to Group (parent group)
        child_group_id: Foreign key to Group (child group)
        link_type: Type of relationship ('step', 'contains', 'depends_on', 'generates')
        position: Optional ordering within parent (for step sequences)
        metadata_json: Additional relationship metadata
        created_at: Timestamp when link was created
    """
    __tablename__ = 'group_links'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_group_id = Column(Integer, ForeignKey('groups.id', ondelete='CASCADE'), nullable=False, index=True)
    child_group_id = Column(Integer, ForeignKey('groups.id', ondelete='CASCADE'), nullable=False, index=True)
    link_type = Column(String(50), nullable=False, index=True)
    position = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    # Relationships
    parent_group = relationship("Group", foreign_keys=[parent_group_id], backref="child_links")
    child_group = relationship("Group", foreign_keys=[child_group_id], backref="parent_links")
    
    __table_args__ = (
        Index('idx_group_links_parent_child', 'parent_group_id', 'child_group_id'),
        Index('idx_group_links_type', 'link_type'),
        Index('idx_group_links_parent_position', 'parent_group_id', 'position'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<GroupLink(id={self.id}, parent={self.parent_group_id}, "
            f"child={self.child_group_id}, type={self.link_type}, "
            f"position={self.position})>"
        )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'parent_group_id': self.parent_group_id,
            'child_group_id': self.child_group_id,
            'link_type': self.link_type,
            'position': self.position,
            'metadata_json': self.metadata_json,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class EmbeddingVector(BaseV2):
    """
    Table for storing embedding vectors.
    
    Supports pgvector if available, otherwise falls back to JSON/ARRAY columns.
    Use pgvector for efficient similarity search when the extension is enabled.
    
    Attributes:
        id: Primary key
        call_id: Foreign key to RawCall
        vector: Embedding vector (pgvector Vector type if available, else JSON/ARRAY)
        dimension: Vector dimensionality
        norm: Optional L2 norm of the vector
        metadata_json: Additional metadata as JSON
    """
    __tablename__ = 'embedding_vectors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey('raw_calls.id', ondelete='CASCADE'), nullable=False, index=True)
    dimension = Column(Integer, nullable=False)
    norm = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    # Relationships
    raw_call = relationship("RawCall", back_populates="embedding_vectors")
    
    # Vector column - will be set dynamically based on pgvector availability
    # If pgvector is available, use Vector type; otherwise use JSON or ARRAY
    vector = Column(JSON, nullable=False)  # Default to JSON, can be overridden
    
    __table_args__ = (
        Index('idx_embedding_vectors_call', 'call_id'),
        Index('idx_embedding_vectors_dimension', 'dimension'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<EmbeddingVector(id={self.id}, call_id={self.call_id}, "
            f"dimension={self.dimension})>"
        )
    
    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'call_id': self.call_id,
            'vector': self.vector,
            'dimension': self.dimension,
            'norm': self.norm,
            'metadata_json': self.metadata_json,
        }
