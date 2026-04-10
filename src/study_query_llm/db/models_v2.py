"""
Database Models V2 - Immutable capture schema for Postgres.

This module defines the v2 schema with immutable raw capture tables,
mutable grouping tables, and support for multimodal/embedding data.

Designed for PostgreSQL with optional pgvector support.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, DateTime, JSON, Float, Text, ForeignKey, Boolean,
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    - `clustering_sweep`: Parameter-grid execution of clustering (completed/consumable)
    - `clustering_sweep_request`: Order/request for sweep runs (pending delivery)
    - `metrics`: Computed metrics/analysis results
    - `summarization_batch`: Batch of LLM summarization calls
    - `batch`: Generic batch (legacy, use specific types above when possible)
    - `experiment`: Generic experiment (legacy, use 'run' for algorithm executions)
    - `label`: Generic label (used for defective data exclusion - convention: group name "defective_data")
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def to_dict(self) -> Dict[str, Any]:
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


class SweepRunClaim(BaseV2):
    """
    Worker-claim table for request-driven sweep execution.

    One row represents ownership/processing state for a specific (request, run_key)
    target, enabling safe multi-worker rollout with leases and heartbeats.
    """

    __tablename__ = "sweep_run_claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_key = Column(String(300), nullable=False, index=True)
    claim_status = Column(String(20), nullable=False, default="claimed", index=True)
    claimed_by = Column(String(120), nullable=True)
    claimed_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    lease_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    run_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metadata_json = Column(JSON, nullable=True)

    request_group = relationship("Group", foreign_keys=[request_group_id])
    run_group = relationship("Group", foreign_keys=[run_group_id])

    __table_args__ = (
        CheckConstraint(
            "claim_status IN ('claimed', 'completed', 'failed', 'released')",
            name="check_sweep_run_claim_status",
        ),
        Index(
            "idx_sweep_run_claim_request_status",
            "request_group_id",
            "claim_status",
        ),
        Index(
            "idx_sweep_run_claim_request_run_key",
            "request_group_id",
            "run_key",
            unique=True,
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "request_group_id": self.request_group_id,
            "run_key": self.run_key,
            "claim_status": self.claim_status,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "lease_expires_at": (
                self.lease_expires_at.isoformat() if self.lease_expires_at else None
            ),
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "run_group_id": self.run_group_id,
            "metadata_json": self.metadata_json,
        }


class OrchestrationJob(BaseV2):
    """
    Generic orchestration job for sharded and non-sharded execution.

    Supports durable leasing/retries for any job type:
    - run_k_try (leaf execution)
    - reduce_k (per-K reducer)
    - finalize_run (run-level reducer)
    """

    __tablename__ = "orchestration_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_job_id = Column(
        Integer,
        ForeignKey("orchestration_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    job_type = Column(String(50), nullable=False, index=True)
    job_key = Column(String(500), nullable=False, index=True)
    base_run_key = Column(String(300), nullable=True, index=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    priority = Column(Integer, nullable=False, default=100, index=True)
    payload_json = Column(JSON, nullable=False, default=dict)
    seed_value = Column(Integer, nullable=True)

    claimed_by = Column(String(120), nullable=True)
    claimed_at = Column(DateTime(timezone=True), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)

    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    result_ref = Column(String(200), nullable=True)
    error_json = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    request_group = relationship("Group", foreign_keys=[request_group_id])
    parent_job = relationship("OrchestrationJob", remote_side=[id], uselist=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'ready', 'claimed', 'completed', 'failed', 'cancelled')",
            name="check_orchestration_job_status",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts >= 1",
            name="check_orchestration_job_attempt_bounds",
        ),
        Index("uq_orchestration_jobs_job_key", "job_key", unique=True),
        Index(
            "idx_orchestration_jobs_claim_ready",
            "request_group_id",
            "job_type",
            "status",
            "priority",
            "lease_expires_at",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "request_group_id": self.request_group_id,
            "parent_job_id": self.parent_job_id,
            "job_type": self.job_type,
            "job_key": self.job_key,
            "base_run_key": self.base_run_key,
            "status": self.status,
            "priority": self.priority,
            "payload_json": self.payload_json,
            "seed_value": self.seed_value,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "lease_expires_at": self.lease_expires_at.isoformat() if self.lease_expires_at else None,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "result_ref": self.result_ref,
            "error_json": self.error_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class OrchestrationJobDependency(BaseV2):
    """Many-to-many dependency edges between orchestration jobs."""

    __tablename__ = "orchestration_job_dependencies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        Integer,
        ForeignKey("orchestration_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    depends_on_job_id = Column(
        Integer,
        ForeignKey("orchestration_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        CheckConstraint("job_id != depends_on_job_id", name="check_orchestration_job_dependency_no_self"),
        Index(
            "uq_orchestration_job_dependency_pair",
            "job_id",
            "depends_on_job_id",
            unique=True,
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "depends_on_job_id": self.depends_on_job_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MethodDefinition(BaseV2):
    """
    Table for versioned analysis method definitions.

    Tracks which code/version produced analysis results. Use is_active to
    designate the default version when querying by name only.

    Attributes:
        id: Primary key
        name: Method name (e.g., "extract_correct_answers")
        version: Version string (e.g., "2.1")
        is_active: True for the current default version
        description: Optional description
        code_ref: Path to code (e.g., "scripts/parse_quiz.py")
        code_commit: Git SHA of the code
        input_schema: JSON describing expected input
        output_schema: JSON describing output shape
        parameters_schema: JSON describing configurable knobs
        parent_version_id: FK to previous version (nullable for v1)
        created_at: Timestamp when definition was created
    """
    __tablename__ = "method_definitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    description = Column(Text, nullable=True)
    code_ref = Column(String(500), nullable=True)
    code_commit = Column(String(64), nullable=True)
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    parameters_schema = Column(JSON, nullable=True)
    parent_version_id = Column(
        Integer,
        ForeignKey("method_definitions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    parent_version = relationship("MethodDefinition", remote_side=[id])
    analysis_results = relationship("AnalysisResult", back_populates="method_definition")

    __table_args__ = (
        Index("uq_method_definitions_name_version", "name", "version", unique=True),
        Index("idx_method_definitions_name_active", "name", "is_active"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "is_active": self.is_active,
            "description": self.description,
            "code_ref": self.code_ref,
            "code_commit": self.code_commit,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "parameters_schema": self.parameters_schema,
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AnalysisResult(BaseV2):
    """
    Table for structured analysis results with method provenance.

    Links a numeric/JSON result to the method that produced it and the
    source data group. Use result_value for scalar metrics, result_json
    for structured data.

    Attributes:
        id: Primary key
        method_definition_id: FK to MethodDefinition
        source_group_id: FK to Group (the data analyzed)
        analysis_group_id: FK to Group (optional analysis_run group)
        result_key: Metric name (e.g., "chi_square", "ari")
        result_value: Numeric scalar (nullable)
        result_json: Structured data (nullable)
        created_at: Timestamp when result was recorded
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    method_definition_id = Column(
        Integer,
        ForeignKey("method_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    analysis_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    result_key = Column(String(200), nullable=False)
    result_value = Column(Float, nullable=True)
    result_json = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    method_definition = relationship("MethodDefinition", back_populates="analysis_results")

    __table_args__ = (
        Index("idx_analysis_results_method_source", "method_definition_id", "source_group_id"),
        Index("idx_analysis_results_source_key", "source_group_id", "result_key"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "method_definition_id": self.method_definition_id,
            "source_group_id": self.source_group_id,
            "analysis_group_id": self.analysis_group_id,
            "result_key": self.result_key,
            "result_value": self.result_value,
            "result_json": self.result_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ProvenancedRun(BaseV2):
    """
    First-class execution record for all run/analysis outcomes.

    Canonical write path uses ``run_kind='execution'`` and stores the optional
    semantic role in ``metadata_json.execution_role``. Legacy values remain
    temporarily accepted for compatibility/backfill windows.
    """

    __tablename__ = "provenanced_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_kind = Column(String(40), nullable=False, index=True)
    run_status = Column(String(20), nullable=False, default="created", index=True)

    request_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    result_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    input_snapshot_group_id = Column(
        Integer,
        ForeignKey("groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    method_definition_id = Column(
        Integer,
        ForeignKey("method_definitions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    orchestration_job_id = Column(
        Integer,
        ForeignKey("orchestration_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    parent_provenanced_run_id = Column(
        Integer,
        ForeignKey("provenanced_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    run_key = Column(String(300), nullable=True, index=True)
    determinism_class = Column(
        String(40),
        nullable=False,
        default="non_deterministic",
        index=True,
    )
    config_hash = Column(String(64), nullable=True, index=True)
    config_json = Column(JSON, nullable=True)
    result_ref = Column(String(400), nullable=True)
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        CheckConstraint(
            "run_kind IN ('execution', 'method_execution', 'analysis_execution')",
            name="check_provenanced_run_kind",
        ),
        CheckConstraint(
            "run_status IN ('created', 'running', 'completed', 'failed', 'cancelled')",
            name="check_provenanced_run_status",
        ),
        CheckConstraint(
            "determinism_class IN ('deterministic', 'pseudo_deterministic', 'non_deterministic')",
            name="check_provenanced_run_determinism_class",
        ),
        Index(
            "idx_provenanced_run_request_key_kind",
            "request_group_id",
            "run_key",
            "run_kind",
            unique=True,
        ),
        Index(
            "idx_provenanced_run_source_kind",
            "source_group_id",
            "run_kind",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_kind": self.run_kind,
            "run_status": self.run_status,
            "request_group_id": self.request_group_id,
            "source_group_id": self.source_group_id,
            "result_group_id": self.result_group_id,
            "input_snapshot_group_id": self.input_snapshot_group_id,
            "method_definition_id": self.method_definition_id,
            "orchestration_job_id": self.orchestration_job_id,
            "parent_provenanced_run_id": self.parent_provenanced_run_id,
            "run_key": self.run_key,
            "determinism_class": self.determinism_class,
            "config_hash": self.config_hash,
            "config_json": self.config_json,
            "result_ref": self.result_ref,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'call_id': self.call_id,
            'vector': self.vector,
            'dimension': self.dimension,
            'norm': self.norm,
            'metadata_json': self.metadata_json,
        }


class EmbeddingCacheEntry(BaseV2):
    """Canonical L2 cache entry for deterministic embedding requests."""

    __tablename__ = "embedding_cache_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(64), nullable=False, index=True)
    key_version = Column(String(20), nullable=False, default="raw_v1", index=True)
    provider = Column(String(100), nullable=False, index=True)
    deployment = Column(String(100), nullable=False, index=True)
    dimensions = Column(Integer, nullable=True)
    encoding_format = Column(String(20), nullable=False, default="float")
    input_text_raw = Column(Text, nullable=False)
    input_text_sha256_raw = Column(String(64), nullable=False, index=True)
    vector = Column(JSON, nullable=False)
    dimension = Column(Integer, nullable=False)
    norm = Column(Float, nullable=True)
    source_raw_call_id = Column(
        Integer, ForeignKey("raw_calls.id", ondelete="SET NULL"), nullable=True, index=True
    )
    hit_count = Column(Integer, nullable=False, default=0)
    last_hit_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("uq_embedding_cache_key", "cache_key", unique=True),
        Index(
            "idx_embedding_cache_lookup",
            "provider",
            "deployment",
            "key_version",
            "input_text_sha256_raw",
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cache_key": self.cache_key,
            "key_version": self.key_version,
            "provider": self.provider,
            "deployment": self.deployment,
            "dimensions": self.dimensions,
            "encoding_format": self.encoding_format,
            "input_text_sha256_raw": self.input_text_sha256_raw,
            "dimension": self.dimension,
            "source_raw_call_id": self.source_raw_call_id,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at.isoformat() if self.last_hit_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class EmbeddingCacheLease(BaseV2):
    """Short-lived lease row used for cross-worker single-flight coordination."""

    __tablename__ = "embedding_cache_leases"

    cache_key = Column(String(64), primary_key=True)
    lease_owner = Column(String(120), nullable=False, index=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("idx_embedding_cache_leases_owner", "lease_owner"),
        Index("idx_embedding_cache_leases_expiry", "lease_expires_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at.isoformat() if self.lease_expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
