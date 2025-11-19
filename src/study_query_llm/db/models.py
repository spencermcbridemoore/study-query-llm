"""
Database Models - SQLAlchemy ORM models.

Defines the database schema for storing LLM inference runs.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class InferenceRun(Base):
    """
    Model for storing LLM inference runs.
    
    This table stores all inference requests and responses, including
    metadata like tokens, latency, and provider information.
    
    Attributes:
        id: Primary key, auto-incrementing integer
        prompt: The input prompt sent to the LLM
        response: The response text from the LLM
        provider: Name of the LLM provider (e.g., 'azure_openai_gpt-4')
        tokens: Total tokens used (prompt + completion)
        latency_ms: Response latency in milliseconds
        metadata: JSON field for provider-specific metadata
        created_at: Timestamp when the inference was run
    """
    __tablename__ = 'inference_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    provider = Column(String(50), nullable=False, index=True)
    tokens = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<InferenceRun(id={self.id}, provider={self.provider}, "
            f"created_at={self.created_at})>"
        )

    def to_dict(self) -> dict:
        """
        Convert model instance to dictionary.
        
        Useful for serialization and API responses.
        
        Returns:
            Dictionary representation of the inference run
        """
        return {
            'id': self.id,
            'prompt': self.prompt,
            'response': self.response,
            'provider': self.provider,
            'tokens': self.tokens,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

