"""
Inference Repository - Database operations for inference runs.

This repository handles all database interactions for storing and querying
LLM inference runs. Uses the Repository pattern to abstract database details.
"""

from datetime import datetime
from typing import Optional, Tuple, List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from .models import InferenceRun
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InferenceRepository:
    """
    Repository for all database operations on inference runs.
    
    This class handles both writes (logging inferences) and queries (analytics).
    All database operations go through this repository, keeping the service
    layer database-agnostic.
    
    Usage:
        with db.session_scope() as session:
            repo = InferenceRepository(session)
            
            # Write
            inference_id = repo.insert_inference_run(...)
            
            # Query
            runs = repo.query_inferences(provider="azure")
    """

    def __init__(self, session: Session):
        """
        Initialize repository with a database session.
        
        Args:
            session: SQLAlchemy Session instance
        """
        self.session = session

    # ========== WRITE OPERATIONS ==========

    def insert_inference_run(
        self,
        prompt: str,
        response: str,
        provider: str,
        tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
        batch_id: Optional[str] = None
    ) -> int:
        """
        Insert a single inference run.
        
        Args:
            prompt: The input prompt sent to the LLM
            response: The response text from the LLM
            provider: Name of the LLM provider
            tokens: Total tokens used (optional)
            latency_ms: Response latency in milliseconds (optional)
            metadata: Provider-specific metadata as dict (optional)
            batch_id: UUID string identifying the batch this run belongs to (optional)
        
        Returns:
            The ID of the inserted record
        """
        inference = InferenceRun(
            prompt=prompt,
            response=response,
            provider=provider,
            tokens=tokens,
            latency_ms=latency_ms,
            metadata_json=metadata or {},
            batch_id=batch_id
        )

        self.session.add(inference)
        self.session.flush()  # Flush to get the ID without committing
        self.session.refresh(inference)
        
        logger.debug(
            f"Inserted inference run: id={inference.id}, provider={provider}, "
            f"tokens={tokens}, batch_id={batch_id}"
        )

        return inference.id

    def batch_insert_inferences(self, inferences: List[dict]) -> List[int]:
        """
        Batch insert multiple inference runs.
        
        More efficient than multiple single inserts for bulk operations.
        
        Args:
            inferences: List of dicts with inference data. Each dict should have:
                - prompt (str)
                - response (str)
                - provider (str)
                - tokens (int, optional)
                - latency_ms (float, optional)
                - metadata (dict, optional)
        
        Returns:
            List of inserted IDs in the same order as input
        """
        inference_objects = [
            InferenceRun(**inf) for inf in inferences
        ]

        self.session.add_all(inference_objects)
        self.session.flush()  # Flush to get IDs without committing
        
        return [inf.id for inf in inference_objects]

    # ========== QUERY OPERATIONS ==========

    def get_inference_by_id(self, inference_id: int) -> Optional[InferenceRun]:
        """
        Retrieve a specific inference run by ID.
        
        Args:
            inference_id: The ID of the inference run
        
        Returns:
            InferenceRun object if found, None otherwise
        """
        return self.session.query(InferenceRun).filter_by(id=inference_id).first()

    def query_inferences(
        self,
        provider: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[InferenceRun]:
        """
        Query inferences with filters.
        
        Args:
            provider: Filter by provider name (optional)
            date_range: Tuple of (start_date, end_date) for filtering (optional)
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip for pagination (default: 0)
        
        Returns:
            List of InferenceRun objects, ordered by created_at descending
        """
        query = self.session.query(InferenceRun)

        if provider:
            query = query.filter(InferenceRun.provider == provider)

        if date_range:
            start_date, end_date = date_range
            query = query.filter(InferenceRun.created_at.between(start_date, end_date))

        query = query.order_by(desc(InferenceRun.created_at))
        query = query.limit(limit).offset(offset)

        return query.all()

    def get_provider_stats(self) -> List[dict]:
        """
        Get aggregate statistics by provider.
        
        Calculates count, average tokens, average latency, and total tokens
        for each provider.
        
        Returns:
            List of dicts with provider statistics:
            [
                {
                    'provider': 'azure_openai_gpt-4',
                    'count': 100,
                    'avg_tokens': 150.5,
                    'avg_latency_ms': 1250.3,
                    'total_tokens': 15050
                },
                ...
            ]
        """
        results = self.session.query(
            InferenceRun.provider,
            func.count(InferenceRun.id).label('count'),
            func.avg(InferenceRun.tokens).label('avg_tokens'),
            func.avg(InferenceRun.latency_ms).label('avg_latency_ms'),
            func.sum(InferenceRun.tokens).label('total_tokens')
        ).group_by(InferenceRun.provider).all()

        return [
            {
                'provider': r.provider,
                'count': r.count,
                'avg_tokens': round(float(r.avg_tokens), 2) if r.avg_tokens else 0,
                'avg_latency_ms': round(float(r.avg_latency_ms), 2) if r.avg_latency_ms else 0,
                'total_tokens': int(r.total_tokens) if r.total_tokens else 0
            }
            for r in results
        ]

    def search_by_prompt(self, search_term: str, limit: int = 50) -> List[InferenceRun]:
        """
        Search inferences by prompt content.
        
        Performs case-insensitive substring search on the prompt field.
        
        Args:
            search_term: Text to search for in prompts
            limit: Maximum number of results (default: 50)
        
        Returns:
            List of matching InferenceRun objects, ordered by created_at descending
        """
        return self.session.query(InferenceRun)\
            .filter(InferenceRun.prompt.ilike(f'%{search_term}%'))\
            .order_by(desc(InferenceRun.created_at))\
            .limit(limit)\
            .all()

    def get_total_count(self) -> int:
        """
        Get total number of inference runs in the database.
        
        Returns:
            Total count of inference runs
        """
        return self.session.query(func.count(InferenceRun.id)).scalar()

    def get_count_by_provider(self, provider: str) -> int:
        """
        Get count of inference runs for a specific provider.
        
        Args:
            provider: Provider name to count
        
        Returns:
            Count of inference runs for the provider
        """
        return self.session.query(func.count(InferenceRun.id))\
            .filter(InferenceRun.provider == provider)\
            .scalar()

    def get_inferences_by_batch_id(self, batch_id: str) -> List[InferenceRun]:
        """
        Get all inference runs belonging to a specific batch.
        
        Args:
            batch_id: UUID string identifying the batch
        
        Returns:
            List of InferenceRun objects in the batch, ordered by created_at
        """
        return self.session.query(InferenceRun)\
            .filter(InferenceRun.batch_id == batch_id)\
            .order_by(desc(InferenceRun.created_at))\
            .all()

    def get_batch_summary(self, batch_id: str) -> dict:
        """
        Get aggregate statistics for a specific batch.
        
        Args:
            batch_id: UUID string identifying the batch
        
        Returns:
            Dictionary with batch statistics:
            {
                'batch_id': '...',
                'total_runs': 5,
                'providers': ['azure', 'openai'],
                'total_tokens': 500,
                'avg_tokens': 100.0,
                'avg_latency_ms': 1250.5,
                'min_latency_ms': 800.0,
                'max_latency_ms': 2000.0,
                'created_at_range': (start_datetime, end_datetime)
            }
        """
        runs = self.get_inferences_by_batch_id(batch_id)
        
        if not runs:
            return {
                'batch_id': batch_id,
                'total_runs': 0,
                'providers': [],
                'total_tokens': 0,
                'avg_tokens': 0.0,
                'avg_latency_ms': 0.0,
                'min_latency_ms': None,
                'max_latency_ms': None,
                'created_at_range': (None, None)
            }
        
        tokens_list = [r.tokens for r in runs if r.tokens is not None]
        latency_list = [r.latency_ms for r in runs if r.latency_ms is not None]
        providers = list(set(r.provider for r in runs))
        created_ats = [r.created_at for r in runs if r.created_at is not None]
        
        return {
            'batch_id': batch_id,
            'total_runs': len(runs),
            'providers': providers,
            'total_tokens': sum(tokens_list),
            'avg_tokens': sum(tokens_list) / len(tokens_list) if tokens_list else 0.0,
            'avg_latency_ms': sum(latency_list) / len(latency_list) if latency_list else 0.0,
            'min_latency_ms': min(latency_list) if latency_list else None,
            'max_latency_ms': max(latency_list) if latency_list else None,
            'created_at_range': (
                min(created_ats) if created_ats else None,
                max(created_ats) if created_ats else None
            )
        }

