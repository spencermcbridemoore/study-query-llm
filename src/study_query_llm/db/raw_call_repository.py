"""
Raw Call Repository - Database operations for v2 immutable capture schema.

This repository handles all database interactions for storing and querying
raw calls in the v2 schema. Uses the Repository pattern to abstract database details.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, cast, Float, String, text
from sqlalchemy.exc import IntegrityError
from .models_v2 import (
    RawCall,
    Group,
    GroupMember,
    CallArtifact,
    EmbeddingVector,
    EmbeddingCacheEntry,
    EmbeddingCacheLease,
    GroupLink,
    OrchestrationJob,
    OrchestrationJobDependency,
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Multiplier for overfetching when searching by prompt (to filter in Python)
SEARCH_OVERFETCH_MULTIPLIER = 10


class RawCallRepository:
    """
    Repository for all database operations on raw calls (v2 schema).
    
    This class handles both writes (logging raw calls) and queries (analytics).
    All database operations go through this repository, keeping the service
    layer database-agnostic.
    
    Usage:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            
            # Write
            call_id = repo.insert_raw_call(...)
            
            # Query
            calls = repo.query_raw_calls(provider="azure")
    """

    def __init__(self, session: Session):
        """
        Initialize repository with a database session.
        
        Args:
            session: SQLAlchemy Session instance
        """
        self.session = session

    # ========== WRITE OPERATIONS ==========

    def insert_raw_call(
        self,
        provider: str,
        request_json: Dict[str, Any],
        model: Optional[str] = None,
        modality: str = "text",
        status: str = "success",
        response_json: Optional[Dict[str, Any]] = None,
        error_json: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        tokens_json: Optional[Dict[str, Any]] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert a single raw call.
        
        Args:
            provider: Provider name
            request_json: Full request payload as JSON dict
            model: Model/deployment name (optional)
            modality: Type of call ('text', 'embedding', etc.)
            status: Call status ('success', 'failed', 'timeout', 'cancelled')
            response_json: Full response payload as JSON dict (optional)
            error_json: Error details as JSON dict (optional)
            latency_ms: Response latency in milliseconds (optional)
            tokens_json: Token usage breakdown as JSON dict (optional)
            metadata_json: Additional metadata as JSON dict (optional)
        
        Returns:
            The ID of the inserted record
        """
        raw_call = RawCall(
            provider=provider,
            model=model,
            modality=modality,
            status=status,
            request_json=request_json,
            response_json=response_json,
            error_json=error_json,
            latency_ms=latency_ms,
            tokens_json=tokens_json,
            metadata_json=metadata_json or {},
        )

        self.session.add(raw_call)
        self.session.flush()  # Flush to get the ID without committing
        self.session.refresh(raw_call)
        
        logger.debug(
            f"Inserted raw call: id={raw_call.id}, provider={provider}, "
            f"modality={modality}, status={status}"
        )

        return raw_call.id

    def batch_insert_raw_calls(self, calls: List[Dict[str, Any]]) -> List[int]:
        """
        Batch insert multiple raw calls.
        
        More efficient than multiple single inserts for bulk operations.
        
        Args:
            calls: List of dicts with raw call data. Each dict should have:
                - provider (str)
                - request_json (dict)
                - model (str, optional)
                - modality (str, optional)
                - status (str, optional)
                - response_json (dict, optional)
                - error_json (dict, optional)
                - latency_ms (float, optional)
                - tokens_json (dict, optional)
                - metadata_json (dict, optional)
        
        Returns:
            List of inserted IDs in the same order as input
        """
        raw_call_objects = [
            RawCall(**call) for call in calls
        ]

        self.session.add_all(raw_call_objects)
        self.session.flush()  # Flush to get IDs without committing
        
        return [call.id for call in raw_call_objects]

    # ========== QUERY OPERATIONS ==========

    def get_raw_call_by_id(self, call_id: int) -> Optional[RawCall]:
        """
        Retrieve a specific raw call by ID.
        
        Args:
            call_id: The ID of the raw call
        
        Returns:
            RawCall object if found, None otherwise
        """
        return self.session.query(RawCall).filter_by(id=call_id).first()

    def query_raw_calls(
        self,
        provider: Optional[str] = None,
        modality: Optional[str] = None,
        status: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[RawCall]:
        """
        Query raw calls with filters.
        
        Args:
            provider: Filter by provider name (optional)
            modality: Filter by modality (optional)
            status: Filter by status (optional)
            date_range: Tuple of (start_date, end_date) for filtering (optional)
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip for pagination (default: 0)
        
        Returns:
            List of RawCall objects, ordered by created_at descending
        """
        query = self.session.query(RawCall)

        if provider:
            query = query.filter(RawCall.provider == provider)
        
        if modality:
            query = query.filter(RawCall.modality == modality)
        
        if status:
            query = query.filter(RawCall.status == status)

        if date_range:
            start_date, end_date = date_range
            query = query.filter(RawCall.created_at.between(start_date, end_date))

        query = query.order_by(desc(RawCall.created_at))
        query = query.limit(limit).offset(offset)

        return query.all()

    def get_provider_stats(self) -> List[Dict[str, Any]]:
        """
        Get aggregate statistics by provider.
        
        Extracts token counts from tokens_json and calculates averages/totals.
        Uses SQL aggregation for efficiency.
        
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
        try:
            # Use raw SQL with JSONB operators to avoid SQLAlchemy version issues
            # with .astext on nested JSON paths.
            sql = text("""
                SELECT
                    provider,
                    COUNT(*) AS count,
                    AVG(
                        COALESCE(
                            (tokens_json->>'total_tokens')::float,
                            (tokens_json->>'totalTokens')::float,
                            (tokens_json->'usage'->>'total_tokens')::float,
                            (tokens_json->'usage'->>'totalTokens')::float,
                            0.0
                        )
                    ) AS avg_tokens,
                    AVG(latency_ms) AS avg_latency_ms,
                    SUM(
                        COALESCE(
                            (tokens_json->>'total_tokens')::float,
                            (tokens_json->>'totalTokens')::float,
                            (tokens_json->'usage'->>'total_tokens')::float,
                            (tokens_json->'usage'->>'totalTokens')::float,
                            0.0
                        )
                    ) AS total_tokens
                FROM raw_calls
                GROUP BY provider
            """)
            rows = self.session.execute(sql).fetchall()
            results = []
            for row in rows:
                results.append({
                    'provider': row.provider,
                    'count': row.count,
                    'avg_tokens': round(float(row.avg_tokens or 0), 2),
                    'avg_latency_ms': round(float(row.avg_latency_ms or 0), 2),
                    'total_tokens': int(row.total_tokens or 0)
                })
            return results

        except Exception as e:
            # Fallback for SQLite or other backends: use Python aggregation
            logger.warning(f"SQL aggregation failed, falling back to Python: {e}")
            all_calls = self.session.query(RawCall).all()
            
            # Group by provider and calculate stats
            provider_data = {}
            for call in all_calls:
                provider = call.provider
                if provider not in provider_data:
                    provider_data[provider] = {
                        'count': 0,
                        'latencies': [],
                        'tokens': []
                    }
                
                provider_data[provider]['count'] += 1
                if call.latency_ms:
                    provider_data[provider]['latencies'].append(call.latency_ms)
                
                # Extract total tokens from tokens_json
                if call.tokens_json and isinstance(call.tokens_json, dict):
                    # Try common token field names
                    total_tokens = (
                        call.tokens_json.get('total_tokens') or
                        call.tokens_json.get('totalTokens') or
                        call.tokens_json.get('usage', {}).get('total_tokens') or
                        call.tokens_json.get('usage', {}).get('totalTokens') or
                        0
                    )
                    if total_tokens:
                        provider_data[provider]['tokens'].append(total_tokens)
            
            # Build result list
            results = []
            for provider, data in provider_data.items():
                avg_latency = (
                    sum(data['latencies']) / len(data['latencies'])
                    if data['latencies'] else 0
                )
                avg_tokens = (
                    sum(data['tokens']) / len(data['tokens'])
                    if data['tokens'] else 0
                )
                total_tokens = sum(data['tokens']) if data['tokens'] else 0
                
                results.append({
                    'provider': provider,
                    'count': data['count'],
                    'avg_tokens': round(float(avg_tokens), 2),
                    'avg_latency_ms': round(float(avg_latency), 2),
                    'total_tokens': int(total_tokens)
                })
            
            return results

    def get_total_count(self) -> int:
        """
        Get total number of raw calls in the database.
        
        Returns:
            Total count of raw calls
        """
        return self.session.query(func.count(RawCall.id)).scalar()

    def get_count_by_provider(self, provider: str) -> int:
        """
        Get count of raw calls for a specific provider.
        
        Args:
            provider: Provider name to count
        
        Returns:
            Count of raw calls for the provider
        """
        return self.session.query(func.count(RawCall.id))\
            .filter(RawCall.provider == provider)\
            .scalar()

    def search_by_prompt(self, search_term: str, limit: int = 50) -> List[RawCall]:
        """
        Search raw calls by prompt content in request_json.
        
        Performs case-insensitive substring search on prompt text extracted
        from request_json. Looks for 'prompt', 'messages', or 'input' fields.
        Uses SQL filtering for efficiency.
        
        Args:
            search_term: Text to search for in prompts
            limit: Maximum number of results (default: 50)
        
        Returns:
            List of matching RawCall objects, ordered by created_at descending
        """
        try:
            # PostgreSQL path: Use JSON operators with ILIKE for case-insensitive search
            # Search in multiple possible JSON fields: prompt, input, and messages array
            search_pattern = f'%{search_term}%'
            
            # Build conditions for different JSON field locations
            # PostgreSQL JSON operators: -> for JSON object, ->> for text extraction
            conditions = []
            
            # Search in 'prompt' field (request_json->>'prompt')
            conditions.append(
                RawCall.request_json['prompt'].astext.ilike(search_pattern)
            )
            
            # Search in 'input' field (request_json->>'input')
            conditions.append(
                RawCall.request_json['input'].astext.ilike(search_pattern)
            )
            
            # For messages array, search the entire array as text
            # This will match content within any message object
            # (request_json->'messages'::text or request_json->>'messages')
            conditions.append(
                func.cast(RawCall.request_json['messages'], String).ilike(search_pattern)
            )
            
            query = self.session.query(RawCall)\
                .filter(RawCall.modality == 'text')\
                .filter(or_(*conditions))\
                .order_by(desc(RawCall.created_at))\
                .limit(limit)
            
            return query.all()
            
        except Exception as e:
            # Fallback for SQLite or other backends: use Python filtering
            logger.warning(f"SQL search failed, falling back to Python: {e}")
            all_calls = self.session.query(RawCall)\
                .filter(RawCall.modality == 'text')\
                .order_by(desc(RawCall.created_at))\
                .limit(limit * SEARCH_OVERFETCH_MULTIPLIER)\
                .all()  # Get more to filter from
            
            matches = []
            search_lower = search_term.lower()
            
            for call in all_calls:
                if len(matches) >= limit:
                    break
                    
                # Extract prompt text from request_json
                request = call.request_json or {}
                prompt_text = None
                
                # Try common prompt field names
                if isinstance(request, dict):
                    prompt_text = (
                        request.get('prompt') or
                        request.get('input') or
                        ''
                    )
                    
                    # If messages array, extract text from first user message
                    if not prompt_text and 'messages' in request:
                        messages = request.get('messages', [])
                        if messages and isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict) and msg.get('role') == 'user':
                                    prompt_text = msg.get('content', '')
                                    break
                
                # Search in prompt text
                if prompt_text and search_lower in str(prompt_text).lower():
                    matches.append(call)
            
            return matches[:limit]

    # ========== GROUP OPERATIONS ==========

    def create_group(
        self,
        group_type: str,
        name: str,
        description: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a new group.
        
        Args:
            group_type: Type of group ('batch', 'experiment', 'label', etc.)
            name: Group name/identifier
            description: Optional description
            metadata_json: Additional metadata as JSON dict (optional)
        
        Returns:
            The ID of the created group
        """
        group = Group(
            group_type=group_type,
            name=name,
            description=description,
            metadata_json=metadata_json or {},
        )
        
        self.session.add(group)
        self.session.flush()
        self.session.refresh(group)
        
        logger.debug(f"Created group: id={group.id}, type={group_type}, name={name}")
        
        return group.id

    def add_call_to_group(
        self,
        group_id: int,
        call_id: int,
        position: Optional[int] = None,
        role: Optional[str] = None,
    ) -> int:
        """
        Add a raw call to a group.
        
        Args:
            group_id: ID of the group
            call_id: ID of the raw call
            position: Optional ordering within the group
            role: Optional role/label for this member
        
        Returns:
            The ID of the created GroupMember
        """
        # Check if already exists
        existing = self.session.query(GroupMember).filter_by(
            group_id=group_id,
            call_id=call_id
        ).first()
        
        if existing:
            logger.debug(f"Call {call_id} already in group {group_id}")
            return existing.id
        
        member = GroupMember(
            group_id=group_id,
            call_id=call_id,
            position=position,
            role=role,
        )
        
        self.session.add(member)
        self.session.flush()
        self.session.refresh(member)
        
        logger.debug(f"Added call {call_id} to group {group_id}")
        
        return member.id

    def get_group_by_id(self, group_id: int) -> Optional[Group]:
        """
        Retrieve a group by ID.
        
        Args:
            group_id: The ID of the group
        
        Returns:
            Group object if found, None otherwise
        """
        return self.session.query(Group).filter_by(id=group_id).first()

    def get_calls_in_group(self, group_id: int) -> List[RawCall]:
        """
        Get all raw calls belonging to a specific group.
        
        Args:
            group_id: ID of the group
        
        Returns:
            List of RawCall objects in the group, ordered by position/created_at
        """
        members = self.session.query(GroupMember).filter_by(
            group_id=group_id
        ).order_by(GroupMember.position, GroupMember.added_at).all()
        
        call_ids = [m.call_id for m in members]
        if not call_ids:
            return []
        
        return self.session.query(RawCall).filter(
            RawCall.id.in_(call_ids)
        ).order_by(desc(RawCall.created_at)).all()

    def get_groups_for_call(self, call_id: int) -> List[Group]:
        """
        Get all groups that contain a specific raw call.
        
        Args:
            call_id: ID of the raw call
        
        Returns:
            List of Group objects
        """
        members = self.session.query(GroupMember).filter_by(
            call_id=call_id
        ).all()
        
        group_ids = [m.group_id for m in members]
        if not group_ids:
            return []
        
        return self.session.query(Group).filter(
            Group.id.in_(group_ids)
        ).all()

    def query_raw_calls_excluding_defective(
        self,
        provider: Optional[str] = None,
        modality: Optional[str] = None,
        status: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100,
        offset: int = 0,
        defective_group_id: Optional[int] = None
    ) -> List[RawCall]:
        """
        Query raw calls with filters, excluding those marked as defective.
        
        Uses SQL LEFT JOIN to exclude defective calls efficiently in the database.
        
        Args:
            provider: Filter by provider name (optional)
            modality: Filter by modality (optional)
            status: Filter by status (optional)
            date_range: Tuple of (start_date, end_date) for filtering (optional)
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip for pagination (default: 0)
            defective_group_id: ID of defective group to exclude (optional, will be looked up if not provided)
        
        Returns:
            List of RawCall objects, ordered by created_at descending, excluding defective calls
        """
        # Get defective group ID if not provided
        # Note: For better separation of concerns, consider using DataQualityService
        # to get the defective_group_id before calling this method
        if defective_group_id is None:
            # Fallback: query directly (for backward compatibility)
            # This maintains backward compatibility but new code should use DataQualityService
            existing = self.session.query(Group).filter_by(
                group_type="label",
                name="defective_data"
            ).first()
            
            if existing:
                defective_group_id = existing.id
            else:
                # If group doesn't exist, no calls are defective, so return all
                return self.query_raw_calls(
                    provider=provider,
                    modality=modality,
                    status=status,
                    date_range=date_range,
                    limit=limit,
                    offset=offset
                )
        
        # Build base query
        query = self.session.query(RawCall).outerjoin(
            GroupMember,
            and_(
                RawCall.id == GroupMember.call_id,
                GroupMember.group_id == defective_group_id
            )
        ).filter(
            GroupMember.id == None  # Exclude calls that are in defective group
        )
        
        # Apply filters
        if provider:
            query = query.filter(RawCall.provider == provider)
        
        if modality:
            query = query.filter(RawCall.modality == modality)
        
        if status:
            query = query.filter(RawCall.status == status)
        
        if date_range:
            start_date, end_date = date_range
            query = query.filter(RawCall.created_at.between(start_date, end_date))
        
        # Order and paginate
        query = query.order_by(desc(RawCall.created_at))
        query = query.limit(limit).offset(offset)
        
        return query.all()

    # ========== EMBEDDING BATCH LOOKUP ==========

    def get_embedding_vectors_by_request_hashes(
        self,
        deployment: str,
        request_hashes: List[str],
    ) -> Dict[str, Tuple[list, int]]:
        """
        Batch-lookup cached embeddings by their request hashes.

        Returns a dict mapping request_hash -> (vector_list, raw_call_id) for
        every hash that already has a successful embedding RawCall persisted for
        the given deployment.  Hashes not found are simply absent from the result.

        This is used by the chunked batching path in EmbeddingService to determine
        which texts still need API calls, enabling resume-without-re-fetching.

        Args:
            deployment: Model/deployment name (e.g. 'text-embedding-3-small').
            request_hashes: List of request hashes to look up.

        Returns:
            Dict mapping request_hash -> (vector, raw_call_id).
        """
        if not request_hashes:
            return {}

        hash_set = set(request_hashes)

        try:
            # Attempt the PostgreSQL JSON path operator (->>)
            pg_results = (
                self.session.query(RawCall, EmbeddingVector)
                .join(EmbeddingVector, EmbeddingVector.call_id == RawCall.id)
                .filter(
                    RawCall.modality == "embedding",
                    RawCall.model == deployment,
                    RawCall.status == "success",
                    RawCall.metadata_json["request_hash"].astext.in_(request_hashes),
                )
                .all()
            )
            return {
                raw_call.metadata_json["request_hash"]: (ev.vector, raw_call.id)
                for raw_call, ev in pg_results
                if raw_call.metadata_json and "request_hash" in raw_call.metadata_json
            }

        except Exception:
            # Fallback for SQLite or other backends: fetch all matching embedding
            # calls and filter by hash in Python.
            try:
                all_calls = (
                    self.session.query(RawCall, EmbeddingVector)
                    .join(EmbeddingVector, EmbeddingVector.call_id == RawCall.id)
                    .filter(
                        RawCall.modality == "embedding",
                        RawCall.model == deployment,
                        RawCall.status == "success",
                    )
                    .all()
                )
                return {
                    raw_call.metadata_json["request_hash"]: (ev.vector, raw_call.id)
                    for raw_call, ev in all_calls
                    if (
                        raw_call.metadata_json
                        and raw_call.metadata_json.get("request_hash") in hash_set
                    )
                }
            except Exception as e2:
                logger.warning(
                    "Batch hash lookup fallback also failed; returning empty result: %s", e2
                )
                return {}

    # ========== EMBEDDING L2 CACHE + SINGLE-FLIGHT LEASES ==========

    def get_embedding_cache_entry(self, cache_key: str) -> Optional[EmbeddingCacheEntry]:
        """Return L2 embedding cache entry for exact cache_key if present."""
        return (
            self.session.query(EmbeddingCacheEntry)
            .filter(EmbeddingCacheEntry.cache_key == cache_key)
            .first()
        )

    def get_embedding_cache_vectors_by_keys(
        self, cache_keys: List[str]
    ) -> Dict[str, Tuple[list, Optional[int]]]:
        """Batch lookup vectors by L2 cache keys."""
        if not cache_keys:
            return {}
        rows = (
            self.session.query(
                EmbeddingCacheEntry.cache_key,
                EmbeddingCacheEntry.vector,
                EmbeddingCacheEntry.source_raw_call_id,
            )
            .filter(EmbeddingCacheEntry.cache_key.in_(cache_keys))
            .all()
        )
        return {str(k): (v, rcid) for k, v, rcid in rows}

    def upsert_embedding_cache_entry(
        self,
        *,
        cache_key: str,
        key_version: str,
        provider: str,
        deployment: str,
        dimensions: Optional[int],
        encoding_format: str,
        input_text_raw: str,
        input_text_sha256_raw: str,
        vector: list,
        dimension: int,
        norm: Optional[float],
        source_raw_call_id: Optional[int],
    ) -> int:
        """
        Insert or update an L2 cache row keyed by cache_key.

        Returns:
            EmbeddingCacheEntry.id
        """
        now = datetime.now(timezone.utc)
        existing = (
            self.session.query(EmbeddingCacheEntry)
            .filter(EmbeddingCacheEntry.cache_key == cache_key)
            .first()
        )
        if existing:
            existing.vector = vector
            existing.dimension = int(dimension)
            existing.norm = norm
            existing.source_raw_call_id = source_raw_call_id
            existing.updated_at = now
            self.session.flush()
            return int(existing.id)

        row = EmbeddingCacheEntry(
            cache_key=cache_key,
            key_version=key_version,
            provider=provider,
            deployment=deployment,
            dimensions=dimensions,
            encoding_format=encoding_format,
            input_text_raw=input_text_raw,
            input_text_sha256_raw=input_text_sha256_raw,
            vector=vector,
            dimension=int(dimension),
            norm=norm,
            source_raw_call_id=source_raw_call_id,
            hit_count=0,
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        self.session.flush()
        return int(row.id)

    def touch_embedding_cache_hit(self, cache_key: str) -> None:
        """Best-effort hit counter update for L2 cache rows."""
        now = datetime.now(timezone.utc)
        row = (
            self.session.query(EmbeddingCacheEntry)
            .filter(EmbeddingCacheEntry.cache_key == cache_key)
            .first()
        )
        if not row:
            return
        row.hit_count = int(row.hit_count or 0) + 1
        row.last_hit_at = now
        row.updated_at = now
        self.session.flush()

    def try_acquire_embedding_cache_lease(
        self,
        *,
        cache_key: str,
        owner: str,
        lease_seconds: int,
    ) -> bool:
        """
        Acquire/steal lease for cache_key when expired.

        Returns True if acquired by owner, False otherwise.
        """
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=max(1, int(lease_seconds)))
        lease = (
            self.session.query(EmbeddingCacheLease)
            .filter(EmbeddingCacheLease.cache_key == cache_key)
            .first()
        )
        if lease is None:
            lease = EmbeddingCacheLease(
                cache_key=cache_key,
                lease_owner=owner,
                lease_expires_at=expires,
                created_at=now,
                updated_at=now,
            )
            self.session.add(lease)
            self.session.flush()
            return True

        if lease.lease_owner == owner or lease.lease_expires_at <= now:
            lease.lease_owner = owner
            lease.lease_expires_at = expires
            lease.updated_at = now
            self.session.flush()
            return True
        return False

    def release_embedding_cache_lease(self, *, cache_key: str, owner: str) -> None:
        """Release lease row only when owned by owner."""
        lease = (
            self.session.query(EmbeddingCacheLease)
            .filter(
                EmbeddingCacheLease.cache_key == cache_key,
                EmbeddingCacheLease.lease_owner == owner,
            )
            .first()
        )
        if not lease:
            return
        self.session.delete(lease)
        self.session.flush()

    def get_embedding_cache_lease(self, cache_key: str) -> Optional[EmbeddingCacheLease]:
        """Return current lease row if present."""
        return (
            self.session.query(EmbeddingCacheLease)
            .filter(EmbeddingCacheLease.cache_key == cache_key)
            .first()
        )

    # ========== GROUP LINK OPERATIONS ==========

    def create_group_link(
        self,
        parent_group_id: int,
        child_group_id: int,
        link_type: str,
        position: Optional[int] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a link between two groups.

        Args:
            parent_group_id: ID of the parent group
            child_group_id: ID of the child group
            link_type: Type of relationship ('step', 'contains', 'depends_on', 'generates')
            position: Optional ordering within parent (for step sequences)
            metadata_json: Optional additional relationship metadata

        Returns:
            GroupLink ID
        """
        # Fast-path: check if link already exists
        existing = self.session.query(GroupLink).filter_by(
            parent_group_id=parent_group_id,
            child_group_id=child_group_id,
            link_type=link_type,
        ).first()

        if existing:
            logger.debug(
                f"Link already exists: parent={parent_group_id}, "
                f"child={child_group_id}, type={link_type}"
            )
            return existing.id

        link = GroupLink(
            parent_group_id=parent_group_id,
            child_group_id=child_group_id,
            link_type=link_type,
            position=position,
            metadata_json=metadata_json or {},
        )

        try:
            # Nested transaction lets us recover from uniqueness conflicts
            # without rolling back the outer unit of work.
            with self.session.begin_nested():
                self.session.add(link)
                self.session.flush()
                self.session.refresh(link)
        except IntegrityError:
            existing = self.session.query(GroupLink).filter_by(
                parent_group_id=parent_group_id,
                child_group_id=child_group_id,
                link_type=link_type,
            ).first()
            if existing:
                logger.debug(
                    "Recovered concurrent create_group_link collision for "
                    "parent=%s child=%s type=%s",
                    parent_group_id,
                    child_group_id,
                    link_type,
                )
                return existing.id
            raise

        logger.debug(
            f"Created group link: id={link.id}, parent={parent_group_id}, "
            f"child={child_group_id}, type={link_type}, position={position}"
        )

        return link.id

    def get_group_children(
        self,
        parent_group_id: int,
        link_type: Optional[str] = None,
    ) -> List[Group]:
        """
        Get all child groups for a parent.

        Args:
            parent_group_id: ID of the parent group
            link_type: Optional filter by link type

        Returns:
            List of child Group objects, ordered by position if available
        """
        query = self.session.query(GroupLink).filter_by(
            parent_group_id=parent_group_id
        )

        if link_type:
            query = query.filter_by(link_type=link_type)

        links = query.order_by(GroupLink.position, GroupLink.created_at).all()

        child_ids = [link.child_group_id for link in links]
        if not child_ids:
            return []

        return self.session.query(Group).filter(Group.id.in_(child_ids)).all()

    def get_group_parents(
        self,
        child_group_id: int,
        link_type: Optional[str] = None,
    ) -> List[Group]:
        """
        Get all parent groups for a child.

        Args:
            child_group_id: ID of the child group
            link_type: Optional filter by link type

        Returns:
            List of parent Group objects
        """
        query = self.session.query(GroupLink).filter_by(
            child_group_id=child_group_id
        )

        if link_type:
            query = query.filter_by(link_type=link_type)

        links = query.order_by(GroupLink.created_at).all()

        parent_ids = [link.parent_group_id for link in links]
        if not parent_ids:
            return []

        return self.session.query(Group).filter(Group.id.in_(parent_ids)).all()

    def get_run_step_sequence(self, run_id: int) -> List[Group]:
        """
        Get ordered step sequence for a run.

        Args:
            run_id: ID of the run group

        Returns:
            List of step Group objects, ordered by position
        """
        links = self.session.query(GroupLink).filter_by(
            parent_group_id=run_id,
            link_type="clustering_step",
        ).order_by(GroupLink.position, GroupLink.created_at).all()

        step_ids = [link.child_group_id for link in links]
        if not step_ids:
            return []

        # Get groups and preserve order
        groups = self.session.query(Group).filter(Group.id.in_(step_ids)).all()
        group_dict = {g.id: g for g in groups}

        # Return in link order
        return [group_dict[step_id] for step_id in step_ids if step_id in group_dict]

    # ========== ORCHESTRATION JOB OPERATIONS ==========

    def enqueue_orchestration_job(
        self,
        *,
        request_group_id: int,
        job_type: str,
        job_key: str,
        base_run_key: Optional[str] = None,
        payload_json: Optional[Dict[str, Any]] = None,
        priority: int = 100,
        max_attempts: int = 3,
        seed_value: Optional[int] = None,
        parent_job_id: Optional[int] = None,
        depends_on_job_ids: Optional[List[int]] = None,
    ) -> int:
        """Idempotently create (or return existing) orchestration job."""
        existing = self.session.query(OrchestrationJob).filter_by(job_key=job_key).first()
        if existing:
            return existing.id

        now = datetime.now(timezone.utc)
        initial_status = "pending" if depends_on_job_ids else "ready"
        job = OrchestrationJob(
            request_group_id=request_group_id,
            parent_job_id=parent_job_id,
            job_type=job_type,
            job_key=job_key,
            base_run_key=base_run_key,
            status=initial_status,
            priority=priority,
            payload_json=payload_json or {},
            seed_value=seed_value,
            attempt_count=0,
            max_attempts=max_attempts,
            created_at=now,
            updated_at=now,
        )
        try:
            with self.session.begin_nested():
                self.session.add(job)
                self.session.flush()
                self.session.refresh(job)
        except IntegrityError:
            existing = self.session.query(OrchestrationJob).filter_by(job_key=job_key).first()
            if existing:
                return existing.id
            raise

        for dep_id in depends_on_job_ids or []:
            dep = OrchestrationJobDependency(job_id=job.id, depends_on_job_id=dep_id)
            try:
                with self.session.begin_nested():
                    self.session.add(dep)
                    self.session.flush()
            except IntegrityError:
                # Idempotent edge insert
                pass

        return job.id

    def claim_next_orchestration_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        job_types: Optional[List[str]] = None,
        request_group_id: Optional[int] = None,
        base_run_key: Optional[str] = None,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[OrchestrationJob]:
        """Claim the highest-priority ready job that matches filters."""
        now = datetime.now(timezone.utc)
        q = self.session.query(OrchestrationJob).filter(
            OrchestrationJob.status.in_(["ready", "claimed"]),
        )
        if job_types:
            q = q.filter(OrchestrationJob.job_type.in_(job_types))
        if request_group_id is not None:
            q = q.filter(OrchestrationJob.request_group_id == request_group_id)
        if base_run_key is not None:
            q = q.filter(OrchestrationJob.base_run_key == base_run_key)

        candidates = q.order_by(
            OrchestrationJob.priority.asc(),
            OrchestrationJob.created_at.asc(),
        ).all()

        for job in candidates:
            if job.status == "claimed" and job.lease_expires_at and job.lease_expires_at > now:
                continue
            if filter_payload:
                payload = job.payload_json or {}
                if any(payload.get(k) != v for k, v in filter_payload.items()):
                    continue
            # Dependency gate
            if not self.is_orchestration_job_ready(job.id):
                continue
            job.status = "claimed"
            job.claimed_by = worker_id
            job.claimed_at = now
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(seconds=lease_seconds)
            job.attempt_count = int(job.attempt_count or 0) + 1
            job.updated_at = now
            self.session.flush()
            return job
        return None

    def claim_orchestration_job_batch(
        self,
        request_group_id: int,
        job_types: List[str],
        claim_owner: str,
        lease_seconds: int,
        limit: int,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Claim up to limit ready jobs matching filters; return detached snapshots."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=max(1, int(lease_seconds)))
        q = self.session.query(OrchestrationJob).filter(
            OrchestrationJob.request_group_id == request_group_id,
            OrchestrationJob.job_type.in_(job_types),
            or_(
                OrchestrationJob.status == "ready",
                and_(
                    OrchestrationJob.status == "claimed",
                    or_(
                        OrchestrationJob.lease_expires_at.is_(None),
                        OrchestrationJob.lease_expires_at <= now,
                    ),
                ),
            ),
        )
        candidates = q.order_by(
            OrchestrationJob.priority.asc(),
            OrchestrationJob.created_at.asc(),
        ).all()

        result: List[Dict[str, Any]] = []
        for job in candidates:
            if len(result) >= limit:
                break
            if filter_payload:
                payload = job.payload_json or {}
                if any(payload.get(k) != v for k, v in filter_payload.items()):
                    continue
            if not self.is_orchestration_job_ready(job.id):
                continue
            job.status = "claimed"
            job.claimed_by = claim_owner
            job.claimed_at = now
            job.heartbeat_at = now
            job.lease_expires_at = expires
            job.attempt_count = int(job.attempt_count or 0) + 1
            job.updated_at = now
            self.session.flush()
            snapshot = {
                "id": int(job.id),
                "job_type": str(job.job_type),
                "payload_json": dict(job.payload_json or {}),
                "seed_value": job.seed_value,
                "job_key": str(job.job_key),
                "base_run_key": job.base_run_key,
            }
            result.append(snapshot)
        return result

    def heartbeat_orchestration_job(self, job_id: int, lease_seconds: int) -> bool:
        now = datetime.now(timezone.utc)
        job = self.session.query(OrchestrationJob).filter_by(id=job_id).first()
        if not job or job.status != "claimed":
            return False
        job.heartbeat_at = now
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.updated_at = now
        self.session.flush()
        return True

    def complete_orchestration_job(self, job_id: int, result_ref: Optional[str] = None) -> bool:
        now = datetime.now(timezone.utc)
        job = self.session.query(OrchestrationJob).filter_by(id=job_id).first()
        if not job:
            return False
        job.status = "completed"
        job.result_ref = result_ref
        job.lease_expires_at = None
        job.heartbeat_at = now
        job.updated_at = now
        self.session.flush()
        self.promote_ready_orchestration_jobs(request_group_id=job.request_group_id)
        return True

    def complete_orchestration_jobs_batch(
        self, items: List[Tuple[int, Optional[str]]]
    ) -> None:
        """Complete multiple jobs; promote once per distinct request_group_id at end."""
        if not items:
            return
        now = datetime.now(timezone.utc)
        request_group_ids: set = set()
        for job_id, result_ref in items:
            job = self.session.query(OrchestrationJob).filter_by(id=job_id).first()
            if job:
                job.status = "completed"
                job.result_ref = result_ref
                job.lease_expires_at = None
                job.heartbeat_at = now
                job.updated_at = now
                request_group_ids.add(job.request_group_id)
            self.session.flush()
        for rgid in request_group_ids:
            self.promote_ready_orchestration_jobs(request_group_id=rgid)

    def fail_orchestration_job(self, job_id: int, error_json: Optional[Dict[str, Any]] = None) -> bool:
        now = datetime.now(timezone.utc)
        job = self.session.query(OrchestrationJob).filter_by(id=job_id).first()
        if not job:
            return False
        if int(job.attempt_count or 0) >= int(job.max_attempts or 1):
            job.status = "failed"
        else:
            job.status = "ready"
        job.error_json = error_json or {}
        job.lease_expires_at = None
        job.updated_at = now
        self.session.flush()
        return True

    def release_orchestration_job(self, job_id: int) -> bool:
        now = datetime.now(timezone.utc)
        job = self.session.query(OrchestrationJob).filter_by(id=job_id).first()
        if not job:
            return False
        if job.status == "claimed":
            job.status = "ready"
            job.lease_expires_at = None
            job.updated_at = now
            self.session.flush()
        return True

    def add_orchestration_job_dependency(self, job_id: int, depends_on_job_id: int) -> int:
        dep = (
            self.session.query(OrchestrationJobDependency)
            .filter_by(job_id=job_id, depends_on_job_id=depends_on_job_id)
            .first()
        )
        if dep:
            return dep.id
        dep = OrchestrationJobDependency(job_id=job_id, depends_on_job_id=depends_on_job_id)
        self.session.add(dep)
        self.session.flush()
        self.session.refresh(dep)
        return dep.id

    def is_orchestration_job_ready(self, job_id: int) -> bool:
        deps = self.session.query(OrchestrationJobDependency).filter_by(job_id=job_id).all()
        if not deps:
            return True
        dep_ids = [d.depends_on_job_id for d in deps]
        statuses = self.session.query(OrchestrationJob.id, OrchestrationJob.status).filter(
            OrchestrationJob.id.in_(dep_ids)
        ).all()
        status_map = {s.id: s.status for s in statuses}
        return all(status_map.get(dep_id) == "completed" for dep_id in dep_ids)

    def promote_ready_orchestration_jobs(self, request_group_id: Optional[int] = None) -> int:
        """Move pending jobs to ready when all dependencies are complete."""
        q = self.session.query(OrchestrationJob).filter(OrchestrationJob.status == "pending")
        if request_group_id is not None:
            q = q.filter(OrchestrationJob.request_group_id == request_group_id)
        jobs = q.all()
        now = datetime.now(timezone.utc)
        promoted = 0
        for job in jobs:
            if self.is_orchestration_job_ready(job.id):
                job.status = "ready"
                job.updated_at = now
                promoted += 1
        if promoted:
            self.session.flush()
        return promoted

    def list_orchestration_jobs(
        self,
        *,
        request_group_id: Optional[int] = None,
        base_run_key: Optional[str] = None,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[OrchestrationJob]:
        q = self.session.query(OrchestrationJob)
        if request_group_id is not None:
            q = q.filter(OrchestrationJob.request_group_id == request_group_id)
        if base_run_key is not None:
            q = q.filter(OrchestrationJob.base_run_key == base_run_key)
        if job_type is not None:
            q = q.filter(OrchestrationJob.job_type == job_type)
        if status is not None:
            q = q.filter(OrchestrationJob.status == status)
        return q.order_by(OrchestrationJob.created_at.asc()).all()
