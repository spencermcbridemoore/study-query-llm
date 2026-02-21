"""
Raw Call Repository - Database operations for v2 immutable capture schema.

This repository handles all database interactions for storing and querying
raw calls in the v2 schema. Uses the Repository pattern to abstract database details.
"""

from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, cast, Float, String
from .models_v2 import RawCall, Group, GroupMember, CallArtifact, EmbeddingVector, GroupLink
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
            # PostgreSQL path: Use JSON operators for efficient extraction
            # Extract total_tokens from tokens_json, trying multiple field names
            token_expr = func.coalesce(
                cast(RawCall.tokens_json['total_tokens'].astext, Float),
                cast(RawCall.tokens_json['totalTokens'].astext, Float),
                cast(RawCall.tokens_json['usage']['total_tokens'].astext, Float),
                cast(RawCall.tokens_json['usage']['totalTokens'].astext, Float),
                0.0
            )
            
            # Aggregate by provider using SQL GROUP BY
            query = self.session.query(
                RawCall.provider,
                func.count(RawCall.id).label('count'),
                func.avg(token_expr).label('avg_tokens'),
                func.avg(RawCall.latency_ms).label('avg_latency_ms'),
                func.sum(token_expr).label('total_tokens')
            ).group_by(RawCall.provider)
            
            results = []
            for row in query.all():
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
        # Check if link already exists
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

        self.session.add(link)
        self.session.flush()
        self.session.refresh(link)

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
            link_type="step",
        ).order_by(GroupLink.position, GroupLink.created_at).all()

        step_ids = [link.child_group_id for link in links]
        if not step_ids:
            return []

        # Get groups and preserve order
        groups = self.session.query(Group).filter(Group.id.in_(step_ids)).all()
        group_dict = {g.id: g for g in groups}

        # Return in link order
        return [group_dict[step_id] for step_id in step_ids if step_id in group_dict]
