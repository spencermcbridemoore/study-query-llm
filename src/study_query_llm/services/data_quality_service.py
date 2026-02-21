"""
Data Quality Service - Business logic for managing defective data labels.

This service handles the business logic for identifying and managing
defective data labels, separating concerns from the repository layer.
"""

from typing import TYPE_CHECKING
from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


class DataQualityService:
    """
    Service for managing data quality labels and defective data identification.
    
    This service encapsulates business logic for marking and checking
    defective data, keeping the repository focused on data access.
    
    Usage:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            quality_service = DataQualityService(repo)
            
            # Check if a call is defective
            is_defective = quality_service.is_call_defective(call_id)
            
            # Get or create the defective group
            group_id = quality_service.get_or_create_defective_group()
    """

    def __init__(self, repository: "RawCallRepository"):
        """
        Initialize the data quality service.
        
        Args:
            repository: RawCallRepository instance for database access
        """
        self.repository = repository

    def get_or_create_defective_group(self) -> int:
        """
        Get or create the standard "defective_data" group.
        
        Convention: Group with name "defective_data" and type "label" is used
        to mark RawCall records as defective/excluded from analysis.
        
        Returns:
            The ID of the defective_data group
        """
        from ..db.models_v2 import Group
        
        # Try to find existing group
        existing = self.repository.session.query(Group).filter_by(
            group_type="label",
            name="defective_data"
        ).first()
        
        if existing:
            return existing.id
        
        # Create if doesn't exist
        return self.repository.create_group(
            group_type="label",
            name="defective_data",
            description="RawCall records marked as defective/excluded from analysis"
        )

    def is_call_defective(self, call_id: int) -> bool:
        """
        Check if a raw call is marked as defective.
        
        Args:
            call_id: ID of the raw call to check
        
        Returns:
            True if call is in the "defective_data" group, False otherwise
        """
        group_id = self.get_or_create_defective_group()
        groups = self.repository.get_groups_for_call(call_id)
        return any(g.id == group_id for g in groups)
