"""
Shared helpers for the Panel application.

Provides database/service accessors and UI constants used across view modules.
"""

from typing import Optional

from study_query_llm.config import config
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.study_service import StudyService
from study_query_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# UI constants
HEADER_BG = "#2596be"
HEADER_FG = "#FFFFFF"

# Global state for database and services
_db_connection: Optional[DatabaseConnectionV2] = None
_inference_service: Optional[InferenceService] = None
_inference_service_provider: Optional[str] = None
_study_service: Optional[StudyService] = None


def get_db_connection() -> DatabaseConnectionV2:
    """Get or create v2 database connection."""
    global _db_connection
    if _db_connection is None:
        logger.info("Initializing v2 database connection...")
        _db_connection = DatabaseConnectionV2(config.database.connection_string)
        _db_connection.init_db()
        logger.info("V2 database connection established")
    return _db_connection


def get_inference_service(provider_name: str, deployment_name: Optional[str] = None) -> InferenceService:
    """
    Get or create inference service for a provider.

    Args:
        provider_name: Name of the provider ('azure', 'openai', etc.)
        deployment_name: Optional deployment name to use (for Azure). If provided,
                        updates the config before creating the provider.
    """
    global _inference_service, _inference_service_provider

    if deployment_name and provider_name == "azure":
        if hasattr(config, '_provider_configs') and 'azure' in config._provider_configs:
            del config._provider_configs['azure']
        azure_config = config.get_provider_config("azure")
        azure_config.deployment_name = deployment_name

    factory = ProviderFactory()
    provider = factory.create_from_config(provider_name)

    db = get_db_connection()
    session = db.get_session()
    repository = RawCallRepository(session)

    service = InferenceService(provider, repository=repository)
    _inference_service_provider = provider_name

    return service


def get_study_service() -> StudyService:
    """Get or create study service for analytics."""
    global _study_service
    if _study_service is None:
        db = get_db_connection()
        session = db.get_session()
        repository = RawCallRepository(session)
        _study_service = StudyService(repository)
    return _study_service
