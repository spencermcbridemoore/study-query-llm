"""
Shared helpers for the Panel application.

Provides database/service accessors and UI constants used across view modules.
"""

from typing import Optional

from study_query_llm.config import config, database_connection_summary
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
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
        logger.info(
            "Initializing v2 database connection (%s)",
            database_connection_summary(config.database.connection_string),
        )
        _db_connection = DatabaseConnectionV2(
            config.database.connection_string,
            write_intent=WriteIntent.CANONICAL,
        )
        _db_connection.init_db()
        logger.info("V2 database connection established")
    return _db_connection


def get_database_health_markdown() -> str:
    """
    Human-readable DB status for the Panel sidebar (connection + row counts).

    Helps distinguish a failed connection from an empty Neon/database.
    """
    try:
        db = get_db_connection()
        with db.session_scope() as session:
            from study_query_llm.db.models_v2 import (
                CallArtifact,
                Group,
                ProvenancedRun,
                RawCall,
            )
            from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN

            n_groups = session.query(Group).count()
            n_raw = session.query(RawCall).count()
            n_provenanced = session.query(ProvenancedRun).count()
            n_mcq_groups = (
                session.query(Group)
                .filter(Group.group_type == GROUP_TYPE_MCQ_RUN)
                .count()
            )
            n_artifacts = session.query(CallArtifact).count()

        target = database_connection_summary(config.database.connection_string)
        any_activity = (
            n_groups > 0
            or n_raw > 0
            or n_provenanced > 0
            or n_mcq_groups > 0
            or n_artifacts > 0
        )
        empty_note = ""
        if not any_activity:
            empty_note = (
                "\n\n**Note:** The Inference tab lists deployments from your **provider API**, "
                "not from these tables. The **Storage / DB stats** tab adds catalog sizes and "
                "artifact/Azure probes. Other tabs only show rows **written to this** `DATABASE_URL`. "
                "If you expected data, point `DATABASE_URL` at the same Postgres where you run workers "
                "or ingest sweeps."
            )
        semantics = (
            "\n\n_**`raw_calls`** = chat-style inference rows (e.g. Panel Inference with DB persistence). "
            "**MCQ** probe calls are not stored per-call in `raw_calls`; runs appear as **`mcq_run` groups** "
            "and **`provenanced_runs`**. **Artifact rows** = `call_artifacts` table._"
        )
        return (
            "### Database\n\n"
            "**Status:** connected  \n"
            f"**Target:** `{target}`  \n"
            f"**Groups:** {n_groups:,} &nbsp;·&nbsp; **Raw calls:** {n_raw:,} &nbsp;·&nbsp; "
            f"**Provenanced runs:** {n_provenanced:,} &nbsp;·&nbsp; **MCQ run groups:** {n_mcq_groups:,} "
            f"&nbsp;·&nbsp; **Artifact rows:** {n_artifacts:,}  \n\n"
            "_If every count above is zero, this database has no v2 activity in those tables yet._"
            f"{semantics}"
            f"{empty_note}"
        )
    except Exception as exc:
        logger.exception("Database health check failed")
        return (
            "### Database\n\n"
            "**Status:** **error** (see below)  \n\n"
            f"```\n{exc!s}\n```\n\n"
            "_Check `DATABASE_URL`, `.env` in the repo root, and server logs._"
        )


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
