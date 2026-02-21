"""
Database Connection V2 - Postgres connection for v2 schema.

Handles SQLAlchemy engine creation, session management, and database initialization
for the v2 immutable capture schema. Designed for PostgreSQL.
"""

from sqlalchemy import text

from .models_v2 import BaseV2
from ._base_connection import BaseDatabaseConnection, logger


class DatabaseConnectionV2(BaseDatabaseConnection):
    """V2 database connection with optional pgvector support.

    Usage::

        db = DatabaseConnectionV2("postgresql://user:pass@localhost/dbname")
        db.init_db()

        with db.session_scope() as session:
            ...
    """

    def __init__(
        self,
        connection_string: str,
        echo: bool = False,
        enable_pgvector: bool = True,
    ):
        self.enable_pgvector = enable_pgvector
        super().__init__(
            connection_string,
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def _get_metadata(self):
        return BaseV2.metadata

    def init_db(self) -> None:
        """Create v2 tables and optionally enable pgvector."""
        logger.info("Initializing v2 database tables...")

        if self.enable_pgvector:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                logger.info("pgvector extension enabled (or already exists)")
            except Exception as e:
                logger.warning(
                    "Could not enable pgvector extension: %s. "
                    "Embedding vectors will be stored as JSON.",
                    e,
                )

        BaseV2.metadata.create_all(bind=self.engine)
        logger.info("V2 database tables initialized successfully")
