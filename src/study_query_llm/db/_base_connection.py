"""Base database connection with shared session/engine management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def _mask_password(connection_string: str) -> str:
    """Return *connection_string* with the password replaced by ``***``."""
    if "@" in connection_string and "://" in connection_string:
        parts = connection_string.split("@", 1)
        if ":" in parts[0]:
            user_pass = parts[0].split("://", 1)[1]
            if ":" in user_pass:
                user = user_pass.split(":")[0]
                return connection_string.replace(user_pass, f"{user}:***")
    return connection_string


class BaseDatabaseConnection:
    """Shared engine, session, and lifecycle management.

    Subclasses supply a SQLAlchemy ``Base.metadata`` via :pymethod:`_get_metadata`
    and may override :pymethod:`init_db` for schema-specific setup (e.g. pgvector).
    """

    def __init__(self, connection_string: str, *, echo: bool = False, **engine_kwargs):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=echo, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info(
            "Initialized database connection: %s", _mask_password(connection_string)
        )

    # ------------------------------------------------------------------
    # Abstract hook
    # ------------------------------------------------------------------

    def _get_metadata(self):
        """Return the ``MetaData`` object for this schema."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables (idempotent)."""
        logger.info("Initializing database tables...")
        self._get_metadata().create_all(bind=self.engine)
        logger.info("Database tables initialized successfully")

    def drop_all_tables(self) -> None:
        """Drop all tables. **Destroys all data.**"""
        self._get_metadata().drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Return a new session. Prefer :pymethod:`session_scope` instead."""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Transactional scope: auto-commit on success, rollback on error."""
        session = self.get_session()
        try:
            yield session
            session.commit()
            logger.debug("Database transaction committed")
        except Exception as e:
            session.rollback()
            logger.error("Database transaction rolled back: %s", e, exc_info=True)
            raise
        finally:
            session.close()

    def recreate_db(self) -> None:
        """Drop and recreate all tables. **Destroys all data.**"""
        self.drop_all_tables()
        self.init_db()
