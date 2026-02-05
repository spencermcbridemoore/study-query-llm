"""
Database Connection V2 - Postgres connection for v2 schema.

Handles SQLAlchemy engine creation, session management, and database initialization
for the v2 immutable capture schema. Designed for PostgreSQL.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator, Optional
from .models_v2 import BaseV2
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseConnectionV2:
    """
    Manages database connections and sessions for v2 schema.
    
    This class provides a unified interface for v2 database operations,
    specifically designed for PostgreSQL with optional pgvector support.
    
    Usage:
        # Initialize connection
        db = DatabaseConnectionV2("postgresql://user:pass@localhost/dbname")
        db.init_db()  # Create tables
        
        # Use session context manager
        with db.session_scope() as session:
            # Perform database operations
            pass
    """

    def __init__(self, connection_string: str, echo: bool = False, enable_pgvector: bool = True):
        """
        Initialize database connection for v2 schema.
        
        Args:
            connection_string: SQLAlchemy connection string (PostgreSQL)
                Example: "postgresql://user:pass@localhost:5432/dbname"
            echo: If True, log all SQL statements (useful for debugging)
            enable_pgvector: If True, attempt to enable pgvector extension
        """
        self.connection_string = connection_string
        self.enable_pgvector = enable_pgvector
        # #region agent log
        import json
        from datetime import datetime, timezone
        try:
            with open(r'c:\Users\spenc\Cursor Repos\study-query-llm\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "pre-fix", "hypothesisId": "B", "location": "connection_v2.py:__init__", "message": "Creating engine with connection pool settings", "data": {"has_ssl": "sslmode" in connection_string.lower()}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        # Add connection pool settings to handle stale SSL connections
        # pool_pre_ping: Test connections before using them (handles stale SSL connections)
        # pool_recycle: Recycle connections after 1 hour to avoid SSL timeouts
        self.engine = create_engine(
            connection_string,
            echo=echo,
            pool_pre_ping=True,  # Test connections before using (handles stale SSL connections)
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Mask password in connection string for logging
        safe_connection_string = connection_string
        if '@' in connection_string and '://' in connection_string:
            parts = connection_string.split('@', 1)
            if ':' in parts[0]:
                user_pass = parts[0].split('://', 1)[1]
                if ':' in user_pass:
                    user = user_pass.split(':')[0]
                    safe_connection_string = connection_string.replace(
                        user_pass, f"{user}:***"
                    )
        
        logger.info(f"Initialized v2 database connection: {safe_connection_string}")

    def init_db(self) -> None:
        """
        Create all v2 database tables.
        
        Also attempts to enable pgvector extension if requested and available.
        This should be called once when setting up the database.
        Safe to call multiple times (won't recreate existing tables).
        """
        logger.info("Initializing v2 database tables...")
        
        # Enable pgvector if requested
        if self.enable_pgvector:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                logger.info("pgvector extension enabled (or already exists)")
            except Exception as e:
                logger.warning(
                    f"Could not enable pgvector extension: {str(e)}. "
                    f"Embedding vectors will be stored as JSON. "
                    f"To enable pgvector, install the extension: "
                    f"CREATE EXTENSION vector;"
                )
        
        BaseV2.metadata.create_all(bind=self.engine)
        logger.info("V2 database tables initialized successfully")

    def drop_all_tables(self) -> None:
        """
        Drop all v2 database tables.
        
        WARNING: This will delete all data! Use with caution.
        Useful for testing or complete database reset.
        """
        BaseV2.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy Session instance
            
        Note:
            You are responsible for closing the session.
            Prefer using session_scope() context manager instead.
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        This context manager handles session creation, commit, rollback,
        and cleanup automatically.
        
        Usage:
            with db.session_scope() as session:
                # Perform operations
                session.add(some_object)
                # Commit happens automatically on exit
                
        If an exception occurs, the transaction is rolled back.
        
        Yields:
            SQLAlchemy Session instance
        """
        # #region agent log
        import json
        from datetime import datetime, timezone
        try:
            with open(r'c:\Users\spenc\Cursor Repos\study-query-llm\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "pre-fix", "hypothesisId": "B", "location": "connection_v2.py:session_scope", "message": "Creating new session", "data": {}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        session = self.get_session()
        try:
            yield session
            session.commit()
            logger.debug("Database transaction committed")
            # #region agent log
            try:
                with open(r'c:\Users\spenc\Cursor Repos\study-query-llm\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "B", "location": "connection_v2.py:session_scope", "message": "Transaction committed successfully", "data": {}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}) + '\n')
            except Exception:
                pass
            # #endregion
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction rolled back: {str(e)}", exc_info=True)
            # #region agent log
            try:
                with open(r'c:\Users\spenc\Cursor Repos\study-query-llm\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "pre-fix", "hypothesisId": "B", "location": "connection_v2.py:session_scope", "message": "Transaction rolled back due to error", "data": {"error_type": type(e).__name__, "error_msg": str(e)[:200]}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}) + '\n')
            except Exception:
                pass
            # #endregion
            raise
        finally:
            session.close()

    def recreate_db(self) -> None:
        """
        Drop and recreate all v2 tables.
        
        WARNING: This will delete all data! Use with caution.
        Useful for testing or complete database reset.
        """
        self.drop_all_tables()
        self.init_db()
