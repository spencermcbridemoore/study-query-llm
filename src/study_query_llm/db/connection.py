"""
Database Connection Management.

Handles SQLAlchemy engine creation, session management, and database initialization.
Supports both SQLite (development) and PostgreSQL (production).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from .models import Base
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Manages database connections and sessions.
    
    This class provides a unified interface for database operations,
    supporting both SQLite (for development) and PostgreSQL (for production).
    
    Usage:
        # Initialize connection
        db = DatabaseConnection("postgresql://user:pass@localhost/dbname")
        db.init_db()  # Create tables
        
        # Use session context manager
        with db.session_scope() as session:
            # Perform database operations
            pass
    """

    def __init__(self, connection_string: str, echo: bool = False):
        """
        Initialize database connection.
        
        Args:
            connection_string: SQLAlchemy connection string
                Examples:
                - PostgreSQL: "postgresql://user:pass@localhost:5432/dbname"
                - SQLite: "sqlite:///path/to/db.sqlite"
                - SQLite in-memory: "sqlite:///:memory:"
            echo: If True, log all SQL statements (useful for debugging)
        """
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=echo)
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
        
        logger.info(f"Initialized database connection: {safe_connection_string}")

    def init_db(self) -> None:
        """
        Create all database tables.
        
        This should be called once when setting up the database.
        Safe to call multiple times (won't recreate existing tables).
        """
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables initialized successfully")

    def drop_all_tables(self) -> None:
        """
        Drop all database tables.
        
        WARNING: This will delete all data! Use with caution.
        Useful for testing or complete database reset.
        """
        Base.metadata.drop_all(bind=self.engine)

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
        session = self.get_session()
        try:
            yield session
            session.commit()
            logger.debug("Database transaction committed")
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction rolled back: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()

    def recreate_db(self) -> None:
        """
        Drop and recreate all tables.
        
        WARNING: This will delete all data! Use with caution.
        Useful for testing or complete database reset.
        """
        self.drop_all_tables()
        self.init_db()

