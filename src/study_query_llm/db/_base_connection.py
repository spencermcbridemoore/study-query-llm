"""Base database connection with shared session/engine management."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Generator
from urllib.parse import urlparse, urlunparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

_DESTRUCTIVE_DDL_OVERRIDE_ENV = "SQLLM_ALLOW_DESTRUCTIVE_DDL"
_JETSTREAM_DATABASE_URL_ENV = "JETSTREAM_DATABASE_URL"


@dataclass(frozen=True)
class _PostgresTarget:
    host: str
    port: int
    dbname: str


def _normalize_host(host: str | None) -> str:
    """Normalize host aliases for consistent target comparisons."""
    normalized = (host or "").strip().lower()
    if normalized == "localhost":
        return "127.0.0.1"
    return normalized


def _parse_postgres_target(url: str) -> _PostgresTarget:
    """Parse and normalize host/port/dbname from a PostgreSQL URL."""
    parsed = urlparse((url or "").strip())
    scheme = (parsed.scheme or "").lower()
    if not scheme.startswith("postgres"):
        raise ValueError("Expected a PostgreSQL URL (scheme must start with 'postgres').")
    dbname = (parsed.path or "").lstrip("/")
    if not dbname:
        raise ValueError("Database URL must include a database name in the path.")
    return _PostgresTarget(
        host=_normalize_host(parsed.hostname),
        port=int(parsed.port or 5432),
        dbname=dbname,
    )


def _same_postgres_target(url_a: str, url_b: str) -> bool:
    """Return True when URLs resolve to the same host/port/dbname target."""
    a = _parse_postgres_target(url_a)
    b = _parse_postgres_target(url_b)
    return (a.host, a.port, a.dbname) == (b.host, b.port, b.dbname)


def _format_host_for_url(host: str) -> str:
    """Return host text suitable for URL netloc rendering."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _mask_password(connection_string: str) -> str:
    """Return *connection_string* with password replaced by ``***``."""
    try:
        parsed = urlparse(connection_string)
        if parsed.password is None:
            return connection_string
        host = _format_host_for_url(parsed.hostname or "")
        if parsed.port:
            host = f"{host}:{parsed.port}"
        if parsed.username:
            netloc = f"{parsed.username}:***@{host}"
        else:
            netloc = f"***@{host}"
        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    except Exception:
        return "***"


def _is_sqlite_url(connection_string: str) -> bool:
    """Return True when the SQLAlchemy URL scheme is sqlite."""
    scheme = (urlparse((connection_string or "").strip()).scheme or "").lower()
    return scheme.startswith("sqlite")


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

    def _assert_destructive_operation_allowed(self, operation_name: str) -> None:
        """Enforce policy guardrails before destructive DDL operations."""
        target_url = (self.connection_string or "").strip()

        if _is_sqlite_url(target_url):
            return

        if os.environ.get(_DESTRUCTIVE_DDL_OVERRIDE_ENV) != "1":
            raise RuntimeError(
                f"Refusing destructive operation '{operation_name}' for non-sqlite target "
                f"{_mask_password(target_url)}. Set {_DESTRUCTIVE_DDL_OVERRIDE_ENV}=1 "
                "to allow this operation."
            )

        jetstream_url = (os.environ.get(_JETSTREAM_DATABASE_URL_ENV) or "").strip()
        if not jetstream_url:
            return

        try:
            _parse_postgres_target(jetstream_url)
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing destructive operation '{operation_name}' because "
                f"{_JETSTREAM_DATABASE_URL_ENV} is set but invalid/non-postgres. "
                f"Fix or unset {_JETSTREAM_DATABASE_URL_ENV} before retrying."
            ) from exc

        try:
            matches_jetstream = _same_postgres_target(target_url, jetstream_url)
        except ValueError:
            # Non-Postgres target with explicit override remains operator-managed.
            return

        if matches_jetstream:
            raise RuntimeError(
                f"Refusing destructive operation '{operation_name}' for "
                f"{_mask_password(target_url)} because it matches "
                f"{_JETSTREAM_DATABASE_URL_ENV} ({_mask_password(jetstream_url)}). "
                "This hard-stop is non-overridable."
            )

    def drop_all_tables(self) -> None:
        """Drop all tables. **Destroys all data.**"""
        self._assert_destructive_operation_allowed("drop_all_tables")
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
