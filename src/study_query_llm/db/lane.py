"""Database lane resolution primitives.

This module introduces a role-based lane model used by the DB guardrail
rollout. It is intentionally side-effect free and can be imported by
connection layers, scripts, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from urllib.parse import urlparse

CANONICAL_DATABASE_URL_ENV = "CANONICAL_DATABASE_URL"
JETSTREAM_DATABASE_URL_ENV = "JETSTREAM_DATABASE_URL"
LOCAL_DATABASE_URL_ENV = "LOCAL_DATABASE_URL"


class Lane(StrEnum):
    """Resolved lane for a connection string."""

    CANONICAL = "canonical"
    LOCAL_POSTGRES = "local_postgres"
    SQLITE_FILE = "sqlite_file"
    SQLITE_MEMORY = "sqlite_memory"
    UNKNOWN = "unknown"


class CanonicalIdentityConflict(RuntimeError):
    """Raised when canonical role and identity URLs disagree."""


class LaneResolutionError(RuntimeError):
    """Raised when lane resolution must fail closed."""


@dataclass(frozen=True)
class CanonicalTarget:
    """Resolved canonical target and provenance metadata."""

    url: str
    source_var: str
    identity: str


@dataclass(frozen=True)
class _PostgresTarget:
    host: str
    port: int
    dbname: str


def _normalize_host(host: str | None) -> str:
    """Normalize host aliases for stable target comparisons."""
    normalized = (host or "").strip().lower()
    if normalized == "localhost":
        return "127.0.0.1"
    return normalized


def _parse_postgres_target(url: str) -> _PostgresTarget:
    """Parse host/port/dbname from a PostgreSQL URL."""
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


def same_db_target(url_a: str, url_b: str) -> bool:
    """Return True when two PostgreSQL URLs resolve to same host/port/dbname."""
    a = _parse_postgres_target(url_a)
    b = _parse_postgres_target(url_b)
    return (a.host, a.port, a.dbname) == (b.host, b.port, b.dbname)


def is_loopback_target(url: str) -> bool:
    """Return True when URL points at localhost/loopback PostgreSQL target."""
    target = _parse_postgres_target(url)
    return target.host in {"127.0.0.1", "::1"}


def resolve_canonical_target(
    *,
    canonical_database_url: str | None = None,
    jetstream_database_url: str | None = None,
    identity_label: str = "Jetstream",
) -> CanonicalTarget | None:
    """Resolve canonical target with role/identity fallback and conflict check.

    Resolution order:
      1) CANONICAL_DATABASE_URL
      2) JETSTREAM_DATABASE_URL (back-compat identity alias)

    If both are set and resolve to different targets, fail closed by raising
    ``CanonicalIdentityConflict``.
    """

    canonical_url = (
        canonical_database_url
        if canonical_database_url is not None
        else os.environ.get(CANONICAL_DATABASE_URL_ENV)
    )
    jetstream_url = (
        jetstream_database_url
        if jetstream_database_url is not None
        else os.environ.get(JETSTREAM_DATABASE_URL_ENV)
    )

    canonical_url = (canonical_url or "").strip()
    jetstream_url = (jetstream_url or "").strip()

    if canonical_url and jetstream_url:
        if not same_db_target(canonical_url, jetstream_url):
            raise CanonicalIdentityConflict(
                f"{CANONICAL_DATABASE_URL_ENV} and {JETSTREAM_DATABASE_URL_ENV} "
                "resolve to different targets. Unset one or make them match."
            )
        return CanonicalTarget(
            url=canonical_url,
            source_var=CANONICAL_DATABASE_URL_ENV,
            identity=identity_label,
        )

    if canonical_url:
        return CanonicalTarget(
            url=canonical_url,
            source_var=CANONICAL_DATABASE_URL_ENV,
            identity=identity_label,
        )

    if jetstream_url:
        return CanonicalTarget(
            url=jetstream_url,
            source_var=JETSTREAM_DATABASE_URL_ENV,
            identity=identity_label,
        )

    return None


def resolve_lane(
    connection_string: str,
    *,
    canonical_database_url: str | None = None,
    jetstream_database_url: str | None = None,
    local_database_url: str | None = None,
) -> Lane:
    """Resolve the lane for a DB URL.

    - sqlite memory URLs -> ``Lane.SQLITE_MEMORY``
    - sqlite file URLs -> ``Lane.SQLITE_FILE``
    - postgres URL matching canonical target -> ``Lane.CANONICAL``
    - postgres URL matching local URL or loopback target -> ``Lane.LOCAL_POSTGRES``
    - otherwise -> ``Lane.UNKNOWN``
    """

    raw = (connection_string or "").strip()
    if not raw:
        return Lane.UNKNOWN

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme.startswith("sqlite"):
        database_path = parsed.path or ""
        if database_path in {"/:memory:", ":memory:"}:
            return Lane.SQLITE_MEMORY
        return Lane.SQLITE_FILE

    try:
        _parse_postgres_target(raw)
    except ValueError:
        return Lane.UNKNOWN

    canonical_target = resolve_canonical_target(
        canonical_database_url=canonical_database_url,
        jetstream_database_url=jetstream_database_url,
    )
    if canonical_target and same_db_target(raw, canonical_target.url):
        return Lane.CANONICAL

    local_url = (
        local_database_url
        if local_database_url is not None
        else os.environ.get(LOCAL_DATABASE_URL_ENV)
    )
    local_url = (local_url or "").strip()
    if local_url:
        try:
            if same_db_target(raw, local_url):
                return Lane.LOCAL_POSTGRES
        except ValueError:
            # If LOCAL_DATABASE_URL is malformed/non-postgres, ignore it here.
            pass

    try:
        if is_loopback_target(raw):
            return Lane.LOCAL_POSTGRES
    except ValueError:
        return Lane.UNKNOWN

    return Lane.UNKNOWN
