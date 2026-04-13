#!/usr/bin/env python3
"""Reusable DB target parsing and safety helpers for scripts."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlparse, urlunparse

LOOPBACK_HOSTS = {"127.0.0.1", "::1"}


@dataclass(frozen=True)
class DbTarget:
    """Normalized PostgreSQL target details."""

    scheme: str
    host: str
    port: int
    dbname: str
    username: str

    @property
    def host_port(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def summary(self) -> str:
        user = f"{self.username}@" if self.username else ""
        return f"{user}{self.host}:{self.port}/{self.dbname}"


def normalize_host(host: str | None) -> str:
    """Normalize host aliases for safer comparisons."""
    h = (host or "").strip().lower()
    if h == "localhost":
        return "127.0.0.1"
    return h


def parse_postgres_target(url: str) -> DbTarget:
    """Parse and validate a PostgreSQL URL into a normalized target."""
    parsed = urlparse((url or "").strip())
    scheme = (parsed.scheme or "").lower()
    if not scheme.startswith("postgres"):
        raise ValueError("Expected a PostgreSQL URL (scheme must start with 'postgres').")
    dbname = (parsed.path or "").lstrip("/")
    if not dbname:
        raise ValueError("Database URL must include a database name in the path.")
    return DbTarget(
        scheme=scheme,
        host=normalize_host(parsed.hostname),
        port=int(parsed.port or 5432),
        dbname=dbname,
        username=unquote(parsed.username) if parsed.username else "",
    )


def is_loopback_target(url: str) -> bool:
    """Return True when URL host resolves to loopback host aliases."""
    target = parse_postgres_target(url)
    return target.host in LOOPBACK_HOSTS


def same_db_target(url_a: str, url_b: str, *, include_dbname: bool = True) -> bool:
    """Return True when two URLs resolve to the same target endpoint."""
    a = parse_postgres_target(url_a)
    b = parse_postgres_target(url_b)
    if include_dbname:
        return (a.host, a.port, a.dbname) == (b.host, b.port, b.dbname)
    return (a.host, a.port) == (b.host, b.port)


def redact_database_url(url: str) -> str:
    """Return printable URL with password redacted."""
    try:
        parsed = urlparse(url)
        if parsed.password is None:
            return url
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        if parsed.username:
            netloc = f"{parsed.username}:***@{netloc}"
        else:
            netloc = f"***@{netloc}"
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
