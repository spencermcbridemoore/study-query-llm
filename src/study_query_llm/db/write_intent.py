"""Write-intent primitives for DB lane guardrails."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

from .lane import Lane, resolve_lane


class WriteIntent(StrEnum):
    """Declared purpose for a DB session."""

    CANONICAL = "canonical"
    READ_MIRROR = "read_mirror"
    SANDBOX = "sandbox"


class LaneIntentMismatch(RuntimeError):
    """Raised when declared write intent does not match resolved lane."""


_ALLOWED_LANES_BY_INTENT: Mapping[WriteIntent, set[Lane]] = {
    WriteIntent.CANONICAL: {Lane.CANONICAL, Lane.UNKNOWN},
    WriteIntent.READ_MIRROR: {Lane.LOCAL_POSTGRES, Lane.SQLITE_FILE},
    WriteIntent.SANDBOX: {Lane.LOCAL_POSTGRES, Lane.SQLITE_FILE, Lane.SQLITE_MEMORY},
}


def parse_write_intent(raw: str | None) -> WriteIntent:
    """Parse a user-provided write-intent string.

    Accepted values are case-insensitive enum values:
    ``canonical``, ``read_mirror``, ``sandbox``.
    """

    value = (raw or "").strip().lower()
    if not value:
        raise ValueError("Write intent is empty.")
    try:
        return WriteIntent(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(intent.value for intent in WriteIntent))
        raise ValueError(
            f"Invalid write intent '{raw}'. Expected one of: {allowed}."
        ) from exc


def allowed_lanes_for_intent(intent: WriteIntent) -> set[Lane]:
    """Return allowed lanes for an intent."""
    return set(_ALLOWED_LANES_BY_INTENT[intent])


def default_write_intent_for_connection(connection_string: str) -> WriteIntent:
    """Return a practical default intent for a concrete DB URL."""
    lane = resolve_lane(connection_string)
    if lane in {Lane.SQLITE_FILE, Lane.SQLITE_MEMORY}:
        return WriteIntent.SANDBOX
    if lane is Lane.LOCAL_POSTGRES:
        return WriteIntent.READ_MIRROR
    return WriteIntent.CANONICAL


def assert_intent_matches_lane(intent: WriteIntent, lane: Lane) -> None:
    """Raise when declared intent is incompatible with resolved lane."""
    allowed = _ALLOWED_LANES_BY_INTENT[intent]
    if lane not in allowed:
        allowed_display = ", ".join(sorted(item.name for item in allowed))
        raise LaneIntentMismatch(
            f"Intent {intent.name} is incompatible with lane {lane.name}. "
            f"Allowed lanes: {allowed_display}."
        )
