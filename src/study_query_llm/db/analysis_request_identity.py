"""Single source of truth for the ``analysis_request`` identity contract.

Identity is ``(method_name, input_id, run_key)`` extracted from the JSON
``groups.metadata_json`` column, scoped to ``group_type='analysis_request'`` and
**NULL-distinct**: two rows collide only when all three fields are present and
equal. Rows with any missing field (the empty-metadata "container" shape, e.g.
``mcq_logprob`` / ``ingest_file`` groups) never collide and are never treated as
duplicates.

Four sites must agree on this contract, and historically drifted (a Postgres-only
probe shipped a NULL-handling bug the SQLite path never exercised):

* the partial UNIQUE functional index DDL (Postgres + SQLite) in ``models_v2``;
* the migration's duplicate-probe SQL
  (``db/migrations/add_analysis_request_unique_index.py``);
* the remediation finder
  (``scripts/remediate_analysis_request_duplicates.find_duplicate_identities``).

This module is the shared contract: the field list, the per-dialect JSON-extract
fragment, the index/probe SQL builders, and the Python extractor. It imports
nothing from the rest of the package (no import cycle with ``models_v2``).

A fifth consumer -- ``RawCallRepository.insert_or_get_analysis_request_group`` --
builds a metadata-equality *lookup* dict (keeping ``input_id`` as ``int`` to match
stored JSON values). That lookup is intentionally left on its own coercion: it
matches stored JSON, not the text extraction the index/probe use. Reconciling the
stored type is a separate, data-affecting change, out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

ANALYSIS_REQUEST_GROUP_TYPE: str = "analysis_request"
ANALYSIS_REQUEST_UNIQUE_INDEX_NAME: str = "uq_groups_analysis_request_identity"
IDENTITY_FIELDS: tuple[str, ...] = ("method_name", "input_id", "run_key")

_POSTGRES_DIALECTS = frozenset({"postgresql", "postgres"})
_SQLITE_DIALECTS = frozenset({"sqlite"})


def json_extract_sql(field: str, dialect: str) -> str:
    """Return the per-dialect SQL expression that extracts one identity field.

    Postgres uses the ``->>`` text operator; SQLite uses ``json_extract``. Both
    yield the field as a scalar so the index/probe compare like-for-like.
    """
    normalized = str(dialect or "").strip().lower()
    if normalized in _POSTGRES_DIALECTS:
        return f"(metadata_json ->> '{field}')"
    if normalized in _SQLITE_DIALECTS:
        return f"json_extract(metadata_json, '$.{field}')"
    raise ValueError(f"unsupported_dialect:{dialect!r}")


def build_unique_index_sql(dialect: str) -> str:
    """Build the partial UNIQUE functional index DDL for one dialect.

    The rendering is byte-stable and pinned by a regression test: the canonical
    Postgres index was created from this exact string, so it must not drift.
    """
    columns = ", ".join(json_extract_sql(field, dialect) for field in IDENTITY_FIELDS)
    return (
        f"CREATE UNIQUE INDEX IF NOT EXISTS {ANALYSIS_REQUEST_UNIQUE_INDEX_NAME} "
        f"ON groups ({columns}) "
        f"WHERE group_type = '{ANALYSIS_REQUEST_GROUP_TYPE}'"
    )


def build_duplicate_probe_sql(dialect: str) -> str:
    """Build the duplicate-identity probe query for one dialect.

    Finds identities shared by more than one ``analysis_request`` group. The
    ``IS NOT NULL`` filter is load-bearing: the UNIQUE index is NULL-distinct, so
    container/half rows must be excluded or ``GROUP BY`` would fold them into a
    single false-positive bucket (the bug that motivated this consolidation).
    """
    inner = ",\n               ".join(
        f"{json_extract_sql(field, dialect)} AS {field}" for field in IDENTITY_FIELDS
    )
    not_null = "\n          AND ".join(
        f"{field} IS NOT NULL" for field in IDENTITY_FIELDS
    )
    fields = ", ".join(IDENTITY_FIELDS)
    return (
        f"\n    SELECT {fields}, COUNT(*) AS n, MIN(id) AS keeper_id"
        f"\n    FROM ("
        f"\n        SELECT id,"
        f"\n               {inner}"
        f"\n        FROM groups"
        f"\n        WHERE group_type = '{ANALYSIS_REQUEST_GROUP_TYPE}'"
        f"\n    ) AS extracted"
        f"\n    WHERE {not_null}"
        f"\n    GROUP BY {fields}"
        f"\n    HAVING COUNT(*) > 1"
        f"\n    ORDER BY n DESC, {fields}\n"
    )


def extract_identity(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[tuple[str, ...]]:
    """Return the canonical identity tuple, or ``None`` for a non-identity row.

    Mirrors the NULL-distinct rule: any missing/``None`` field yields ``None`` (a
    container/half row the index and probe also ignore). Present values are
    str-coerced to match the text semantics of the SQL extraction.
    """
    if not metadata:
        return None
    values: list[str] = []
    for field in IDENTITY_FIELDS:
        value = metadata.get(field)
        if value is None:
            return None
        values.append(str(value))
    return tuple(values)


__all__ = [
    "ANALYSIS_REQUEST_GROUP_TYPE",
    "ANALYSIS_REQUEST_UNIQUE_INDEX_NAME",
    "IDENTITY_FIELDS",
    "json_extract_sql",
    "build_unique_index_sql",
    "build_duplicate_probe_sql",
    "extract_identity",
]
