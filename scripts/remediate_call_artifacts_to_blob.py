#!/usr/bin/env python3
"""Remediate local-path call_artifacts URIs into Azure blob URIs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.lane import Lane
from study_query_llm.db.models_v2 import CallArtifact, RawCall
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.services.artifact_service import ArtifactService

DEFAULT_CONSTRAINT_NAME = "call_artifacts_uri_must_be_blob"


def _is_azure_blob_uri(uri: str) -> bool:
    parsed = urlparse((uri or "").strip())
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return ".blob.core.windows.net" in host


def _resolve_database_url(explicit_url: str | None, env_var: str) -> str:
    if explicit_url:
        return explicit_url.strip()
    for key in (env_var, "DATABASE_URL", "JETSTREAM_DATABASE_URL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    raise ValueError(
        f"No database URL found. Set --database-url, {env_var}, DATABASE_URL, or JETSTREAM_DATABASE_URL."
    )


def _validate_constraint_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", candidate):
        raise ValueError(f"Invalid constraint name: {name!r}")
    return candidate


def _sanitize_segment(raw: str) -> str:
    return str(raw).replace("..", "_").replace("/", "_").replace("\\", "_").strip() or "artifact"


def _local_path_from_uri(uri: str) -> Path | None:
    value = (uri or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path or ""))
    return Path(value)


def _derive_logical_path(artifact: CallArtifact, source_path: Path) -> str:
    meta = dict(artifact.metadata_json or {})
    group_id = meta.get("group_id")
    step_name = meta.get("step_name")
    logical_filename = meta.get("logical_filename")
    if group_id is not None and step_name and logical_filename:
        return (
            f"{int(group_id)}/"
            f"{_sanitize_segment(str(step_name))}/"
            f"{_sanitize_segment(str(logical_filename))}"
        )

    ext = source_path.suffix or ".bin"
    filename = source_path.name or f"{_sanitize_segment(artifact.artifact_type)}{ext}"

    if group_id is not None:
        return (
            f"{int(group_id)}/legacy_remediation/"
            f"{_sanitize_segment(filename)}"
        )
    return (
        f"legacy_remediation/"
        f"{int(artifact.id)}_{_sanitize_segment(filename)}"
    )


def _constraint_validated(conn, constraint_name: str) -> bool | None:
    row = conn.execute(
        text(
            """
            SELECT c.convalidated
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = 'public'
              AND t.relname = 'call_artifacts'
              AND c.conname = :constraint_name
            LIMIT 1
            """
        ),
        {"constraint_name": constraint_name},
    ).fetchone()
    if row is None:
        return None
    return bool(row[0])


def _replace_uri_values(payload: object, *, old_uri: str, new_uri: str) -> tuple[object, bool]:
    if isinstance(payload, str):
        if payload == old_uri:
            return new_uri, True
        return payload, False
    if isinstance(payload, list):
        changed = False
        updated_items = []
        for item in payload:
            updated_item, item_changed = _replace_uri_values(
                item,
                old_uri=old_uri,
                new_uri=new_uri,
            )
            updated_items.append(updated_item)
            changed = changed or item_changed
        return updated_items, changed
    if isinstance(payload, dict):
        changed = False
        updated_dict: dict[object, object] = {}
        for key, value in payload.items():
            updated_value, value_changed = _replace_uri_values(
                value,
                old_uri=old_uri,
                new_uri=new_uri,
            )
            updated_dict[key] = updated_value
            changed = changed or value_changed
        return updated_dict, changed
    return payload, False


def _sync_raw_call_uri_mirror(session, *, apply: bool) -> tuple[int, int]:
    candidates = 0
    updated = 0
    artifacts = session.query(CallArtifact).order_by(CallArtifact.id.asc()).all()
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        remediation = metadata.get("remediation")
        if not isinstance(remediation, dict):
            continue
        previous_uri = str(remediation.get("previous_uri") or "").strip()
        current_uri = str(artifact.uri or "").strip()
        if not previous_uri or not current_uri or previous_uri == current_uri:
            continue
        raw_call = (
            session.query(RawCall).filter(RawCall.id == int(artifact.call_id)).first()
            if artifact.call_id is not None
            else None
        )
        if raw_call is None:
            continue
        payload = raw_call.response_json
        updated_payload, changed = _replace_uri_values(
            payload,
            old_uri=previous_uri,
            new_uri=current_uri,
        )
        if not changed:
            continue
        candidates += 1
        if apply:
            raw_call.response_json = updated_payload
            updated += 1
    return candidates, updated


def _validate_constraint(engine, constraint_name: str) -> bool:
    with engine.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE call_artifacts VALIDATE CONSTRAINT {constraint_name}"
            )
        )
        validated = _constraint_validated(conn, constraint_name)
    return bool(validated)


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Re-upload local-path call_artifacts URIs to Azure blob and update DB rows."
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Explicit target database URL.",
    )
    parser.add_argument(
        "--env-var",
        type=str,
        default="CANONICAL_DATABASE_URL",
        help="Primary env var used to resolve the target URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max polluted rows to process (0=all).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply remediation updates (default is dry-run).",
    )
    parser.add_argument(
        "--validate-constraint",
        action="store_true",
        help=f"Run VALIDATE CONSTRAINT {DEFAULT_CONSTRAINT_NAME} after remediation.",
    )
    parser.add_argument(
        "--constraint-name",
        type=str,
        default=DEFAULT_CONSTRAINT_NAME,
        help="Constraint name used for optional validation.",
    )
    args = parser.parse_args()

    try:
        resolved_url = _resolve_database_url(args.database_url, args.env_var)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    constraint_name = _validate_constraint_name(args.constraint_name)
    db = DatabaseConnectionV2(
        resolved_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    if db.engine.dialect.name != "postgresql":
        print("ERROR: remediation is Postgres-only.")
        return 2
    if db.lane is not Lane.CANONICAL:
        print(f"ERROR: refusing remediation on non-canonical lane {db.lane.name}.")
        return 2

    with db.session_scope() as session:
        artifact_service = ArtifactService()
        backend_type = getattr(artifact_service.storage, "backend_type", "unknown")
        print(f"storage_backend={backend_type}")
        if backend_type != "azure_blob":
            raise RuntimeError(
                "Remediation requires ARTIFACT_STORAGE_BACKEND=azure_blob."
            )

        rows = session.query(CallArtifact).order_by(CallArtifact.id.asc()).all()
        polluted = [row for row in rows if not _is_azure_blob_uri(str(row.uri or ""))]
        if args.limit > 0:
            polluted = polluted[: int(args.limit)]

        print(f"polluted_rows_found={len(polluted)}")
        missing_paths: list[int] = []
        updated = 0

        if not polluted:
            print("No polluted rows detected.")
        else:
            for row in polluted:
                source_path = _local_path_from_uri(str(row.uri or ""))
                if source_path is None or not source_path.exists():
                    missing_paths.append(int(row.id))
                    print(f"missing_source id={int(row.id)} uri={row.uri!r}")
                    continue

                logical_path = _derive_logical_path(row, source_path)
                payload = source_path.read_bytes()
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                new_uri = artifact_service.storage.write(
                    logical_path=logical_path,
                    data=payload,
                    content_type=(row.content_type or None),
                )

                print(
                    f"remediate id={int(row.id)} old_uri={str(row.uri)[:80]!r} "
                    f"new_uri={str(new_uri)[:100]!r} logical_path={logical_path!r}"
                )

                if args.apply:
                    old_uri = str(row.uri)
                    meta = dict(row.metadata_json or {})
                    remediation_meta = dict(meta.get("remediation") or {})
                    remediation_meta.update(
                        {
                            "previous_uri": old_uri,
                            "remediated_at": datetime.now(timezone.utc).isoformat(),
                            "remediation_version": "call_artifacts_uri_blob_v1",
                            "logical_path": logical_path,
                            "sha256": payload_sha256,
                        }
                    )
                    meta["remediation"] = remediation_meta
                    meta["storage_backend"] = "azure_blob"
                    row.uri = str(new_uri)
                    row.byte_size = int(row.byte_size or len(payload))
                    row.metadata_json = meta
                    updated += 1

        mirror_candidates, mirror_updates = _sync_raw_call_uri_mirror(
            session,
            apply=args.apply,
        )

        if args.apply:
            session.flush()

        print(f"updated_rows={updated}")
        print(f"raw_call_uri_mirror_candidates={mirror_candidates}")
        print(f"raw_call_uri_mirror_updates={mirror_updates}")
        if missing_paths:
            print(f"missing_source_rows={len(missing_paths)} ids={missing_paths}")
            if args.apply:
                raise RuntimeError(
                    "Remediation halted: missing local source files detected; no partial approval."
                )
        if not args.apply:
            print("dry_run=true (no DB updates applied)")

    # Re-scan after session commit/rollback.
    with db.session_scope() as session:
        remaining = [
            row
            for row in session.query(CallArtifact).all()
            if not _is_azure_blob_uri(str(row.uri or ""))
        ]
        print(f"remaining_non_blob_rows={len(remaining)}")

    if args.validate_constraint:
        if not args.apply:
            print("ERROR: --validate-constraint requires --apply.", file=sys.stderr)
            return 2
        if remaining:
            print(
                "ERROR: cannot VALIDATE CONSTRAINT while non-blob rows remain.",
                file=sys.stderr,
            )
            return 3
        validated = _validate_constraint(db.engine, constraint_name)
        print(f"constraint_validated={str(validated).lower()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
