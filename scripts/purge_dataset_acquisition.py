#!/usr/bin/env python3
"""
Purge Layer-0 dataset acquisition blobs and DB rows so ``record_dataset_download.py
--persist-db`` can be re-run for the same ``--dataset-group-name``.

Blob deletion order (deterministic, safe for humans re-reading logs):
  1. All ``dataset_acquisition_file`` artifacts for the group (sorted by ``call_artifacts.id``).
  2. ``dataset_acquisition_manifest`` last (index / manifest file).

Database deletion order (FK-safe):
  1. ``call_artifacts`` rows for those URIs.
  2. Placeholder ``raw_calls`` referenced by those artifacts.
  3. The ``groups`` row (must be ``group_type = dataset``).

Requires the same storage env as acquisition (``ARTIFACT_STORAGE_BACKEND=azure_blob`` and
``AZURE_STORAGE_*`` for Azure; or ``local`` for local paths).

Usage (repo root, default dry-run):
  python scripts/purge_dataset_acquisition.py --group-name acquire_ausem
  python scripts/purge_dataset_acquisition.py --group-id 1 --execute
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import sessionmaker

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _delete_blob_for_uri(storage: Any, uri: str) -> str:
    """Return status line for one delete attempt."""
    bt = getattr(storage, "backend_type", "")
    try:
        if bt == "azure_blob":
            logical = storage._blob_path_from_uri(uri)
            storage.delete(logical)
            return f"blob_ok azure:{logical[:96]}..."
        if bt == "local":
            p = Path(uri)
            if p.exists():
                p.unlink()
                return f"blob_ok local:{p}"
            return f"blob_skip missing local:{p}"
        return f"blob_skip unknown_backend={bt!r}"
    except Exception as exc:
        return f"blob_err {type(exc).__name__}: {exc}"


def _ordered_artifacts(
    session: Any, group_id: int
) -> List[Tuple[int, int, str, str]]:
    """
    Return (call_artifact_id, call_id, artifact_type, uri) with data files first, manifest last.
    """
    rows = session.execute(
        text(
            """
            SELECT ca.id, ca.call_id, ca.artifact_type, ca.uri
            FROM call_artifacts ca
            WHERE (ca.metadata_json->>'group_id')::int = :gid
            ORDER BY
              CASE ca.artifact_type
                WHEN 'dataset_acquisition_manifest' THEN 1
                ELSE 0
              END,
              ca.id
            """
        ),
        {"gid": group_id},
    ).fetchall()
    return [(int(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purge dataset acquisition artifacts + DB rows for a clean re-import."
    )
    parser.add_argument(
        "--group-name",
        default=None,
        help="Dataset group name (e.g. acquire_ausem)",
    )
    parser.add_argument(
        "--group-id",
        type=int,
        default=None,
        help="Dataset group primary key (alternative to --group-name)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete blobs and DB rows (default is dry-run)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: DATABASE_URL from .env; use Jetstream tunnel URL to purge remote)",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    db_url = (args.database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        print(
            "ERROR: DATABASE_URL not set and --database-url not passed.",
            file=sys.stderr,
        )
        return 1

    if bool(args.group_name) == bool(args.group_id):
        print("ERROR: Provide exactly one of --group-name or --group-id.", file=sys.stderr)
        return 1

    from study_query_llm.db.models_v2 import CallArtifact, Group, RawCall
    from study_query_llm.services.artifact_service import ArtifactService

    storage = ArtifactService(repository=None, artifact_dir="artifacts").storage

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        if args.group_id is not None:
            gid = int(args.group_id)
            row = session.execute(
                text(
                    "SELECT id, group_type, name FROM groups WHERE id = :id AND group_type = 'dataset'"
                ),
                {"id": gid},
            ).fetchone()
        else:
            row = session.execute(
                text(
                    "SELECT id, group_type, name FROM groups "
                    "WHERE name = :n AND group_type = 'dataset'"
                ),
                {"n": args.group_name},
            ).fetchone()
        if not row:
            print("ERROR: No matching dataset group found.", file=sys.stderr)
            return 1
        gid, gtype, gname = int(row[0]), str(row[1]), str(row[2])
        print(f"Target group id={gid} type={gtype!r} name={gname!r}")

        ordered = _ordered_artifacts(session, gid)
        if not ordered:
            print("No call_artifacts with metadata_json.group_id for this group; nothing to purge.")
            return 0

        call_ids: List[int] = sorted({c for _, c, _, _ in ordered})
        art_ids: Sequence[int] = [a[0] for a in ordered]

        print("Planned blob order (files first, manifest last):")
        for i, (aid, cid, atype, uri) in enumerate(ordered, 1):
            print(f"  {i}. call_artifact_id={aid} type={atype!r} call_id={cid}")
            print(f"      uri={uri[:120]}{'...' if len(uri) > 120 else ''}")

        if not args.execute:
            print("\nDry-run only. Re-run with --execute to apply.")
            return 0

        for aid, cid, atype, uri in ordered:
            msg = _delete_blob_for_uri(storage, uri)
            print(f"[blob] {atype} id={aid}: {msg}")

        session.execute(delete(CallArtifact).where(CallArtifact.id.in_(list(art_ids))))
        session.execute(delete(RawCall).where(RawCall.id.in_(list(call_ids))))
        session.execute(delete(Group).where(Group.id == gid))
        session.commit()
        print(f"[db] Deleted call_artifacts ids={list(art_ids)}, raw_calls ids={call_ids}, group id={gid}")
        print("OK. You can re-run: python scripts/record_dataset_download.py ... --persist-db ...")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
