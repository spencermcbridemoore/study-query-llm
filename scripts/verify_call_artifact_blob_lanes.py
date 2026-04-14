#!/usr/bin/env python3
"""Verify call_artifacts.uri Azure blob URLs target the expected container (lane).

Read-only: SELECT on ``call_artifacts``. Parses ``https://…blob.core.windows.net/<container>/…``.

Usage (repo root):
  python scripts/verify_call_artifact_blob_lanes.py
  python scripts/verify_call_artifact_blob_lanes.py --env-var JETSTREAM_DATABASE_URL
  python scripts/verify_call_artifact_blob_lanes.py --database-url postgresql://...
  python scripts/verify_call_artifact_blob_lanes.py --expected-container artifacts-dev --expected-prefix dev

Exit codes:
  0 — all Azure URIs match ``--expected-container`` (and prefix rule if set); or no Azure URIs
  1 — mismatch, connection error, or missing URL
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent


def azure_blob_container_from_uri(uri: str) -> str | None:
    """Return blob container name if *uri* is an Azure Blob HTTPS URL, else None."""
    parsed = urlparse((uri or "").strip())
    if parsed.scheme != "https":
        return None
    host = (parsed.hostname or "").lower()
    if ".blob.core.windows.net" not in host:
        return None
    path = unquote(parsed.path or "").strip("/")
    if not path:
        return None
    return path.split("/", 1)[0]


def azure_blob_path_after_container(uri: str) -> str | None:
    """Return blob name (key) after container segment, or None if not an Azure blob HTTPS URL."""
    parsed = urlparse((uri or "").strip())
    if parsed.scheme != "https":
        return None
    host = (parsed.hostname or "").lower()
    if ".blob.core.windows.net" not in host:
        return None
    path = unquote(parsed.path or "").strip("/")
    if not path:
        return None
    parts = path.split("/", 1)
    return parts[1] if len(parts) > 1 else ""


def _redact(url: str) -> str:
    if "@" in url and "://" in url:
        try:
            rest = url.split("@", 1)[-1]
            scheme = url.split("://", 1)[0]
            return f"{scheme}://***@{rest[:160]}"
        except Exception:
            return "***"
    return url[:80]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify call_artifacts Azure URIs use the expected blob container (lane).",
    )
    parser.add_argument(
        "--env-var",
        default="DATABASE_URL",
        help="Env var for Postgres URL (default: DATABASE_URL)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override Postgres URL instead of --env-var",
    )
    parser.add_argument(
        "--expected-container",
        default="artifacts-dev",
        help="Azure container name every blob URI must use (default: artifacts-dev)",
    )
    parser.add_argument(
        "--expected-prefix",
        default=None,
        help="If set, Azure blob keys must start with this prefix (e.g. dev/)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="TCP connect timeout seconds",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=20,
        help="Max sample URIs printed per mismatch category",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    url = (args.database_url or "").strip() or (os.environ.get(args.env_var) or "").strip()
    if not url:
        print(
            f"ERROR: {args.env_var} not set and --database-url not passed.",
            file=sys.stderr,
        )
        return 1

    expected_c = (args.expected_container or "").strip()
    if not expected_c:
        print("ERROR: --expected-container must be non-empty.", file=sys.stderr)
        return 1

    expected_prefix = (args.expected_prefix or "").strip()
    print("url_redacted:", _redact(url))
    print("expected_container:", expected_c)
    if expected_prefix:
        print("expected_blob_key_prefix:", repr(expected_prefix))

    eng = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(args.connect_timeout)},
    )

    try:
        with eng.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM call_artifacts")).scalar()
            total = int(total or 0)
            print("call_artifacts_total:", total)

            rows = conn.execute(
                text("SELECT id, uri FROM call_artifacts ORDER BY id")
            ).fetchall()
    except Exception as e:
        print(f"ERROR: database query failed: {e}", file=sys.stderr)
        return 1

    by_container: Counter[str] = Counter()
    local_or_other = 0
    mismatch_ids: list[tuple[int, str, str]] = []
    prefix_mismatch: list[tuple[int, str]] = []

    for aid, uri in rows:
        u = (uri or "").strip()
        if not u:
            local_or_other += 1
            continue
        c = azure_blob_container_from_uri(u)
        if c is None:
            local_or_other += 1
            continue
        by_container[c] += 1
        if c != expected_c:
            mismatch_ids.append((int(aid), c, u[:200]))
            continue
        if expected_prefix:
            key = azure_blob_path_after_container(u) or ""
            if not key.startswith(expected_prefix):
                prefix_mismatch.append((int(aid), u[:200]))

    print("\n=== Azure blob URIs by container (from URL path) ===")
    if not by_container:
        print("(none — no https://*.blob.core.windows.net/ URIs)")
    else:
        for name, n in by_container.most_common():
            mark = "OK" if name == expected_c else "MISMATCH"
            print(f"  {name}: {n}  [{mark}]")

    print("\n=== Non-Azure URIs (local paths, empty, or other schemes) ===")
    print("  count:", local_or_other)

    bad = False
    if mismatch_ids:
        bad = True
        print(f"\n=== MISMATCH: not container {expected_c!r} (showing up to {args.limit_samples}) ===")
        for aid, c, sample in mismatch_ids[: args.limit_samples]:
            print(f"  id={aid} container={c!r} uri_prefix={sample!r}")
        if len(mismatch_ids) > args.limit_samples:
            print(f"  ... and {len(mismatch_ids) - args.limit_samples} more")

    if expected_prefix and prefix_mismatch:
        bad = True
        print(
            f"\n=== MISMATCH: blob key does not start with {expected_prefix!r} "
            f"(showing up to {args.limit_samples}) ==="
        )
        for aid, sample in prefix_mismatch[: args.limit_samples]:
            print(f"  id={aid} uri_prefix={sample!r}")
        if len(prefix_mismatch) > args.limit_samples:
            print(f"  ... and {len(prefix_mismatch) - args.limit_samples} more")

    if bad:
        print("\nRESULT: FAIL (see mismatches above).", file=sys.stderr)
        return 1

    print("\nRESULT: OK — all Azure blob call_artifacts URIs use the expected container.")
    if expected_prefix:
        print(f"         All matching rows use blob key prefix {expected_prefix!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
