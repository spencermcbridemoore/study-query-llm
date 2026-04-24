#!/usr/bin/env python3
"""Live count of local-path / non-blob URIs across artifact-bearing columns in Jetstream.

This is a READ-ONLY audit script bundled with docs/audit/db_target_lane_audit_2026-04-24/.
It is NOT production code; it is the empirical evidence that backs the audit doc.

Tables/columns covered (per subagent 3 inventory of artifact-bearing columns in
src/study_query_llm/db/models_v2.py):

    call_artifacts.uri                 (line 227, String(1000), NOT NULL)
    raw_calls.response_json            (line 51,  JSON; may embed {"uri": ...})
    groups.metadata_json               (line 128, JSON; may embed {"artifact_uri": ...})
    analysis_results.result_json       (line 660, JSON; may embed {"uris": {...}})
    provenanced_runs.result_ref        (line 755, String(400))
    orchestration_jobs.result_ref      (line 433, String(200))

Classification rule:
    - "blob"        : starts with https:// and host contains .blob.core.windows.net
    - "local_path"  : looks like a filesystem path (Windows drive letter, POSIX root, file://, etc.)
    - "empty"       : null or whitespace
    - "other"       : anything else (e.g. http://, opaque tokens, IDs)

Usage:
    python docs/audit/db_target_lane_audit_2026-04-24/live_count.py
        --env-var JETSTREAM_DATABASE_URL
        --output  docs/audit/db_target_lane_audit_2026-04-24/live_count_output.txt

Exit code is always 0 (audit reports findings; it does not enforce policy).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent.parent.parent

WIN_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def classify(value: Any) -> str:
    """Classify a single URI-like value into one of: blob, local_path, empty, other."""
    if value is None:
        return "empty"
    s = str(value).strip()
    if not s:
        return "empty"
    if s.lower().startswith("https://"):
        try:
            host = (urlparse(s).hostname or "").lower()
        except Exception:
            host = ""
        if ".blob.core.windows.net" in host:
            return "blob"
        return "other"
    if (
        WIN_DRIVE_RE.match(s)
        or s.startswith("/")
        or s.startswith("\\\\")
        or s.lower().startswith("file://")
    ):
        return "local_path"
    return "other"


def extract_uris_from_json(payload: Any, key_hints: Iterable[str]) -> list[str]:
    """Walk a JSON payload and return any URI-like strings found under hint keys.

    Recurses into dicts/lists. Hints are keys whose values we *suspect* hold URIs
    (e.g. 'uri', 'artifact_uri'). For 'uris' the value may itself be a dict mapping
    artifact_type -> uri or a list of URIs.
    """
    found: list[str] = []
    hint_set = {h.lower() for h in key_hints}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                if kl in hint_set:
                    if isinstance(v, str):
                        found.append(v)
                    elif isinstance(v, dict):
                        for sv in v.values():
                            if isinstance(sv, str):
                                found.append(sv)
                    elif isinstance(v, list):
                        for sv in v:
                            if isinstance(sv, str):
                                found.append(sv)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def redact_url(url: str) -> str:
    if "@" in url and "://" in url:
        try:
            scheme = url.split("://", 1)[0]
            rest = url.split("@", 1)[-1]
            return f"{scheme}://***@{rest[:160]}"
        except Exception:
            return "***"
    return url[:80]


def fmt_counter(c: Counter) -> str:
    if not c:
        return "(empty)"
    return ", ".join(f"{k}={v}" for k, v in c.most_common())


@contextmanager
def section(out, title: str):
    out.write("\n" + "=" * 78 + "\n")
    out.write(f"== {title}\n")
    out.write("=" * 78 + "\n")
    yield
    out.write("\n")


def query_call_artifacts(conn, out, sample_n: int) -> None:
    with section(out, "call_artifacts.uri"):
        total = int(conn.execute(text("SELECT COUNT(*) FROM call_artifacts")).scalar() or 0)
        out.write(f"total_rows: {total}\n")

        rows = conn.execute(
            text("SELECT id, artifact_type, uri FROM call_artifacts ORDER BY id")
        ).fetchall()

        by_class: Counter[str] = Counter()
        by_type_class: Counter[tuple[str, str]] = Counter()
        bad_samples: list[tuple[int, str, str, str]] = []

        for aid, atype, uri in rows:
            cls = classify(uri)
            by_class[cls] += 1
            by_type_class[(str(atype), cls)] += 1
            if cls in ("local_path", "other") or cls == "empty":
                if len(bad_samples) < sample_n:
                    sample_uri = (uri or "")[:200]
                    bad_samples.append((int(aid), str(atype), cls, sample_uri))

        out.write(f"by_class: {fmt_counter(by_class)}\n")
        out.write("by_artifact_type x class:\n")
        for (atype, cls), n in sorted(by_type_class.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"  {atype:40s} {cls:12s} {n}\n")

        out.write(f"\nsample non-blob URIs (up to {sample_n}):\n")
        for aid, atype, cls, sample in bad_samples:
            out.write(f"  id={aid:6d}  type={atype:40s}  class={cls:12s}  uri={sample!r}\n")


def query_raw_calls_response_json(conn, out, sample_n: int) -> None:
    with section(out, "raw_calls.response_json[uri]"):
        total = int(conn.execute(text("SELECT COUNT(*) FROM raw_calls")).scalar() or 0)
        with_uri = int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM raw_calls "
                    "WHERE response_json IS NOT NULL "
                    "AND response_json::text LIKE '%\"uri\"%'"
                )
            ).scalar()
            or 0
        )
        out.write(f"total_rows: {total}\n")
        out.write(f"rows_with_uri_substring: {with_uri}\n")

        if with_uri == 0:
            out.write("(no rows with embedded 'uri' key — skipping classification)\n")
            return

        rows = conn.execute(
            text(
                "SELECT id, response_json, created_at FROM raw_calls "
                "WHERE response_json IS NOT NULL "
                "AND response_json::text LIKE '%\"uri\"%' "
                "ORDER BY id LIMIT 5000"
            )
        ).fetchall()

        by_class: Counter[str] = Counter()
        bad_samples: list[tuple[int, str, str]] = []

        for rid, payload, created_at in rows:
            try:
                data = payload if isinstance(payload, (dict, list)) else json.loads(payload)
            except Exception:
                continue
            uris = extract_uris_from_json(data, ["uri"])
            for u in uris:
                cls = classify(u)
                by_class[cls] += 1
                if cls in ("local_path", "other"):
                    if len(bad_samples) < sample_n:
                        bad_samples.append((int(rid), cls, u[:200]))

        out.write(f"by_class (over extracted URIs): {fmt_counter(by_class)}\n")
        out.write(f"\nsample non-blob URIs (up to {sample_n}):\n")
        for rid, cls, sample in bad_samples:
            out.write(f"  raw_call_id={rid:6d}  class={cls:12s}  uri={sample!r}\n")


def query_groups_metadata_json(conn, out, sample_n: int) -> None:
    with section(out, "groups.metadata_json[artifact_uri]"):
        total = int(conn.execute(text("SELECT COUNT(*) FROM groups")).scalar() or 0)
        with_uri = int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM groups "
                    "WHERE metadata_json IS NOT NULL "
                    "AND metadata_json::text LIKE '%artifact_uri%'"
                )
            ).scalar()
            or 0
        )
        out.write(f"total_rows: {total}\n")
        out.write(f"rows_with_artifact_uri_substring: {with_uri}\n")

        if with_uri == 0:
            out.write("(no rows with 'artifact_uri' — skipping classification)\n")
            return

        rows = conn.execute(
            text(
                "SELECT id, group_type, name, metadata_json, created_at FROM groups "
                "WHERE metadata_json IS NOT NULL "
                "AND metadata_json::text LIKE '%artifact_uri%' "
                "ORDER BY id LIMIT 5000"
            )
        ).fetchall()

        by_class: Counter[str] = Counter()
        by_type_class: Counter[tuple[str, str]] = Counter()
        bad_samples: list[tuple[int, str, str, str]] = []

        for gid, gtype, name, payload, created_at in rows:
            try:
                data = payload if isinstance(payload, (dict, list)) else json.loads(payload)
            except Exception:
                continue
            uris = extract_uris_from_json(data, ["artifact_uri"])
            for u in uris:
                cls = classify(u)
                by_class[cls] += 1
                by_type_class[(str(gtype), cls)] += 1
                if cls in ("local_path", "other"):
                    if len(bad_samples) < sample_n:
                        bad_samples.append((int(gid), str(gtype), cls, u[:200]))

        out.write(f"by_class: {fmt_counter(by_class)}\n")
        out.write("by_group_type x class:\n")
        for (gtype, cls), n in sorted(by_type_class.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"  {gtype:30s} {cls:12s} {n}\n")
        out.write(f"\nsample non-blob URIs (up to {sample_n}):\n")
        for gid, gtype, cls, sample in bad_samples:
            out.write(f"  group_id={gid:6d}  type={gtype:30s}  class={cls:12s}  uri={sample!r}\n")


def query_analysis_results_result_json(conn, out, sample_n: int) -> None:
    with section(out, "analysis_results.result_json[uris]"):
        total = int(conn.execute(text("SELECT COUNT(*) FROM analysis_results")).scalar() or 0)
        with_uri = int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM analysis_results "
                    "WHERE result_json IS NOT NULL "
                    "AND result_json::text LIKE '%uris%'"
                )
            ).scalar()
            or 0
        )
        out.write(f"total_rows: {total}\n")
        out.write(f"rows_with_uris_substring: {with_uri}\n")

        if with_uri == 0:
            out.write("(no rows with 'uris' key — skipping classification)\n")
            return

        rows = conn.execute(
            text(
                "SELECT id, result_json FROM analysis_results "
                "WHERE result_json IS NOT NULL "
                "AND result_json::text LIKE '%uris%' "
                "ORDER BY id LIMIT 5000"
            )
        ).fetchall()

        by_class: Counter[str] = Counter()
        bad_samples: list[tuple[int, str, str]] = []

        for arid, payload in rows:
            try:
                data = payload if isinstance(payload, (dict, list)) else json.loads(payload)
            except Exception:
                continue
            uris = extract_uris_from_json(data, ["uris", "uri", "artifact_uri"])
            for u in uris:
                cls = classify(u)
                by_class[cls] += 1
                if cls in ("local_path", "other"):
                    if len(bad_samples) < sample_n:
                        bad_samples.append((int(arid), cls, u[:200]))

        out.write(f"by_class: {fmt_counter(by_class)}\n")
        out.write(f"\nsample non-blob URIs (up to {sample_n}):\n")
        for arid, cls, sample in bad_samples:
            out.write(f"  analysis_result_id={arid:6d}  class={cls:12s}  uri={sample!r}\n")


def query_provenanced_runs_result_ref(conn, out, sample_n: int) -> None:
    with section(out, "provenanced_runs.result_ref"):
        total = int(conn.execute(text("SELECT COUNT(*) FROM provenanced_runs")).scalar() or 0)
        out.write(f"total_rows: {total}\n")

        rows = conn.execute(
            text("SELECT id, run_kind, result_ref FROM provenanced_runs ORDER BY id")
        ).fetchall()

        by_class: Counter[str] = Counter()
        by_kind_class: Counter[tuple[str, str]] = Counter()
        bad_samples: list[tuple[int, str, str, str]] = []

        for prid, kind, ref in rows:
            cls = classify(ref)
            by_class[cls] += 1
            by_kind_class[(str(kind), cls)] += 1
            if cls in ("local_path", "other"):
                if len(bad_samples) < sample_n:
                    bad_samples.append((int(prid), str(kind), cls, str(ref or "")[:200]))

        out.write(f"by_class: {fmt_counter(by_class)}\n")
        out.write("by_kind x class:\n")
        for (kind, cls), n in sorted(by_kind_class.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"  {kind:30s} {cls:12s} {n}\n")
        out.write(f"\nsample non-blob result_refs (up to {sample_n}):\n")
        for prid, kind, cls, sample in bad_samples:
            out.write(f"  prov_run_id={prid:6d}  kind={kind:30s}  class={cls:12s}  ref={sample!r}\n")


def query_orchestration_jobs_result_ref(conn, out, sample_n: int) -> None:
    with section(out, "orchestration_jobs.result_ref"):
        total = int(conn.execute(text("SELECT COUNT(*) FROM orchestration_jobs")).scalar() or 0)
        out.write(f"total_rows: {total}\n")

        rows = conn.execute(
            text(
                "SELECT id, job_type, status, result_ref FROM orchestration_jobs ORDER BY id"
            )
        ).fetchall()

        by_class: Counter[str] = Counter()
        by_jobtype_class: Counter[tuple[str, str]] = Counter()
        bad_samples: list[tuple[int, str, str, str, str]] = []

        for ojid, jtype, status, ref in rows:
            cls = classify(ref)
            by_class[cls] += 1
            by_jobtype_class[(str(jtype), cls)] += 1
            if cls in ("local_path", "other"):
                if len(bad_samples) < sample_n:
                    bad_samples.append(
                        (int(ojid), str(jtype), str(status), cls, str(ref or "")[:200])
                    )

        out.write(f"by_class: {fmt_counter(by_class)}\n")
        out.write("by_job_type x class:\n")
        for (jtype, cls), n in sorted(by_jobtype_class.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"  {jtype:30s} {cls:12s} {n}\n")
        out.write(f"\nsample non-blob result_refs (up to {sample_n}):\n")
        for ojid, jtype, status, cls, sample in bad_samples:
            out.write(
                f"  job_id={ojid:6d}  type={jtype:30s}  status={status:10s}  "
                f"class={cls:12s}  ref={sample!r}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-var", default="JETSTREAM_DATABASE_URL")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--output", default=None, help="Optional path to write report to")
    parser.add_argument("--sample-n", type=int, default=15)
    parser.add_argument("--connect-timeout", type=int, default=15)
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    url = (args.database_url or "").strip() or (os.environ.get(args.env_var) or "").strip()
    if not url:
        print(f"ERROR: {args.env_var} not set and --database-url not provided.", file=sys.stderr)
        return 1

    eng = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(args.connect_timeout)},
    )

    out_paths: list[Any] = [sys.stdout]
    out_file = None
    if args.output:
        out_file = open(args.output, "w", encoding="utf-8")
        out_paths.append(out_file)

    class Tee:
        def __init__(self, streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    out = Tee(out_paths)

    out.write("Jetstream Local-Path URI Audit\n")
    out.write(f"url_redacted: {redact_url(url)}\n")
    out.write(f"env_var: {args.env_var}\n")

    try:
        with eng.connect() as conn:
            query_call_artifacts(conn, out, args.sample_n)
            query_raw_calls_response_json(conn, out, args.sample_n)
            query_groups_metadata_json(conn, out, args.sample_n)
            query_analysis_results_result_json(conn, out, args.sample_n)
            query_provenanced_runs_result_ref(conn, out, args.sample_n)
            query_orchestration_jobs_result_ref(conn, out, args.sample_n)
    except Exception as exc:
        out.write(f"\nERROR during query: {exc}\n")
        if out_file:
            out_file.close()
        return 1
    finally:
        if out_file:
            out_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
