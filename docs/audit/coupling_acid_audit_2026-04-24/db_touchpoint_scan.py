#!/usr/bin/env python3
"""Scan the package for direct DB access (sessions, ORM ops, raw SQL).

READ-ONLY audit script. Identifies every module that:
  - imports SQLAlchemy session / engine APIs
  - imports v2 ORM models (`models_v2`) or repositories
  - imports v1 ORM models (`models`) or repositories — flagged separately
  - calls `session.add` / `session.query` / `session.commit` / `session.flush`
  - uses raw SQL via `text(...)` from sqlalchemy
  - opens `session_scope()` / `engine.begin()` / `engine.connect()`

Outputs to `docs/audit/coupling_acid_audit_2026-04-24/outputs/`:
    - db_touchpoints.txt : categorized list with file:line evidence

Usage:
    python docs/audit/coupling_acid_audit_2026-04-24/db_touchpoint_scan.py
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC = REPO / "src"
PACKAGE = "study_query_llm"
PACKAGE_ROOT = SRC / PACKAGE
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Categories
LEGITIMATE_DB_PREFIX = (f"{PACKAGE}.db.",)
LEGITIMATE_DB_FILES = (
    "db/_base_connection.py",
    "db/connection.py",
    "db/connection_v2.py",
    "db/raw_call_repository.py",
    "db/inference_repository.py",
)

V1_MARKERS = (
    "from study_query_llm.db.models import",
    "from study_query_llm.db.connection import",
    "from study_query_llm.db.inference_repository import",
    "from ..db.models import",
    "from ..db.connection import",
    "from ..db.inference_repository import",
)

V2_MARKERS = (
    "from study_query_llm.db.models_v2 import",
    "from study_query_llm.db.connection_v2 import",
    "from study_query_llm.db.raw_call_repository import",
    "from ..db.models_v2 import",
    "from ..db.connection_v2 import",
    "from ..db.raw_call_repository import",
    "from .models_v2 import",
    "from .connection_v2 import",
    "from .raw_call_repository import",
)

SESSION_OPS = (
    re.compile(r"\bsession\.add\("),
    re.compile(r"\bsession\.query\("),
    re.compile(r"\bsession\.commit\("),
    re.compile(r"\bsession\.flush\("),
    re.compile(r"\bsession\.execute\("),
    re.compile(r"\bsession\.scalar\("),
    re.compile(r"\bsession\.scalars\("),
    re.compile(r"\bsession\.delete\("),
    re.compile(r"\bsession_scope\(\)"),
    re.compile(r"\bengine\.connect\(\)"),
    re.compile(r"\bengine\.begin\(\)"),
    re.compile(r"\bbegin_nested\(\)"),
)

RAW_SQL = (
    re.compile(r"\bfrom\s+sqlalchemy\s+import\s+[^\n]*\btext\b"),
    re.compile(r"\btext\(\s*['\"]"),  # text("SELECT ...") inline
)


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def scan_file(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()

    info = {
        "v1_imports": [],   # (line, text)
        "v2_imports": [],   # (line, text)
        "session_ops": [],  # (line, op, snippet)
        "raw_sql": [],      # (line, snippet)
    }

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        for marker in V1_MARKERS:
            if marker in line:
                info["v1_imports"].append((i, stripped))
                break
        for marker in V2_MARKERS:
            if marker in line:
                info["v2_imports"].append((i, stripped))
                break
        for pat in SESSION_OPS:
            m = pat.search(line)
            if m:
                info["session_ops"].append((i, m.group(0), stripped[:120]))
        for pat in RAW_SQL:
            m = pat.search(line)
            if m:
                info["raw_sql"].append((i, stripped[:120]))
    return info


def categorize(rel_path: str) -> str:
    if rel_path.startswith("src/study_query_llm/db/"):
        return "REPOSITORY (legitimate)"
    if rel_path.startswith("src/study_query_llm/services/"):
        return "SERVICE (potentially legitimate via repository)"
    if rel_path.startswith("src/study_query_llm/pipeline/"):
        return "PIPELINE (stage code)"
    if rel_path.startswith("src/study_query_llm/experiments/"):
        return "EXPERIMENTS (workers / ingestion)"
    if rel_path.startswith("src/study_query_llm/analysis/"):
        return "ANALYSIS"
    if rel_path.startswith("src/study_query_llm/algorithms/"):
        return "ALGORITHMS (LEAK if any)"
    if rel_path.startswith("src/study_query_llm/providers/"):
        return "PROVIDERS (LEAK if any)"
    if rel_path.startswith("src/study_query_llm/domain/"):
        return "DOMAIN (LEAK if any)"
    if rel_path.startswith("src/study_query_llm/storage/"):
        return "STORAGE (LEAK if any)"
    if rel_path.startswith("src/study_query_llm/utils/"):
        return "UTILS"
    return "OTHER"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    v1_v2_dual: list[tuple[str, dict]] = []

    files_with_db = 0
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        info = scan_file(path)
        if not any(info[k] for k in ("v1_imports", "v2_imports", "session_ops", "raw_sql")):
            continue
        files_with_db += 1
        rel_path = rel(path)
        by_cat[categorize(rel_path)].append((rel_path, info))
        if info["v1_imports"] and info["v2_imports"]:
            v1_v2_dual.append((rel_path, info))

    out: list[str] = []
    out.append("# DB Touchpoint Inventory")
    out.append(f"Files with any DB indicator: {files_with_db}")
    out.append("")

    cat_order = [
        "REPOSITORY (legitimate)",
        "SERVICE (potentially legitimate via repository)",
        "PIPELINE (stage code)",
        "EXPERIMENTS (workers / ingestion)",
        "ANALYSIS",
        "ALGORITHMS (LEAK if any)",
        "PROVIDERS (LEAK if any)",
        "DOMAIN (LEAK if any)",
        "STORAGE (LEAK if any)",
        "UTILS",
        "OTHER",
    ]

    for cat in cat_order:
        items = by_cat.get(cat, [])
        if not items:
            continue
        out.append(f"## {cat}  ({len(items)} files)")
        items.sort(key=lambda kv: kv[0])
        for rel_path, info in items:
            n_session = len(info["session_ops"])
            n_sql = len(info["raw_sql"])
            n_v1 = len(info["v1_imports"])
            n_v2 = len(info["v2_imports"])
            tags = []
            if n_v1:
                tags.append(f"v1_import={n_v1}")
            if n_v2:
                tags.append(f"v2_import={n_v2}")
            if n_session:
                tags.append(f"session_ops={n_session}")
            if n_sql:
                tags.append(f"raw_sql={n_sql}")
            out.append(f"- {rel_path}  ({', '.join(tags)})")
        out.append("")

    out.append("## v1 + v2 dual-import bridges")
    if v1_v2_dual:
        for rel_path, info in v1_v2_dual:
            out.append(f"- {rel_path}")
            for ln, txt in info["v1_imports"][:3]:
                out.append(f"    v1@{ln}: {txt}")
            for ln, txt in info["v2_imports"][:3]:
                out.append(f"    v2@{ln}: {txt}")
    else:
        out.append("  (none)")
    out.append("")

    out.append("## Direct session ops outside db/ (top 30 by op count)")
    leaks: list[tuple[int, str, dict]] = []
    for cat, items in by_cat.items():
        if cat in ("REPOSITORY (legitimate)",):
            continue
        for rel_path, info in items:
            n = len(info["session_ops"])
            if n:
                leaks.append((n, rel_path, info))
    leaks.sort(reverse=True)
    for n, rel_path, info in leaks[:30]:
        out.append(f"- {n:3d}  {rel_path}")
        for ln, op, snippet in info["session_ops"][:5]:
            out.append(f"      L{ln}  {op}  -- {snippet}")
    out.append("")

    out_path = OUTPUT_DIR / "db_touchpoints.txt"
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Files with DB activity: {files_with_db}")
    print(f"v1+v2 dual-import bridges: {len(v1_v2_dual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
