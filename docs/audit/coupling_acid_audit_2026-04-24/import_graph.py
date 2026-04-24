#!/usr/bin/env python3
"""Build the intra-package import graph for `src/study_query_llm/` and detect cycles.

READ-ONLY audit script bundled with `docs/audit/coupling_acid_audit_2026-04-24/`.
Walks every .py file under the package, parses its AST, and records every
`import study_query_llm.X` / `from study_query_llm.X import ...` /
relative `from ..X import ...` resolved to a package-qualified module name.

Outputs (to `docs/audit/coupling_acid_audit_2026-04-24/outputs/`):
    - import_graph.json  : {"nodes": [...], "edges": [[from, to], ...]}
    - cycles.txt         : human-readable list of strongly-connected components
                           with size > 1 (true cycles).

Usage:
    python docs/audit/coupling_acid_audit_2026-04-24/import_graph.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC = REPO / "src"
PACKAGE = "study_query_llm"
PACKAGE_ROOT = SRC / PACKAGE
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def file_to_module_and_package(path: Path) -> tuple[str, str]:
    """Return (module_dotted_name, importer_package).

    For ``foo/bar/baz.py``:        module="foo.bar.baz", package="foo.bar"
    For ``foo/bar/baz/__init__.py``: module="foo.bar.baz", package="foo.bar.baz"

    The *package* is what relative-import resolution anchors on (PEP 328): for a
    regular module the package is the parent directory; for ``__init__.py`` the
    package IS the directory itself.
    """
    rel = path.relative_to(SRC).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        pkg = parts[:-1]
        return ".".join(pkg), ".".join(pkg)
    return ".".join(parts), ".".join(parts[:-1])


def resolve_relative_import(
    importer_package: str, from_module: str | None, level: int
) -> str | None:
    """Resolve a `from .X import ...` to a fully qualified name (PEP 328).

    For a relative import at *level* N from importer package P:
      - level=1 -> base = P
      - level=2 -> base = parent(P)
      - level=N -> peel (N-1) components from P
    """
    if level == 0:
        return from_module
    pkg_parts = importer_package.split(".") if importer_package else []
    peel = level - 1
    if len(pkg_parts) < peel:
        return None
    base = pkg_parts[: len(pkg_parts) - peel]
    if from_module:
        base = base + from_module.split(".")
    return ".".join(base) if base else None


def collect_imports(file_path: Path) -> set[str]:
    """Return the set of fully qualified module imports from `file_path` that point
    inside the `study_query_llm` package."""
    src = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(file_path))
    except SyntaxError:
        return set()

    _module, importer_package = file_to_module_and_package(file_path)
    out: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            from_mod = node.module
            target = resolve_relative_import(importer_package, from_mod, level)
            if not target:
                continue
            if not target.startswith(PACKAGE):
                continue
            for alias in node.names:
                out.add(target)
                candidate = f"{target}.{alias.name}"
                out.add(candidate)
    return out


def file_to_module(path: Path) -> str:
    """Backward-compatible wrapper returning only the module dotted name."""
    return file_to_module_and_package(path)[0]


def build_graph() -> tuple[dict[str, set[str]], set[str]]:
    """Walk all .py under `src/study_query_llm/` and build {module: {imported_modules}}."""
    nodes: set[str] = set()
    edges: dict[str, set[str]] = {}

    for path in PACKAGE_ROOT.rglob("*.py"):
        # skip caches, tests-shaped extras
        if "__pycache__" in path.parts:
            continue
        mod = file_to_module(path)
        nodes.add(mod)
        edges.setdefault(mod, set())
        for imp in collect_imports(path):
            edges[mod].add(imp)

    # Restrict targets to known nodes (fall back to nearest ancestor when imp points
    # to a name inside a module that we did register).
    resolved: dict[str, set[str]] = {}
    for mod, imps in edges.items():
        resolved[mod] = set()
        for imp in imps:
            if imp == mod:
                continue
            if imp in nodes:
                resolved[mod].add(imp)
            else:
                # Try ancestor rollup (e.g. study_query_llm.db.models_v2.RawCall -> study_query_llm.db.models_v2)
                parts = imp.split(".")
                for i in range(len(parts) - 1, 0, -1):
                    candidate = ".".join(parts[:i])
                    if candidate in nodes and candidate != mod:
                        resolved[mod].add(candidate)
                        break
    return resolved, nodes


def tarjan_scc(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC algorithm. Returns SCCs as lists of nodes."""
    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in graph.get(node, ()):
            if neighbor not in index:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack.get(neighbor, False):
                lowlinks[node] = min(lowlinks[node], index[neighbor])

        if lowlinks[node] == index[node]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == node:
                    break
            result.append(component)

    sys.setrecursionlimit(10_000)
    for node in graph:
        if node not in index:
            strongconnect(node)
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph, nodes = build_graph()

    # Persist graph
    serial = {
        "package": PACKAGE,
        "node_count": len(nodes),
        "edge_count": sum(len(v) for v in graph.values()),
        "nodes": sorted(nodes),
        "edges": sorted([f, t] for f, ts in graph.items() for t in ts),
    }
    (OUTPUT_DIR / "import_graph.json").write_text(
        json.dumps(serial, indent=2), encoding="utf-8"
    )

    sccs = tarjan_scc(graph)
    cycles = [c for c in sccs if len(c) > 1]
    cycles.sort(key=lambda c: -len(c))

    lines: list[str] = []
    lines.append(f"Modules: {len(nodes)}")
    lines.append(f"Edges:   {sum(len(v) for v in graph.values())}")
    lines.append(f"SCCs:    {len(sccs)} (true cycles size>1: {len(cycles)})")
    lines.append("")
    if not cycles:
        lines.append("No import cycles detected.")
    else:
        lines.append("Import cycles (largest first):")
        for i, comp in enumerate(cycles, 1):
            lines.append(f"\n[{i}] size={len(comp)}")
            for m in sorted(comp):
                lines.append(f"  - {m}")
    (OUTPUT_DIR / "cycles.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_DIR / 'import_graph.json'}")
    print(f"Wrote {OUTPUT_DIR / 'cycles.txt'}")
    print(f"Modules: {len(nodes)}, Edges: {sum(len(v) for v in graph.values())}, "
          f"True cycles: {len(cycles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
