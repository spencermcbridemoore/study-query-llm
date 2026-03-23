"""
Scan src/study_query_llm and emit architecture graphs under scratch/out/.

Usage (from repository root):
  python scratch/repo_architecture_graph.py
  python scratch/repo_architecture_graph.py --classes
  python scratch/repo_architecture_graph.py --dot-max-edges 400

Outputs (UTF-8):
  scratch/out/graph.json       — nodes + edges
  scratch/out/graph.dot        — Graphviz (internal modules + layers)
  scratch/out/graph_external.dot — optional third-party rollup (see --external-dot)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _src_pkg_root(root: Path) -> Path:
    return root / "src" / "study_query_llm"


def path_to_module(py_file: Path, pkg_root: Path) -> str:
    rel = py_file.relative_to(pkg_root)
    if rel.name == "__init__.py":
        rel = rel.parent
    else:
        rel = rel.with_suffix("")
    parts = rel.parts
    if not parts:
        return "study_query_llm"
    return "study_query_llm." + ".".join(parts)


def layer_for_module(mod: str) -> str:
    if mod == "study_query_llm" or not mod.startswith("study_query_llm."):
        return "root"
    rest = mod.removeprefix("study_query_llm.").split(".")
    return rest[0] if rest else "root"


def discover_modules(pkg_root: Path) -> dict[str, Path]:
    """Map dotted module name -> file path for .py files under the package."""
    modules: dict[str, Path] = {}
    if not pkg_root.is_dir():
        return modules
    for p in pkg_root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        modules[path_to_module(p, pkg_root)] = p
    return modules


def resolve_internal_target(
    module: str | None,
    names: tuple[str, ...] | list[str] | None,
    level: int,
    source_module: str,
    known: set[str],
) -> list[str]:
    """
    Best-effort resolution of import target to known study_query_llm.* modules.
    """
    if module is None:
        # relative import
        if level <= 0:
            return []
        parts = source_module.split(".")
        if level > len(parts):
            return []
        base_parts = parts[:-level]
        if names:
            cand = ".".join(base_parts + [names[0]]) if base_parts else names[0]
            if not cand.startswith("study_query_llm"):
                cand = "study_query_llm." + cand if cand else "study_query_llm"
        else:
            cand = ".".join(base_parts) if base_parts else "study_query_llm"
        if cand in known:
            return [cand]
        # try stripping last segment
        while cand.count(".") >= 2:
            cand = cand.rsplit(".", 1)[0]
            if cand in known:
                return [cand]
        return []

    if not module.startswith("study_query_llm"):
        return []

    if module in known:
        return [module]

    # from study_query_llm.services import submodule_or_attr
    if names:
        first = names[0]
        sub = f"{module}.{first}"
        if sub in known:
            return [sub]
    if module in known:
        return [module]

    # walk parents
    cur = module
    while cur.count(".") >= 2:
        cur = cur.rsplit(".", 1)[0]
        if cur in known:
            return [cur]
    return []


class ImportVisitor(ast.NodeVisitor):
    def __init__(self, source_module: str, known: set[str]) -> None:
        self.source_module = source_module
        self.known = known
        self.internal_edges: list[tuple[str, str, str]] = []  # src, dst, kind
        self.external_refs: list[tuple[str, str, str]] = []  # src, top_level, kind

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            if name == "study_query_llm" or name.startswith("study_query_llm."):
                tgt = name.split(".", 1)[0] if name == "study_query_llm" else name
                for resolved in resolve_internal_target(
                    tgt, (), 0, self.source_module, self.known
                ):
                    self.internal_edges.append((self.source_module, resolved, "import"))
                # deeper: study_query_llm.foo.bar
                if name.startswith("study_query_llm.") and name not in self.known:
                    cur = name
                    while cur not in self.known and cur.count(".") >= 2:
                        cur = cur.rsplit(".", 1)[0]
                    if cur in self.known:
                        self.internal_edges.append((self.source_module, cur, "import"))
            else:
                top = name.split(".")[0]
                self.external_refs.append((self.source_module, top, "import"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and node.module.startswith("study_query_llm"):
            targets = resolve_internal_target(
                node.module,
                tuple(n.name for n in node.names),
                0,
                self.source_module,
                self.known,
            )
            for t in targets:
                self.internal_edges.append((self.source_module, t, "from"))
        elif node.module is not None and not node.module.startswith("."):
            top = node.module.split(".")[0]
            self.external_refs.append((self.source_module, top, "from"))
        else:
            # relative
            targets = resolve_internal_target(
                node.module,
                tuple(n.name for n in node.names),
                node.level,
                self.source_module,
                self.known,
            )
            for t in targets:
                self.internal_edges.append((self.source_module, t, "relative"))
        self.generic_visit(node)


def class_kind(name: str) -> str:
    if name.endswith("Factory"):
        return "factory"
    if name.endswith("Manager"):
        return "manager"
    if name.endswith("Service"):
        return "service"
    if name.endswith("Provider"):
        return "provider"
    return "class"


class ClassVisitor(ast.NodeVisitor):
    def __init__(self, source_module: str) -> None:
        self.source_module = source_module
        self.classes: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases: list[str] = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                parts: list[str] = []
                cur: ast.expr = b
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                bases.append(".".join(reversed(parts)))
        self.classes.append(
            {
                "id": f"{self.source_module}:{node.name}",
                "module": self.source_module,
                "name": node.name,
                "kind": class_kind(node.name),
                "bases": bases,
            }
        )
        self.generic_visit(node)


def build_graph(
    pkg_root: Path,
    include_classes: bool,
) -> dict[str, Any]:
    modules_map = discover_modules(pkg_root)
    known = set(modules_map.keys())

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_layers: set[str] = set()

    for mod in sorted(known):
        lay = layer_for_module(mod)
        seen_layers.add(lay)
        nodes.append(
            {
                "id": mod,
                "type": "module",
                "layer": lay,
                "path": str(modules_map[mod].relative_to(pkg_root.parent.parent)).replace(
                    "\\", "/"
                ),
            }
        )

    for lay in sorted(seen_layers):
        nodes.append({"id": f"layer:{lay}", "type": "layer", "layer": lay})

    internal_pairs: set[tuple[str, str, str]] = set()
    external_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for mod, py_path in sorted(modules_map.items(), key=lambda x: x[0]):
        try:
            src = py_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(src, filename=str(py_path))
        except SyntaxError:
            continue
        v = ImportVisitor(mod, known)
        v.visit(tree)
        for s, t, k in v.internal_edges:
            internal_pairs.add((s, t, k))
        for s, top, k in v.external_refs:
            external_counts[(s, top)] += 1

        if include_classes:
            cv = ClassVisitor(mod)
            cv.visit(tree)
            for c in cv.classes:
                cid = c["id"]
                nodes.append(
                    {
                        "id": cid,
                        "type": "class",
                        "kind": c["kind"],
                        "layer": layer_for_module(mod),
                        "module": mod,
                        "name": c["name"],
                        "bases": c["bases"],
                    }
                )
                edges.append({"source": mod, "target": cid, "kind": "contains"})
                for b in c["bases"]:
                    if b in ("ABC", "Protocol", "object", "Enum", "TypedDict"):
                        continue
                    edges.append(
                        {"source": cid, "target": b, "kind": "base", "unresolved": True}
                    )

    for s, t, k in sorted(internal_pairs):
        edges.append({"source": s, "target": t, "kind": k})

    ext_packages = sorted({top for (_, top) in external_counts.keys()})
    for pkg in ext_packages:
        nodes.append({"id": f"ext:{pkg}", "type": "external", "name": pkg})

    for (s, top), n in sorted(external_counts.items()):
        edges.append(
            {
                "source": s,
                "target": f"ext:{top}",
                "kind": "uses_external",
                "count": n,
            }
        )

    return {
        "meta": {
            "package_root": str(pkg_root).replace("\\", "/"),
            "module_count": len(known),
            "include_classes": include_classes,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_dot_modules(
    graph: dict[str, Any],
    path: Path,
    max_edges: int | None,
) -> None:
    modules = [n for n in graph["nodes"] if n["type"] == "module"]
    layers: dict[str, list[str]] = defaultdict(list)
    for n in modules:
        layers[n["layer"]].append(n["id"])

    internal_edges = [
        e
        for e in graph["edges"]
        if e["kind"] in ("import", "from", "relative")
        and not str(e["target"]).startswith("ext:")
    ]
    if max_edges is not None and len(internal_edges) > max_edges:
        internal_edges = internal_edges[:max_edges]

    lines: list[str] = [
        "digraph study_query_llm {",
        '  graph [rankdir=LR, fontsize=10, label="study_query_llm internal modules"];',
        "  node [shape=box, style=rounded];",
    ]

    for lay in sorted(layers.keys()):
        lines.append(f'  subgraph "cluster_{_dot_escape(lay)}" {{')
        lines.append(f'    label="{_dot_escape(lay)}";')
        for mid in sorted(layers[lay]):
            short = mid.removeprefix("study_query_llm.")
            lines.append(f'    "{_dot_escape(mid)}" [label="{_dot_escape(short)}"];')
        lines.append("  }")

    for e in internal_edges:
        lines.append(
            f'  "{_dot_escape(e["source"])}" -> "{_dot_escape(e["target"])}" '
            f'[label="{_dot_escape(e["kind"])}"];'
        )

    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dot_external(graph: dict[str, Any], path: Path, max_edges: int | None) -> None:
    ext_edges = [e for e in graph["edges"] if e["kind"] == "uses_external"]
    if max_edges is not None and len(ext_edges) > max_edges:
        ext_edges = ext_edges[:max_edges]

    lines: list[str] = [
        "digraph external_deps {",
        '  graph [rankdir=LR, fontsize=10, label="External package references"];',
        "  node [shape=ellipse];",
    ]
    for e in ext_edges:
        tgt = e["target"]
        label = f'{e.get("count", 1)}'
        lines.append(
            f'  "{_dot_escape(e["source"])}" -> "{_dot_escape(tgt)}" [label="{label}"];'
        )
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        action="store_true",
        help="Include ClassDef nodes and contains/base edges (larger graph).",
    )
    parser.add_argument(
        "--dot-max-edges",
        type=int,
        default=800,
        help="Cap edges in graph.dot (default 800). Use 0 for no cap.",
    )
    parser.add_argument(
        "--external-dot",
        action="store_true",
        help="Also write graph_external.dot for third-party import rollup.",
    )
    args = parser.parse_args()

    root = _repo_root()
    pkg = _src_pkg_root(root)
    if not pkg.is_dir():
        print(f"Package root not found: {pkg}", file=sys.stderr)
        return 1

    out = root / "scratch" / "out"
    out.mkdir(parents=True, exist_ok=True)

    graph = build_graph(pkg, include_classes=args.classes)
    cap = None if args.dot_max_edges == 0 else args.dot_max_edges

    graph_path = out / "graph.json"
    graph_path.write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dot_modules(graph, out / "graph.dot", max_edges=cap)
    if args.external_dot:
        write_dot_external(graph, out / "graph_external.dot", max_edges=cap)

    print(f"Wrote {graph_path}")
    print(f"Wrote {out / 'graph.dot'}")
    if args.external_dot:
        print(f"Wrote {out / 'graph_external.dot'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
