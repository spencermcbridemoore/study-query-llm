#!/usr/bin/env python3
"""Inventory every Protocol / ABC / runtime_checkable in the package and find
their concrete implementers (best-effort).

READ-ONLY audit script.

Outputs to `docs/audit/coupling_acid_audit_2026-04-24/outputs/`:
    - protocol_inventory.txt

Usage:
    python docs/audit/coupling_acid_audit_2026-04-24/protocol_inventory.py
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC = REPO / "src"
PACKAGE = "study_query_llm"
PACKAGE_ROOT = SRC / PACKAGE
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    return ""


def extract_methods(cls: ast.ClassDef) -> list[tuple[str, bool]]:
    """Return (name, is_abstract) for each function in the class body."""
    out = []
    for body_item in cls.body:
        if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_abstract = any(
                (isinstance(d, ast.Name) and d.id == "abstractmethod")
                or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                for d in body_item.decorator_list
            )
            out.append((body_item.name, is_abstract))
    return out


def is_runtime_checkable(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        if isinstance(d, ast.Name) and d.id == "runtime_checkable":
            return True
        if isinstance(d, ast.Attribute) and d.attr == "runtime_checkable":
            return True
    return False


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1: find all class defs with their bases.
    class_index: dict[str, list[tuple[str, ast.ClassDef]]] = defaultdict(list)
    # name -> list of (file_rel, classdef)
    file_class_bases: dict[str, dict[str, list[str]]] = defaultdict(dict)
    # file_rel -> {class_name: [base_names]}

    protocols: list[tuple[str, int, ast.ClassDef]] = []   # file, line, cls
    abcs: list[tuple[str, int, ast.ClassDef]] = []

    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        rel_path = rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [base_name(b) for b in node.bases]
            file_class_bases[rel_path][node.name] = bases
            class_index[node.name].append((rel_path, node))
            if any(b in ("Protocol",) for b in bases):
                protocols.append((rel_path, node.lineno, node))
            if any(b in ("ABC", "ABCMeta") for b in bases):
                abcs.append((rel_path, node.lineno, node))

    # Pass 2: implementers for each Protocol/ABC by base-class name match.
    proto_names = {p[2].name for p in protocols}
    abc_names = {a[2].name for a in abcs}

    impl_for: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # base_name -> [(file_rel, child_class_name)]

    for file_rel, classes in file_class_bases.items():
        for child_name, bases in classes.items():
            for b in bases:
                if b in proto_names or b in abc_names:
                    impl_for[b].append((file_rel, child_name))

    out: list[str] = []
    out.append("# Protocol / ABC Inventory")
    out.append(f"Protocols: {len(protocols)}   ABCs: {len(abcs)}")
    out.append("")

    out.append("## Protocols (with @runtime_checkable annotation noted)")
    for file_rel, lineno, cls in sorted(protocols, key=lambda x: (x[0], x[1])):
        rt = "[runtime_checkable]" if is_runtime_checkable(cls) else "[structural]"
        out.append(f"- {cls.name}  {rt}  -- {file_rel}:{lineno}")
        for mname, abstract in extract_methods(cls):
            mark = "*" if abstract else " "
            out.append(f"     {mark} {mname}")
        impls = impl_for.get(cls.name, [])
        if impls:
            out.append(f"    implementers (subclassing): {len(impls)}")
            for ifile, iname in impls:
                out.append(f"        - {iname}  -- {ifile}")
        else:
            out.append("    implementers (subclassing): 0  (likely structural / duck-typed)")
        out.append("")

    out.append("## ABCs")
    for file_rel, lineno, cls in sorted(abcs, key=lambda x: (x[0], x[1])):
        out.append(f"- {cls.name}  -- {file_rel}:{lineno}")
        for mname, abstract in extract_methods(cls):
            mark = "*" if abstract else " "
            out.append(f"     {mark} {mname}")
        impls = impl_for.get(cls.name, [])
        if impls:
            out.append(f"    implementers (subclassing): {len(impls)}")
            for ifile, iname in impls:
                out.append(f"        - {iname}  -- {ifile}")
        else:
            out.append("    implementers (subclassing): 0  (no concrete subclasses found)")
        out.append("")

    out_path = OUTPUT_DIR / "protocol_inventory.txt"
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Protocols: {len(protocols)}   ABCs: {len(abcs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
