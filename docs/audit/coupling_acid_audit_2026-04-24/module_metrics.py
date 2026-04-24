#!/usr/bin/env python3
"""Compute fan-in / fan-out / size metrics from `outputs/import_graph.json`.

READ-ONLY audit script. Run `import_graph.py` first to produce the graph.

Outputs to `docs/audit/coupling_acid_audit_2026-04-24/outputs/`:
    - module_metrics.txt : top-N fan-in, top-N fan-out, top-N by line count,
                           "god module" candidates (high fan-in AND >600 lines).

Usage:
    python docs/audit/coupling_acid_audit_2026-04-24/module_metrics.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC = REPO / "src"
PACKAGE = "study_query_llm"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
GRAPH_PATH = OUTPUT_DIR / "import_graph.json"


def module_to_file(module: str) -> Path | None:
    rel = module.split(".")
    candidate_pkg = SRC.joinpath(*rel) / "__init__.py"
    if candidate_pkg.exists():
        return candidate_pkg
    candidate_mod = SRC.joinpath(*rel[:-1]) / f"{rel[-1]}.py"
    if candidate_mod.exists():
        return candidate_mod
    return None


def line_count(path: Path) -> int:
    try:
        return sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def main() -> int:
    if not GRAPH_PATH.exists():
        print(f"Graph not found at {GRAPH_PATH}. Run import_graph.py first.")
        return 1

    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes: list[str] = sorted(data["nodes"])
    edges: list[list[str]] = data["edges"]

    fan_out: Counter[str] = Counter()
    fan_in: Counter[str] = Counter()
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_neighbors: dict[str, set[str]] = defaultdict(set)

    for src_mod, dst_mod in edges:
        out_neighbors[src_mod].add(dst_mod)
        in_neighbors[dst_mod].add(src_mod)

    for m, outs in out_neighbors.items():
        fan_out[m] = len(outs)
    for m, ins in in_neighbors.items():
        fan_in[m] = len(ins)

    sizes: dict[str, int] = {}
    for m in nodes:
        f = module_to_file(m)
        sizes[m] = line_count(f) if f else 0

    def fmt_top(counter: Counter[str], n: int = 20, value_label: str = "count") -> str:
        lines = []
        for mod, val in counter.most_common(n):
            lines.append(f"  {val:5d}  {mod}")
        return "\n".join(lines)

    by_size = sorted(nodes, key=lambda m: -sizes[m])

    lines: list[str] = []
    lines.append("# Module Metrics")
    lines.append(f"Total modules: {len(nodes)}")
    lines.append(f"Total edges:   {sum(fan_out.values())}")
    lines.append("")

    lines.append("## Top 25 by FAN-IN (most depended-upon)")
    lines.append(fmt_top(fan_in, 25))
    lines.append("")

    lines.append("## Top 25 by FAN-OUT (most depending-on-others)")
    lines.append(fmt_top(fan_out, 25))
    lines.append("")

    lines.append("## Top 25 by LINE COUNT")
    for m in by_size[:25]:
        lines.append(f"  {sizes[m]:5d}  {m}")
    lines.append("")

    lines.append("## God-Module Candidates (fan-in >= 15 AND lines >= 600)")
    god = [m for m in nodes if fan_in[m] >= 15 and sizes[m] >= 600]
    god.sort(key=lambda m: -(fan_in[m] + sizes[m] // 100))
    for m in god:
        lines.append(f"  fan_in={fan_in[m]:4d}  lines={sizes[m]:5d}  {m}")
    if not god:
        lines.append("  (none)")
    lines.append("")

    lines.append("## Hubs (fan-in >= 25)")
    hubs = sorted(nodes, key=lambda m: -fan_in[m])
    for m in hubs:
        if fan_in[m] < 25:
            break
        lines.append(f"  fan_in={fan_in[m]:4d}  lines={sizes[m]:5d}  {m}")
    lines.append("")

    lines.append("## Orphans (fan-in == 0, fan-out == 0) — possibly dead modules")
    orphans = [m for m in nodes if fan_in[m] == 0 and fan_out[m] == 0]
    if orphans:
        for m in sorted(orphans):
            lines.append(f"  {m}  (lines={sizes[m]})")
    else:
        lines.append("  (none)")
    lines.append("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "module_metrics.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Top fan-in: {fan_in.most_common(3)}")
    print(f"Top size:   {[(m, sizes[m]) for m in by_size[:3]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
