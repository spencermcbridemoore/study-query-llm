# Coupling / ACID / Interface Audit — Plan

Status: complete
Started: 2026-04-24
Completed: 2026-04-24
Owner: audit-session

## Charge

Per session request:

> Make liberal use of subagents and really scour the repo in terms of dependency
> and atomic, consistent, isolated, durable (ACID) principles. Are there
> opportunities to decouple functionality or focus more on interfaces? We are at
> a very VERY early stage with few database entries to speak of.

The "early stage with few DB entries" framing is critical: now is the cheapest
time to propose schema seams, protocol surfaces, and module re-cuts, because
later such changes carry data-migration cost.

## Scope

In scope:

- `src/study_query_llm/` (full package, ~149 .py files)
- Coupling/cohesion of subpackages (services, db, pipeline, providers,
  algorithms, datasets/source_specs, experiments, execution, storage)
- ACID-style guarantees of the persistence layer (transactionality, idempotency,
  dual-write risks, repository compliance, connection lifecycle)
- Quality of declared extension points (Protocols, ABCs, registries)

Out of scope (this pass):

- `panel_app/` (UI; coupled by definition to services)
- `notebooks/`
- `tests/` (other than as evidence of testability seams)
- Performance / latency / throughput

## Method

1. **Static analysis scripts** (under `docs/audit/coupling_acid_audit_2026-04-24/`):
   - `import_graph.py` — full intra-package import graph + cycle detection
   - `module_metrics.py` — fan-in / fan-out / module size / SCC report
   - `db_touchpoint_scan.py` — every file that imports DB models / sessions
   - `protocol_inventory.py` — all `Protocol` / `ABC` / `runtime_checkable`
     declarations and their concrete implementers
2. **Four parallel exploration subagents**, each producing a structured report
   stored under `subagent_reports/`:
   - A — Dependency graph & layering / cycles / god-modules
   - B — Persistence & ACID seams (transactions, idempotency, dual-writes)
   - C — Five-stage pipeline coupling (acquire/parse/snapshot/embed/analyze)
   - D — Extension points (providers / source_specs / algorithms)
3. **Synthesis**:
   - `FINDINGS.md` — observations only, with file:line evidence
   - `PROPOSALS.md` — concrete decoupling / interface recommendations,
     prioritized by leverage × cost (cheapest now while DB is empty)

## Deliverables

```
docs/audit/coupling_acid_audit_2026-04-24/
├── PLAN.md                         (this file)
├── import_graph.py
├── module_metrics.py
├── db_touchpoint_scan.py
├── protocol_inventory.py
├── outputs/
│   ├── import_graph.json        (gitignored — see Reproducing below)
│   ├── module_metrics.txt
│   ├── cycles.txt
│   ├── db_touchpoints.txt
│   └── protocol_inventory.txt
├── subagent_reports/
│   ├── A_dependency_layering.md
│   ├── B_persistence_acid.md
│   ├── C_pipeline_coupling.md
│   └── D_extension_points.md
├── FINDINGS.md
└── PROPOSALS.md
```

## Commit Cadence (actual)

- Commit 1: scaffold + scripts + raw outputs + four subagent reports
- Commit 2: synthesis (FINDINGS + PROPOSALS)

## Reproducing

From repo root, with the package importable on `PYTHONPATH`:

```powershell
python docs/audit/coupling_acid_audit_2026-04-24/import_graph.py
python docs/audit/coupling_acid_audit_2026-04-24/module_metrics.py
python docs/audit/coupling_acid_audit_2026-04-24/db_touchpoint_scan.py
python docs/audit/coupling_acid_audit_2026-04-24/protocol_inventory.py
```

Each script writes its primary artifact under `outputs/`. The
`outputs/import_graph.json` is gitignored (matches the global `*.json` rule)
because it is a large intermediate; re-run `import_graph.py` to regenerate.
`outputs/cycles.txt` is the human-readable summary that is committed.

## Lessons Learned (audit-tooling)

- The first run of `import_graph.py` reported a 3-node cycle inside the
  `pipeline` package. Investigation showed the resolver was peeling one too
  many package levels for relative imports inside `__init__.py` files. Per
  PEP 328 the importer's package equals its own dotted name when the importer
  is a package (`__init__.py`), and equals the parent for plain modules. The
  script was rewritten around `file_to_module_and_package` to track both, and
  `resolve_relative_import` now consumes the importer's *package* directly
  with `level - 1` peeling. Once corrected the spurious pipeline cycle
  disappeared and only the (benign) `services.embeddings` self-cycle remained.
  Documented here so future audits don't re-derive this.

## Non-Goals

- This audit does NOT modify any production code. Only adds analysis artifacts
  under `docs/audit/`.
- Proposals are sketches for discussion; nothing here is a unilateral refactor.
