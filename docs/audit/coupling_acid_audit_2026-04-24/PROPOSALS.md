# Proposals — Coupling, ACID, and Interface Audit

Status: synthesis (recommendations)
Date: 2026-04-24
Companion: [FINDINGS.md](./FINDINGS.md), [PLAN.md](./PLAN.md)

These are **proposals for discussion**, not unilateral refactors. Each one is graded by where it falls on a leverage-vs-cost curve, with an explicit "what does early-stage (near-empty DB) buy us?" line.

## Triage Legend

- **T1 — Do before data lands.** Cost is asymmetric: dirt-cheap now, painful (or migration-hostile) once production data exists. Most schema/constraint changes live here.
- **T2 — Do at next refactor opportunity.** Pure code-shape; can be done incrementally; no migration cost. Worth getting right while the surface is small.
- **T3 — Strategic direction.** Larger architectural moves; recommend planning, not jumping in. Often most valuable once T1/T2 are done.

Each proposal includes an `Evidence:` link to the relevant section of `FINDINGS.md` so we can re-derive the rationale later.

---

## Tier 1 — Do Before Data Lands

### T1.1 — Add `UNIQUE` constraint + upsert to `analysis_results`

**Problem.** `analysis_results` lacks any unique constraint over `(method_definition_id, source_group_id, analysis_group_id, result_key)`. `MethodService.record_result` always inserts. Retrying any analyze run silently writes duplicates. Evidence: [FINDINGS §2.2](./FINDINGS.md#22-analysis_results-has-no-uniqueness-constraint).

**Cost now.** One Alembic migration adding the index; one `.on_conflict_do_update(...)` (Postgres) or equivalent in `record_result`. Trivial when the table is small.

**Cost later.** A duplicate-row dedup migration is non-trivial and will need ad-hoc decisions for each analysis run shape.

**Sketch.**

```sql
CREATE UNIQUE INDEX uq_analysis_results_idem ON analysis_results (
  method_definition_id,
  source_group_id,
  COALESCE(analysis_group_id, -1),
  result_key
);
```

```python
def record_result(self, ...):
    stmt = pg_insert(AnalysisResult).values(...).on_conflict_do_update(
        index_elements=[..., COALESCE(...)],
        set_=dict(result_json=..., updated_at=now()),
    )
```

**Risk.** Need to confirm any in-flight workflows are okay being treated as upserts. With the DB nearly empty this is essentially free to verify.

---

### T1.2 — Make `OrchestrationJob.claim` use `FOR UPDATE SKIP LOCKED`

**Problem.** Current claim path does select-then-mutate-in-Python. Two workers racing the same job can both succeed. Evidence: [FINDINGS §2.3](./FINDINGS.md#23-orchestrationjobclaim-has-no-row-lock).

**Cost now.** A single SQL change in `db/raw_call_repository.py` `claim_orchestration_job`. No data migration.

**Cost later.** Same change, but with concurrent workers in production already double-claiming; data corruption / wasted compute risk grows with worker fleet.

**Sketch.**

```python
def claim_next_orchestration_job(self, *, lease_owner: str, ...) -> OrchestrationJob | None:
    row = self.session.execute(text("""
        WITH cand AS (
            SELECT id FROM orchestration_jobs
            WHERE status = 'pending' AND ...
            ORDER BY priority DESC, created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE orchestration_jobs SET status='claimed', leased_by=:owner, leased_at=now()
        FROM cand WHERE orchestration_jobs.id = cand.id
        RETURNING orchestration_jobs.*
    """), {"owner": lease_owner}).first()
    return ... if row else None
```

**Risk.** SQLite (used by some tests) does not support `FOR UPDATE SKIP LOCKED`. Either keep the existing path as a SQLite fallback, or move integration-of-claims tests to Postgres-only.

---

### T1.3 — Add `IntegrityError` recovery to `create_provenanced_run` (and `try_acquire_embedding_cache_lease`)

**Problem.** Both paths do select-then-insert against unique constraints, but neither uses `begin_nested` + `IntegrityError` recovery. Concurrent races force transaction rollback rather than clean upsert. The pattern is already used correctly in `create_group_link`. Evidence: [FINDINGS §2.4](./FINDINGS.md#24-create_provenanced_run-has-no-integrityerror-recovery), [§2.5](./FINDINGS.md#25-embedding-cache-lease-has-the-same-race).

**Cost now.** Two small repository edits modeled on `create_group_link` (`db/raw_call_repository.py` 1027-1049). No data migration.

**Cost later.** Worker fleet scale will surface mysterious transaction rollbacks under load.

**Sketch.**

```python
def create_provenanced_run(self, ...):
    try:
        with self.session.begin_nested():
            self.session.add(run)
            self.session.flush()
        return run
    except IntegrityError:
        self.session.rollback()  # release nested savepoint
        existing = self.session.query(ProvenancedRun).filter_by(
            request_group_id=..., run_key=..., run_kind=...).one()
        # merge fingerprints / update metadata as appropriate
        return existing
```

**Risk.** None significant. Pattern already exists in the same file.

---

### T1.4 — Decide + execute the legacy-vs-canonical lane cutover for execution provenance

**Problem.** `ProvenancedRunService.list_unified_execution_view` synthesizes execution rows from legacy `clustering_run` groups + `analysis_results` and merges them with real `provenanced_runs`. `SweepQueryService.get_sweep_metrics_df` still walks `clustering_run` groups directly, in parallel with the unified view. Synthetic rows have `id=None` — easy footgun for callers. Evidence: [FINDINGS §2.6](./FINDINGS.md#26-the-unified-execution-view-merges-canonical-and-legacy-shapes).

**Cost now.** With the DB nearly empty, a one-shot backfill from `clustering_run` + `analysis_results` -> `provenanced_runs` is small. Then the unified view becomes "just read `provenanced_runs`" and the legacy branches can be deleted.

**Cost later.** Backfill becomes a long migration with edge cases; consumers calcify around the dual shape.

**Sketch.**

1. One backfill script (one-off, can live under `scripts/audits/` for the audit) that walks legacy `clustering_run` groups + `analysis_results` and inserts the matching `provenanced_runs` rows.
2. Add a feature flag `EXECUTION_PROVENANCE_CANONICAL_ONLY` on the read path. Run side-by-side comparison once.
3. Delete the legacy branches in `provenanced_run_service.py` 296-452 and the legacy metrics path in `sweep_query_service.py`.

**Risk.** Loss of legacy-only data if backfill misses something. Mitigated by the side-by-side comparison step.

---

### T1.5 — Decide v1 schema's status and remove it from `db/__init__.py.__all__`

**Problem.** `db/__init__.py` (10-23) re-exports v1 names (`Base`, `InferenceRun`, `DatabaseConnection`, `InferenceRepository`) right next to v2. Encourages accidental v1 use via `from study_query_llm.db import *`. Empirically, **no production source file** imports them directly anymore (`outputs/db_touchpoints.txt` "v1 + v2 dual-import bridges: 0"); only tests still touch v1 (`tests/test_db/`).

**Cost now.** Move v1 re-exports into `study_query_llm.db.compat` (or just delete the re-exports in `__all__`). Update tests that broke.

**Cost later.** Same change but with more accreted v1 callers.

**Sketch.**

```python
# db/__init__.py (after)
from .models_v2 import BaseV2, RawCall, Group, GroupMember, CallArtifact
from .connection_v2 import DatabaseConnectionV2
from .raw_call_repository import RawCallRepository

__all__ = ["BaseV2", "RawCall", "Group", "GroupMember", "CallArtifact",
           "DatabaseConnectionV2", "RawCallRepository"]

# db/compat.py (new — explicit, narrow)
from .models import Base, InferenceRun
from .connection import DatabaseConnection
from .inference_repository import InferenceRepository
__all__ = ["Base", "InferenceRun", "DatabaseConnection", "InferenceRepository"]
```

**Risk.** Test files that `from study_query_llm.db import InferenceRun` need updating. That's the audit.

---

## Tier 2 — Code-Shape Wins (Incremental, No Migration Cost)

### T2.1 — Add `RawCallRepository.list_group_artifacts(group_id, artifact_types=...)` and remove the 5-way duplication

**Problem.** Every pipeline stage has its own copy of `_call_artifact_uri_by_id(repo, artifact_id)` and broad `repo.session.query(CallArtifact).filter(...)` calls. Evidence: [FINDINGS §3, §5](./FINDINGS.md#3-repository-pattern-drift-service-layer-leaks).

**Cost now.** One repository method + one tiny `pipeline/_artifact_lookup.py` helper. Replace 5 copies. ~30 minutes of changes.

**Sketch.**

```python
# db/raw_call_repository.py
def get_artifact_uri(self, artifact_id: int) -> str | None:
    row = self.session.query(CallArtifact).filter(
        CallArtifact.id == int(artifact_id)).first()
    return row.uri if row else None

def list_group_artifacts(
    self, group_id: int, artifact_types: list[str] | None = None
) -> list[CallArtifact]:
    q = self.session.query(CallArtifact).filter(CallArtifact.group_id == group_id)
    if artifact_types:
        q = q.filter(CallArtifact.artifact_type.in_(artifact_types))
    return q.order_by(CallArtifact.id.asc()).all()
```

Then `pipeline/{acquire,parse,snapshot,embed,analyze}.py` each delete their `_call_artifact_uri_by_id` and call `repo.get_artifact_uri(...)`.

**Risk.** None.

---

### T2.2 — Centralize `_resolve_db` for pipeline stages

**Problem.** Five copies of the same function. Evidence: [FINDINGS §5](./FINDINGS.md#5-pipeline-findings-validated).

**Cost now.** Create `pipeline/_db.py` with `resolve_db(...)`. Replace 5 imports. Trivial.

**Risk.** None.

---

### T2.3 — Either implement `cosine_kllmeans_no_pca` runner or remove it from `COMPOSITE_RECIPES`

**Problem.** Recipe is registered, DB rows reference it, but no builtin runner dispatches it — it falls through to the default summary runner. **Latent provenance/behavior mismatch.** Evidence: [FINDINGS §8](./FINDINGS.md#8-the-composite-method-no-runner-bug).

**Cost now.** Pick one:
- (a) Implement the runner. Ship it as `pipeline/clustering/cosine_kllmeans_runner.py` and add to the dispatcher.
- (b) Drop `cosine_kllmeans_no_pca` from `COMPOSITE_RECIPES` until a runner exists.
- (c) Add a runtime guard: when a method name is in `COMPOSITE_RECIPES` but absent from the dispatcher, raise rather than silently falling through.

**Recommendation.** Do (c) immediately (5-line change) regardless of which of (a)/(b) ends up shipped. (c) is the safety net.

---

### T2.4 — Convert `_resolve_builtin_method_runner` to a registry

**Problem.** String `if/elif` dispatch in `pipeline/analyze.py` 455-463. New methods require editing analyze. Evidence: [FINDINGS §5](./FINDINGS.md#5-pipeline-findings-validated), [Subagent C §7 item 2](./subagent_reports/C_pipeline_coupling.md).

**Cost now.** A small `dict[str, AnalysisRunner]` populated at import time from `pipeline.clustering` + `pipeline.hdbscan_runner`. Empty rows raise (T2.3 safety net).

**Sketch.**

```python
# pipeline/clustering/__init__.py (or new pipeline/_method_registry.py)
BUILTIN_METHOD_RUNNERS: dict[str, AnalysisRunner] = {
    "kmeans+silhouette+kneedle": run_kmeans_silhouette_kneedle_analysis,
    "gmm+bic+argmin": run_gmm_bic_argmin_analysis,
    "hdbscan": run_hdbscan_analysis,
    # cosine_kllmeans_no_pca: when implemented
}
```

**Risk.** None — `analyze.py` becomes a one-liner lookup.

---

### T2.5 — Move `pipeline/hdbscan_runner.py` into `pipeline/clustering/`

**Problem.** Asymmetric packaging: kmeans/gmm runners are inside `pipeline/clustering/`, hdbscan is a sibling. Same callable shape; arbitrary location. Evidence: [FINDINGS §5](./FINDINGS.md#5-pipeline-findings-validated).

**Cost now.** Move the file; keep a thin re-export at `pipeline/hdbscan_runner.py` for one release if anything imports it externally.

**Sketch.**

```python
# pipeline/hdbscan_runner.py (compat shim, after move)
"""Compat shim — moved to pipeline.clustering.hdbscan_runner; kept for one release."""
from .clustering.hdbscan_runner import *  # noqa
```

**Risk.** Need to check imports across the codebase. Probably <10 references.

---

### T2.6 — Replace `if/elif` factories with name-keyed registries (`ProviderFactory`, `StorageBackendFactory`)

**Problem.** String `if/elif` chains for provider creation; advertised lists drift from implemented branches. Evidence: [FINDINGS §7](./FINDINGS.md#7-interface-quality), [Subagent D §7 item 1](./subagent_reports/D_extension_points.md).

**Cost now.** Two registries:

```python
# providers/factory.py
_CHAT_PROVIDERS: dict[str, Callable[..., BaseLLMProvider]] = {
    "azure": _make_azure_chat,
    "openrouter": _make_openrouter_chat,
    ...
}

@classmethod
def create_chat_provider(cls, name: str, model: str | None = None) -> BaseLLMProvider:
    try:
        return cls._CHAT_PROVIDERS[name](model)
    except KeyError:
        raise ValueError(f"Unknown chat provider {name!r}; "
                         f"available: {sorted(cls._CHAT_PROVIDERS)}")

@classmethod
def get_available_chat_providers(cls) -> list[str]:
    return sorted(cls._CHAT_PROVIDERS)
```

`get_available_*` is now derived from keys, eliminating drift.

**Risk.** None significant.

---

### T2.7 — Make `ParserCallable` a real `Protocol`

**Problem.** It's currently a `Callable[[ParserContext], Iterable[SnapshotRow]]` alias. Cannot be `isinstance`-checked, cannot grow optional default helpers. Evidence: [FINDINGS §7](./FINDINGS.md#7-interface-quality), [Subagent D §7 item 3](./subagent_reports/D_extension_points.md).

**Sketch.**

```python
# datasets/source_specs/parser_protocol.py
@runtime_checkable
class SnapshotParser(Protocol):
    def __call__(self, ctx: ParserContext) -> Iterable[SnapshotRow]: ...

# Keep ParserCallable as alias for back-compat:
ParserCallable = SnapshotParser
```

Future per-dataset registration helpers can attach default behavior or sanity-check `isinstance(p, SnapshotParser)`.

**Risk.** None.

---

### T2.8 — Symmetric chat/embedding provider lifecycles (`close()` on chat ABC)

**Problem.** `BaseEmbeddingProvider` has `close()` and async context-manager methods; `BaseLLMProvider` does not. Resource cleanup is asymmetric. Evidence: [Subagent D §7 item 4](./subagent_reports/D_extension_points.md).

**Sketch.** Add `async def close(self) -> None:` (default: `pass`) and `__aenter__/__aexit__` to `BaseLLMProvider`, or extract a shared `BaseAsyncProvider` parent.

**Risk.** None — additive.

---

### T2.9 — Untangle the benign `services/embeddings/` static cycle

**Problem.** `service.py` line 50 does `from . import persistence`, creating a static cycle with the package init and `helpers`. Benign at runtime by Python's submodule rules but fragile. Evidence: [FINDINGS §9](./FINDINGS.md#9-the-one-real-cycle), `outputs/cycles.txt`.

**Sketch.** Replace `from . import persistence; persistence.upsert_embedding_cache_entry(...)` with an explicit `from .persistence import upsert_embedding_cache_entry` (which goes to a sibling submodule, not back through the package init).

**Risk.** None.

---

### T2.10 — Verify and either revive or remove the four orphan modules

**Problem.** Four modules (totaling **>1,300 LOC**) appear with fan-in/fan-out 0 in the static graph. Some are likely false positives from package-rolled-up imports; others (e.g. `domain/representation_hierarchy.py` at 761 lines, `algorithms/canonical_configs.py` at 335 lines per its own dormant docstring) may be genuinely unused. Evidence: [FINDINGS §6](./FINDINGS.md#6-the-orphan-modules).

**Sketch.** For each orphan, run `rg -n '<symbol_name>' src/ tests/ scripts/ panel_app/ notebooks/` for the major exports. Either:
- (a) Document the dynamic-import path in the module docstring (e.g. "loaded by `providers/factory.py` via `from .managers import ACITEIManager`"), or
- (b) Delete it.

`domain.representation_hierarchy` deserves manual eyes; 761 lines of "is this in use?" is a real question.

---

## Tier 3 — Strategic Direction

### T3.1 — Split `RawCallRepository` into per-aggregate repositories

**Problem.** 1701 lines, fan-in 30. The single largest god module in the package. Mixes concerns: raw calls, groups, group_members, group_links, call_artifacts, provenanced runs, orchestration jobs, embedding cache, sweep claims, etc. Evidence: [FINDINGS §4](./FINDINGS.md#4-god-modules-validated-by-both-subagents-and-scripts).

**Approach.** Introduce per-aggregate repositories:

```
db/repositories/
    raw_calls.py           # RawCall CRUD
    groups.py              # Group, GroupMember, GroupLink
    artifacts.py           # CallArtifact
    orchestration.py       # OrchestrationJob*, dependencies
    provenanced_runs.py    # ProvenancedRun, fingerprints
    embedding_cache.py     # EmbeddingCacheEntry, EmbeddingCacheLease
    sweep_claims.py        # SweepRunClaim
```

Keep the existing `RawCallRepository` as a **thin facade** that composes the per-aggregate repositories during the transition. The facade preserves call sites.

**Cost.** Real engineering — multi-PR. T3, not T2.

**Why now-ish.** It is meaningfully cheaper to split when the DB is empty because integration tests don't need to round-trip much data; the surface area of behavior-preservation tests is small.

---

### T3.2 — Move provenance/method orchestration out of `pipeline/analyze.py`

**Problem.** `analyze.py` is 1072 lines and does I/O loading, slicing, representation derivation, method dispatch, MethodService writes, ProvenancedRunService writes, locking. Evidence: [FINDINGS §5](./FINDINGS.md#5-pipeline-findings-validated), [Subagent C §7 item 5](./subagent_reports/C_pipeline_coupling.md).

**Approach.** Decompose:

```
pipeline/analyze/
    __init__.py          # re-export public `analyze()`
    prepare.py           # load + slice + representation derivation
    methods.py           # method runner registry (T2.4)
    persist.py           # finalize_db: provenance + method-result writes
    orchestrate.py       # the public `analyze()` thin composition
```

This makes T3 easier because per-file ownership becomes legible.

**Cost.** Multi-PR. T3.

---

### T3.3 — Make `run_stage` cover idempotent reuse, not just new runs

**Problem.** Each stage implements its own pre-`run_stage` "is there an existing matching group" check, and on hit returns a `StageResult` directly without going through `run_stage`. The reuse path and the run path duplicate `StageResult` shaping logic. Evidence: [Subagent C §2](./subagent_reports/C_pipeline_coupling.md).

**Approach.** Extend `run_stage` with optional `idempotency_lookup: Callable[[RawCallRepository], StageResult | None]`. If supplied and returns a hit, `run_stage` short-circuits (still records a "reused" provenance row if appropriate). One entry point owns both branches.

**Cost.** Worthwhile T3 — cleans the contract considerably.

---

### T3.4 — Re-cut `experiments/` vs `services/jobs/` boundary

**Problem.** `experiments/sweep_worker_main.py` (1354 lines, fan-out 21) and `experiments/runtime_sweeps.py` (717 lines) overlap with `services/jobs/runtime_workers.py` and `services/jobs/runtime_supervisors.py`. Subagent A judged `experiments/` as "grab-bag." Evidence: [Subagent A §5](./subagent_reports/A_dependency_layering.md).

**Approach.** Pick one:
- (a) Merge worker-runtime concerns into `services/jobs/`. Keep `experiments/` for **probes and one-off scripts** only (MCQ probes, ingestion).
- (b) Promote `experiments/` to a top-level "worker-runtime" package and rename it.

Either is fine; the current shape is the worst of both.

**Cost.** Multi-PR rename + import-fix work. Lower-priority than T1/T2.

---

### T3.5 — Introduce `MethodRegistryPort` in `algorithms/`

**Problem.** `algorithms/recipes.py` and `algorithms/text_classification_methods.py` declare `if TYPE_CHECKING: from ..services.method_service import MethodService` and accept `MethodService` instances as parameters. Algorithm-layer module is **typed against** a service-layer concrete class. Evidence: [FINDINGS §1](./FINDINGS.md#1-the-architecture-mostly-holds), [Subagent A §6 item 1](./subagent_reports/A_dependency_layering.md).

**Approach.**

```python
# algorithms/_registry_port.py (or domain/)
class MethodRegistryPort(Protocol):
    def upsert_method_definition(self, name: str, version: str, ...) -> int: ...
    def ensure_recipe(self, ...) -> None: ...

# algorithms/recipes.py
def register_clustering_components(registry: MethodRegistryPort) -> None: ...

# services/method_service.py — MethodService already implements the port structurally;
# add an explicit comment + isinstance test to keep them in sync.
```

**Cost.** Small. Pure code-shape; valuable as a layering signal.

---

### T3.6 — Panel UI facade over services (not over ORM)

**Problem.** Panel views import v2 models and `RawCallRepository` directly (`panel_app/helpers.py` 10-11, 53-58; `panel_app/views/inference.py` 9-10; etc.). UI code knows ORM row shapes. Evidence: [Subagent A §1, §6 item 2](./subagent_reports/A_dependency_layering.md).

**Approach.** Introduce a `PanelDataService` (or `ReadModelService`) returning DTOs (TypedDict / dataclass / Pydantic). Panel views never see ORM objects. Keeps `panel_app/` outside the v1/v2 schema decisions and lets the unified-execution view (T1.4) become the single read shape.

**Cost.** Moderate; do incrementally view-by-view.

---

## Things I Considered and Recommend AGAINST

These came up in subagent reports but I think the cost outweighs the benefit at this stage:

1. **Splitting `services/embeddings/` further.** It's 769 lines but it's a single coherent vertical (cache + provider + persistence + API). Leave it.
2. **Splitting `pipeline.types`.** It's the 5th-most-imported module in the package precisely because it's a clean shared-types seam. Don't fragment it.
3. **Inheriting Protocols (`class X(StorageBackend)`).** The `@runtime_checkable` + structural pattern is fine and is the idiomatic modern Python style. Adding `class LocalStorageBackend(StorageBackend)` doesn't pay for itself.
4. **A general-purpose plugin/discovery system for source-specs or providers** (e.g. entry-points). Premature for the current scale; the registry-as-dict approach (T2.6) is sufficient and easier to reason about.
5. **A unified "stage idempotency key" object with cryptographic signing.** Current per-stage hash strings are sufficient.

---

## Suggested Sequencing

If we did a single focused refactor pass, here's a sensible order (each row is one PR-sized unit):

| # | Tier | Work | Why this slot |
|---|------|------|---------------|
| 1 | T2 | T2.1 (repository artifact methods) + T2.2 (centralize `_resolve_db`) | Highest-leverage code-shape fix; enables reasoning about the next ones. |
| 2 | T2 | T2.3 + T2.4 + T2.5 (method registry + cosine fix + hdbscan move) | Closes the "silent provenance mismatch" hazard and untangles dispatch. |
| 3 | T1 | T1.1 (analysis_results uniqueness) + T1.3 (provenanced-run integrity recovery) | Schema-shaped fixes; cheapest now. |
| 4 | T1 | T1.2 (orchestration claim FOR UPDATE SKIP LOCKED) + T1.5 (db __init__ cleanup) | Concurrency + import hygiene. |
| 5 | T1 | T1.4 (legacy execution-provenance cutover) | Bigger; needs feature flag + side-by-side comparison. |
| 6 | T2 | T2.6 + T2.7 + T2.8 (registries + Protocol + symmetric ABCs) | Interface polish. |
| 7 | T2 | T2.9 (embeddings cycle) + T2.10 (orphan triage) | Hygiene. |
| 8 | T3 | T3.1 (split `RawCallRepository`) | Now cheap because T2.1 already extracted the most-used artifact methods. |
| 9 | T3 | T3.2 + T3.3 (analyze split + run_stage idempotency) | Pipeline modernization. |
| 10 | T3 | T3.4 + T3.5 + T3.6 (experiments boundary, MethodRegistryPort, Panel facade) | Strategic clean-up. |

---

## Summary Heuristic

If you do nothing else, do **T1.1, T1.2, T1.3** and **T2.1, T2.3** in the next pass. Those five together close the most expensive-to-fix-later issues and the most likely-to-bite-soon code-shape bug, with low risk and small diffs.
