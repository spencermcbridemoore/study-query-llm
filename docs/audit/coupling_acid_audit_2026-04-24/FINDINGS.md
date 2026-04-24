# Findings — Coupling, ACID, and Interface Audit

Status: synthesis (observations only)
Date: 2026-04-24
Companion: [PROPOSALS.md](./PROPOSALS.md), [PLAN.md](./PLAN.md)
Subagent reports: [A](./subagent_reports/A_dependency_layering.md) · [B](./subagent_reports/B_persistence_acid.md) · [C](./subagent_reports/C_pipeline_coupling.md) · [D](./subagent_reports/D_extension_points.md)
Empirical outputs: [outputs/](./outputs/)

This document records what is true today. Recommendations live in [PROPOSALS.md](./PROPOSALS.md).

## 0. Audit Snapshot (Empirical)

From the four scripts in this folder against `src/study_query_llm/`:

| Metric | Value | Source |
|---|---|---|
| Modules | 140 | `outputs/import_graph.json` |
| Internal import edges | 421 | `outputs/import_graph.json` |
| True import cycles (SCC size > 1) | 1 (size 3, benign) | `outputs/cycles.txt` |
| Files with any DB indicator | 43 | `outputs/db_touchpoints.txt` |
| v1 + v2 dual-import bridges | 0 | `outputs/db_touchpoints.txt` |
| Protocols declared | 7 | `outputs/protocol_inventory.txt` |
| ABCs declared | 2 | `outputs/protocol_inventory.txt` |
| Protocols with subclassing implementers | 0 of 7 | `outputs/protocol_inventory.txt` |
| ABCs with subclassing implementers | 2 of 2 (Azure + OpenAI-compat for both) | `outputs/protocol_inventory.txt` |
| Modules >= 600 lines | 13 | `outputs/module_metrics.txt` |
| Modules with fan-in >= 25 | 4 | `outputs/module_metrics.txt` |
| God-module candidates (fan-in >= 15 AND lines >= 600) | 2 (`raw_call_repository`, `models_v2`) | `outputs/module_metrics.txt` |

> **Methodology note.** The first run of `import_graph.py` reported a 3-module cycle through `pipeline / pipeline.analyze / pipeline.clustering`. That was a **false positive caused by a bug in the script's own relative-import resolution** (the analyzer was peeling one extra package level for `__init__.py` files). After fixing per PEP 328, that cycle disappeared and a different one — through `services/embeddings/__init__.py` — was surfaced. Subagent C's report was written before the fix; its claim that "no cycles between subpackages" stands, the surviving cycle is intra-package within `services/embeddings/`.

## 1. The Architecture Mostly Holds

The declared layering in `docs/living/ARCHITECTURE_CURRENT.md`
(panel -> services -> {providers, repository, jobRuntimes} -> v2 db) and
`.cursorrules`'s "algorithms must not depend on db/LLM layers" rule is **largely respected** in production source code.

Empirical validation (`outputs/db_touchpoints.txt`):

- `algorithms/` -> **0 files** with DB activity
- `providers/` -> **0 files** with DB activity
- `domain/` -> **0 files** with DB activity
- `storage/` -> **0 files** with DB activity
- `utils/` -> **0 files** with DB activity (excl. `logging_config` which is purely cross-cutting)

That's a real architectural achievement worth preserving. The audit's recommendations should not regress these clean boundaries.

The one **typed exception** flagged by Subagent A: `algorithms/recipes.py` (47-48) and `algorithms/text_classification_methods.py` (43-44) declare `if TYPE_CHECKING: from ..services.method_service import MethodService` and accept `MethodService` instances as parameters of `register_*` helpers. The runtime import does not happen, but the **public API of an "algorithms" module is parameterized by a service-layer concrete class**. This is a typing-level layering inversion (db-shaped registry depends on a higher-layer service type).

## 2. Real ACID/Atomicity Risks

The most consequential findings are persistence-layer. Evidence in [B_persistence_acid.md](./subagent_reports/B_persistence_acid.md) §2-§4.

### 2.1 `run_stage` splits one logical stage across three transactions

`pipeline/runner.py` lines 98-178 opens **three separate `session_scope()` blocks**: (1) create group + provenanced run = `running`, (2) blob writes + `CallArtifact` rows, (3) `finalize_db` + run = `completed`. A **fourth** session opens on failure to mark `failed`.

Failure modes that exist today:

- (2) commits but (3) crashes -> blobs and `CallArtifact` rows are persisted, run row stuck `running` until the failure handler runs in a fresh session.
- (1) commits but (2) crashes -> empty group + `running` run -> orphaned identity rows.
- The failure-handler session (4) is **not transactional with** the failure cleanup; if the process dies between (3) failure and (4) commit, the run stays `running` forever.

There is **no compensating delete for blobs** if the DB step that should have referenced them fails.

### 2.2 `analysis_results` has no uniqueness constraint

`db/models_v2.py` 670-672 — no `UNIQUE` over `(method_definition_id, source_group_id, analysis_group_id, result_key)`. `services/method_service.py` 255-265 (`record_result`) always **inserts**. Retrying any analyze run **silently duplicates rows**.

This is the cleanest "fix-before-DB-fills" example in the audit.

### 2.3 `OrchestrationJob.claim` has no row lock

`db/raw_call_repository.py` 1218-1256 selects candidate jobs and mutates the first eligible row in Python. Two workers can race the same job; nothing in the SQL prevents both from claiming it. PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED` is the standard fix.

Note: enqueue *is* protected (`begin_nested` + `IntegrityError` recovery on unique `job_key` at 1183-1191) — but claim isn't.

### 2.4 `create_provenanced_run` has no `IntegrityError` recovery

`db/raw_call_repository.py` 1518-1596 does select-then-insert with a unique key on `(request_group_id, run_key, run_kind)` (`models_v2.py` 787-793) but, **unlike `create_group_link`** (1027-1049), does **not** wrap in `begin_nested` + catch `IntegrityError`. Concurrent races force a transaction rollback instead of a clean upsert.

### 2.5 Embedding-cache lease has the same race

`db/raw_call_repository.py` 920-948 (`try_acquire_embedding_cache_lease`) is read-then-insert against a primary-key column (`models_v2.py` 884-886). Same fix: `begin_nested` + `IntegrityError` recovery (or `INSERT ... ON CONFLICT DO NOTHING RETURNING`).

### 2.6 The unified-execution view merges canonical and legacy shapes

`services/provenanced_run_service.py` 296-452 (`list_unified_execution_view`) emits **synthetic rows** (with `id=None`) reconstructed from legacy `clustering_run` groups + `analysis_results`, alongside real `provenanced_runs` rows. `services/sweep_query_service.py` 220-228 delegates to it; `get_sweep_metrics_df` (264-271) still walks `clustering_run` groups directly.

Risks: double-counting if a row is represented in both shapes; consumer code may forget that some rows have no PK; "v2 canonical" claim weakens whenever metrics readers prefer the legacy lane.

## 3. Repository-Pattern Drift (Service-Layer Leaks)

The codebase declares a "all DB through repository" pattern. Empirically (`outputs/db_touchpoints.txt`, "Direct session ops outside db/"), this has eroded in the following way:

Many services hold a `repository` instance and then call `self.repository.session.query(...)` to do ad-hoc ORM work that the repository doesn't expose as a method. Examples:

- `services/sweep_query_service.py` — 12 session ops, including `session.query(Group)` (145), `session.query(GroupLink)` (254), filtered Group queries (264).
- `services/method_service.py` — 11 session ops, mixing `session_scope()` and `session.query(MethodDefinition)`.
- `services/sweep_request_service.py` — 10 session ops, `repository.session.query(Group)` (798, 867).
- `services/provenance_service.py` — 7 session ops including `session.add(artifact); session.flush()` (367-368).
- `services/provenanced_run_service.py` — 7 session ops including direct `ProvenancedRun.filter_by(id=...).first()` (290-291).
- `pipeline/analyze.py`, `embed.py`, `parse.py`, `snapshot.py`, `acquire.py` — all do `repo.session.query(CallArtifact).filter(...)` for artifact lookups.

This is the **artifact-URI lookup pattern** that Subagent C flagged in §7 item 4: every stage has its own little `_call_artifact_uri_by_id(repo, artifact_id)` helper that reaches through `repo.session` because the repository doesn't expose a typed equivalent. **The fix is one repository method**, not five.

The `experiments/` package is the worst offender (`sweep_worker_main.py` = 24 session ops, `runtime_sweeps.py` = 21, `ingestion.py` = 9). These are workers — they legitimately own the session lifetime — but they also do raw `session.query(SweepRunClaim)` and `session.add(claim)` in business code that could live behind a `SweepRunClaimRepository`.

## 4. God Modules (Validated by Both Subagents and Scripts)

`outputs/module_metrics.txt` "Top 25 by line count" cross-referenced with fan-in:

| Module | Lines | Fan-in | Comment |
|---|---|---|---|
| `db/raw_call_repository.py` | 1701 | **30** | Confirmed god module: schema-wide repository covering raw calls, groups, group_members, group_links, call_artifacts, provenanced runs, orchestration jobs, embedding cache, sweep claims. |
| `experiments/sweep_worker_main.py` | 1354 | 1 | Single-purpose worker entrypoint, but has fan-out **21** (most outgoing edges in the package). It pulls from everywhere. |
| `services/sweep_request_service.py` | 1187 | 8 | Sweep vertical; cohesive in topic. |
| `pipeline/analyze.py` | 1072 | 0 | Stage entrypoint; fan-out **13**. Is a thin-stage-on-paper, fat-stage-in-practice. |
| `services/artifact_service.py` | 1039 | 8 | DB + storage + naming. |
| `db/models_v2.py` | 904 | **26** | ORM definitions; size is expected. |
| `services/embeddings/service.py` | 769 | 1 | Embedding orchestration; cohesive. |
| `domain/representation_hierarchy.py` | 761 | 0 (orphan!) | Single-concept domain logic; **no other module imports it** in the static graph — see §6. |
| `services/provenance_service.py` | 740 | 9 | Provenance rules. |
| `experiments/runtime_sweeps.py` | 717 | 1 | Worker logic. |

**Hubs (fan-in >= 25)**: only 4. `utils.logging_config` (35), `db.raw_call_repository` (30), `db.connection_v2` (28), `db.models_v2` (26). Three of those are db; one is logging. Splitting the repository would meaningfully reduce one of the only true coupling hotspots in the package.

## 5. Pipeline Findings (Validated)

From [C_pipeline_coupling.md](./subagent_reports/C_pipeline_coupling.md) and confirmed by `outputs/db_touchpoints.txt`:

- All 5 stage modules duplicate a `_resolve_db` helper.
- All 5 stage modules duplicate an inline `repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()` artifact-URI lookup. Empirical evidence: `outputs/db_touchpoints.txt` shows this exact pattern at `pipeline/analyze.py:102`, `pipeline/embed.py:47`, `pipeline/parse.py:195`, `pipeline/snapshot.py:154`, `pipeline/acquire.py:75`.
- `analyze.py` carries **builtin-method string dispatch** in `_resolve_builtin_method_runner` (455-463) for only 3 of the 4 names registered by `recipes.COMPOSITE_RECIPES` (273-277). `cosine_kllmeans_no_pca` is registered as a recipe but **falls through to the default runner** rather than being dispatched. This is a **silent provenance/behavior mismatch**.
- `pipeline/hdbscan_runner.py` is a sibling of `pipeline/clustering/`, while `kmeans_runner.py` and `gmm_runner.py` live **inside** `pipeline/clustering/`. Same callable shape, inconsistent location.
- `run_stage` does **not** parameterize idempotency. Each stage implements its own pre-`run_stage` reuse-check by metadata-equality query and returns a `StageResult` directly when a cached match is found. The reuse and the run paths share little code.
- `pipeline.types` is the only well-loved seam in pipeline-land (fan-in **14** — the 5th highest in the package).

## 6. The "Orphan" Modules

`outputs/module_metrics.txt` flags 8 modules with both fan-in == 0 AND fan-out == 0 in the static graph. These deserve manual review (the static analyzer can miss dynamic registries and `from .managers import ACITEIManager` rollups, but each one needs a justification):

| Orphan module | Lines | Status to verify |
|---|---|---|
| `algorithms.canonical_configs` | 335 | Subagent D notes this is **dormant per its own docstring** ("unused in production"). Confirm and either revive or remove. |
| `domain.representation_hierarchy` | **761** | Largest "orphan". Either it's loaded dynamically and the analyzer missed it, or it's genuinely unused 761-line dead code. **High priority to verify.** |
| `providers.managers.aci_tei` | 423 | Likely loaded via `from .managers import ACITEIManager` in `factory.py` (rollup miss). Verify. |
| `providers.managers.ollama` | 241 | Same. |
| `providers.managers.protocol` | 39 | Same — defines `ModelManager(Protocol)`. |
| `utils.session_utils` | 125 | Cursor session utilities; verify it's still referenced from somewhere. |
| `analysis` (the package init) | 1 | Empty `__init__.py` — fine. |
| `cli` (the package init) | 1 | Empty `__init__.py` — fine. |

The `providers.managers.*` cases are likely false positives from package-level `from .managers import X` imports (the analyzer attributes the edge to the package, not the leaf module). **The `domain.representation_hierarchy` case is not** — Subagent D didn't mention it being used elsewhere either. Worth confirming.

## 7. Interface Quality

Cross-referencing [D_extension_points.md](./subagent_reports/D_extension_points.md) and `outputs/protocol_inventory.txt`:

- **All 7 declared Protocols have 0 subclassing implementers** in the static graph. The implementers exist but rely on **structural duck-typing**. Three of the 7 are `@runtime_checkable` (`StorageBackend`, `ExecutionBackend`, `ModelManager`) so `isinstance` checks at runtime will work; the other 4 (`SweepTypeAdapter`, `JobRunner`, `SupervisorMode`, `WorkerOrchestrator`) are not.
- **2 ABCs** (`BaseLLMProvider`, `BaseEmbeddingProvider`) have proper subclasses: `AzureOpenAIProvider`, `OpenAICompatibleChatProvider`, `AzureEmbeddingProvider`, `OpenAICompatibleEmbeddingProvider`.
- The chat ABC has no `close()`; the embedding ABC has it. Asymmetry per Subagent D §2.
- **`ParserCallable` is a `Callable` alias** (`datasets/source_specs/parser_protocol.py` 33), not a `Protocol`. There is no way to `isinstance(obj, SnapshotParser)` even if you wanted to.
- **`ProviderFactory` advertised lists drift from `create()` branches**. `get_available_providers()` returns `["azure", "local_llm", "ollama"]` (`factory.py` 110) but `create()`'s if-chain only implements `"azure"` (72-80). A user can list a provider that cannot be instantiated.

## 8. The "Composite Method, No Runner" Bug

Subagent D §4 surfaced (and the `analyze.py` line evidence corroborates): `algorithms/recipes.py` `COMPOSITE_RECIPES` lists 4 composite clustering methods including `cosine_kllmeans_no_pca`. The DB-side `ensure_composite_recipe` is called for any name in `COMPOSITE_RECIPES` (`pipeline/analyze.py` 365-388). But the runtime dispatcher `_resolve_builtin_method_runner` (`pipeline/analyze.py` 455-463) maps **only 3 of the 4 names** to runners — `cosine_kllmeans_no_pca` is **silently absent**. That method, if requested, falls through to the default summary runner.

This is not just an interface gap; it's a latent **provenance/behavior mismatch**: the DB will record a method definition matching the recipe, but the actual computation that runs may be the default. The fingerprint-based reuse logic could conceivably reuse a "default-runner" result for a future "real cosine_kllmeans" run.

## 9. The One Real Cycle

`outputs/cycles.txt`: a 3-node SCC inside `services.embeddings`:

```
services.embeddings (package init)
services.embeddings.helpers
services.embeddings.service
```

The `service.py` line 50 does `from . import persistence`, which creates an edge to the package itself. That edge plus `helpers -> service -> ...` and `__init__ -> {helpers, service}` closes the loop.

This is **benign at runtime** because `from . import persistence` triggers loading of the `persistence` *submodule* directly (Python's `_handle_fromlist` machinery), not an attribute lookup against the partially-loaded package. But it's a **code smell**: re-ordering the imports in `__init__.py` (e.g. moving `helpers` before `service`) could expose it.

## 10. v1/v2 Summary (Less Bad Than It Looks On Paper)

Per `outputs/db_touchpoints.txt` "v1 + v2 dual-import bridges": **0 production files** import both v1 and v2 schemas. The only co-export point is `db/__init__.py` (10-23), which still exports v1 names alongside v2 in `__all__`. Tests still touch v1.

So the cohabitation hazard is **schema-shaped** (the `analysis_results` legacy table read by the unified-execution view) rather than **code-shaped** (almost no code imports both schemas). That makes the cleanup cost lower than initial impressions suggested.

---

## Cross-Reference Index of Evidence

For each finding above, the source(s):

| Finding | Subagent | Script |
|---|---|---|
| Layering largely holds | A §1 | `db_touchpoints.txt` (0 leaks in algorithms/providers/domain/storage) |
| `algorithms/` types depend on `MethodService` | A §1, §6 (1) | — |
| `run_stage` is multi-TX | B §2 | — |
| `analysis_results` no uniqueness | B §3, §7 (2) | — |
| Orchestration claim race | B §4, §7 (3) | — |
| `create_provenanced_run` race | B §4, §7 (4) | — |
| Unified-view legacy/canonical merge | B §6, §7 (5) | — |
| Repository-pattern drift | B §1 (LEAK list) | `db_touchpoints.txt` (top-30 leakers) |
| God modules: `raw_call_repository`, `models_v2` | A §3 | `module_metrics.txt` (god-module + hub sections) |
| 5x duplicated `_resolve_db` and artifact lookup | C §2, §7 (1) | `db_touchpoints.txt` (the 5 pipeline files) |
| `cosine_kllmeans_no_pca` recipe with no runner | D §4, §7 (2) | — |
| HDBSCAN packaging asymmetry | C §6, §7 (3) | — |
| 7 Protocols, 0 subclass implementers | D §1 | `protocol_inventory.txt` |
| Provider lists != provider branches | D §2, §7 (1) | — |
| `ParserCallable` not a Protocol | D §1, §7 (3) | — |
| Embeddings package import cycle | (script-only finding) | `cycles.txt` |
| `domain.representation_hierarchy` orphan | (script-only finding) | `module_metrics.txt` |
