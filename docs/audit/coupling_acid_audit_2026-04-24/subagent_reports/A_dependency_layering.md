# Subagent A — Dependency Graph & Layering

Status: raw subagent finding (verbatim, edited only for fenced code blocks)
Authored by: explore subagent A, 2026-04-24
Scope: cross-package imports, layering violations, cycles, god-modules, v1/v2 cohabitation.

---

## 1. Layering Violations

**Declared target** (`docs/living/ARCHITECTURE_CURRENT.md`): Panel UI -> service layer -> providers + `RawCallRepository` -> v2 DB. `.cursorrules` adds: `algorithms/` should stay minimal-deps (not tied to DB/LLM orchestration).

| Finding | Evidence |
|--------|----------|
| **`algorithms/` references `services.method_service` (static coupling)** | `src/study_query_llm/algorithms/recipes.py` 47-48 (`if TYPE_CHECKING:` + `from ..services.method_service import MethodService`). `src/study_query_llm/algorithms/text_classification_methods.py` 43-44 (same pattern). Public helpers take `MethodService` as a parameter (e.g. `register_clustering_components` in `recipes.py` ~310+), so the algorithm package's API is typed against the service layer. **Runtime** imports from `services/` do not run at module load (TYPE_CHECKING only), but **layering and typing still depend "upward."** |
| **No `algorithms/` -> `db/`, `pipeline/`, or `providers/`** | Grep over `algorithms/`: no `study_query_llm.db`, `..db`, `..pipeline`, or `..providers` imports. |
| **Panel bypasses the service layer for v2 DB** (UI -> DB/repo/models directly, not only via services) | `panel_app/helpers.py` 10-11 (`DatabaseConnectionV2`, `RawCallRepository`), 53-58 (lazy `from study_query_llm.db.models_v2 import ...`). `panel_app/views/inference.py` 9-10. `panel_app/views/analytics.py` 10+, `storage_stats.py` 13, `embeddings.py` 5, `groups.py` 32 (lazy), `sweep_explorer_load.py` 121, `sweep_explorer_perspective.py` 187, `sweep_explorer.py` 925 (lazy). |
| **`db/` does not import `services/` or `pipeline/`** | No `from ..services` / `study_query_llm.services` under `src/study_query_llm/db/` (only docstrings mention "pipeline" in `db/migrations/add_recipe_json_column.py` 5 and `db/models_v2.py` 559). |
| **`providers/` does not import `services/`, `db/`, or `pipeline/`** | Absolute imports in `providers/` are LLM/embedding/infra; config + `utils` only. |

---

## 2. Import Cycles

**`TYPE_CHECKING` (type-check-only imports)** — used to avoid runtime cycles / trim load-time deps, e.g.:

- `algorithms/recipes.py` 43-48, `algorithms/text_classification_methods.py` 39-44 -> `MethodService`
- `services/method_service.py` 45-51 -> `RawCallRepository`
- `services/artifact_service.py` 41-44 -> `RawCallRepository`, `StorageBackend`
- `services/inference_service.py` 25-39 -> `RawCallRepository`
- Several `datasets/source_specs/*`, `services/embeddings/*`, `storage/factory.py`, etc.

**Late imports inside functions** (deferred / optional / cycle-avoidance):

- `services/method_service.py` 110+ (and similar blocks 170, 206, 253, 292): `from ..db.models_v2 import ...` inside methods.
- `services/artifact_service.py` 91: `from ..storage.factory import StorageBackendFactory` inside `_resolve_default_backend`.
- `services/artifact_service.py` 430, 722: `from ..db.models_v2 import CallArtifact`.
- `services/provenance_service.py` 205, 349, 390: `from ..db.models_v2 import ...`.
- `services/inference_service.py` 491: `from ..db.models_v2 import Group`.
- `services/data_quality_service.py` 55: `from ..db.models_v2 import Group`.
- `algorithms/method_plugins.py` 79: `import hdbscan` inside code path; `pipeline/hdbscan_runner.py` 196: same (optional dep).
- `services/embeddings/service.py` 101-102: dynamic provider/config imports.

**Actual circular import between subpackages:** Not observed as a **runtime** import loop from this search: `algorithms` does not import `services` at runtime; `db` does not import `services`. **Near-cycle shape:** `method_service` <-> `models_v2` is one-way at runtime (service -> ORM); late imports suggest vigilance if `models_v2` ever imported services.

---

## 3. God Modules (high fan-out OR high fan-in)

**Size (line counts, `src/study_query_llm`, physical lines via Python walk):** top files **>600** lines:
`db/raw_call_repository.py` **1701**; `experiments/sweep_worker_main.py` **1354**; `services/sweep_request_service.py` **1187**; `pipeline/analyze.py` **1072**; `services/artifact_service.py` **1039**; `db/models_v2.py` **904**; `services/embeddings/service.py` **769**; `domain/representation_hierarchy.py` **761**; `services/provenance_service.py` **740**; `experiments/runtime_sweeps.py` **717**; `services/sweep_query_service.py` **675**; `services/inference_service.py` **623**; `algorithms/clustering.py` **611**.

**Fan-in (distinct repo `.py` files with `from study_query_llm.<module> import`, approximate):**
`db/connection_v2` **119**; `db/raw_call_repository` **89**; `db/models_v2` **74**; `services/provenance_service` **33**; `pipeline/types` **28**; `utils/logging_config` **24**; `services/sweep_request_service` **20**; `datasets/source_specs/registry` **20**; `services/method_service` **19**; `pipeline/parse` **18**.

**Top 10 by size (cohesion note):**

1. **`raw_call_repository.py`** — **Repository + many query paths**; feels like **data-access "god"** (high fan-in **89**).
2. **`sweep_worker_main.py`** — **Worker/orchestration + MCQ/sweep**; **fat entrypoint / manager**.
3. **`sweep_request_service.py`** — **Sweep request domain**; large but **one vertical**; still "does everything for sweeps."
4. **`pipeline/analyze.py`** — **Stage + services + algorithms**; **orchestration hub** (cohesive as "analyze stage," heavy).
5. **`artifact_service.py`** — **Artifacts + storage + DB**; **multi-responsibility service** (fan-in **17**).
6. **`models_v2.py`** — **ORM schema**; **expected** to be large (fan-in **74**).
7. **`embeddings/service.py`** — **Embedding pipeline**; cohesive **service** (fan-in **13**).
8. **`representation_hierarchy.py`** — **Math/domain**; single concept, **long** by nature.
9. **`provenance_service.py`** — **Provenance rules**; cohesive theme, high fan-in **33**.
10. **`runtime_sweeps.py`** — **Sweep execution**; overlaps **experiments** cluster with `sweep_worker_main`.

**Top 10 by fan-in (cohesion note):**

1. **`connection_v2`** (119) — **infra singleton**; narrow role.
2. **`raw_call_repository`** (89) — **data hub**; **god-y**.
3. **`models_v2`** (74) — **schema**; OK.
4. **`provenance_service`** (33) — **orchestration**; many callers.
5. **`pipeline/types`** (28) — **shared types**; good seam.
6. **`logging_config`** (24) — **cross-cutting**; expected.
7. **`sweep_request_service`** (20) — **sweep** vertical.
8. **`source_specs/registry`** (20) — **registry**; focal.
9. **`method_service`** (19) — **method registry**; focal.
10. **`pipeline/parse`** (18) — **stage** surface.

---

## 4. v1/v2 Cohabitation Coupling

**Files that import both v1 and v2 "lanes" (connection/repo/models):**

- **`src/study_query_llm/db/__init__.py`** 10-23 — exports `Base`, `InferenceRun`, `DatabaseConnection`, `InferenceRepository` **and** v2 `BaseV2`...`RawCallRepository`. **Primary package-level bridge.**
- **`tests/test_db/test_migration_v1_to_v2.py`** 7-11 — `DatabaseConnection`, `DatabaseConnectionV2`, `InferenceRepository`, `RawCallRepository`, `InferenceRun` — **intentional migration test bridge.**

**Production code outside `db/__init__.py`:** v1 is largely confined to `db/connection.py`, `db/models.py`, `db/inference_repository.py` and **tests** (`test_repository`, `test_connection`, `test_models`, `test_batch_tracking`, `test_e2e_verification.py`). **No other `src/` file** was found importing `InferenceRepository` (grep).

---

## 5. Subpackage Cohesion (brief)

| Subpackage | Judgment |
|------------|----------|
| **`algorithms/`** | **Cohesive** (math/sweep/plugins). **Exception:** `recipes` / `text_classification_methods` **registration API** drags in **`MethodService`** types (47-48, 43-44). |
| **`analysis/`** | **Small, MCQ-oriented**; `mcq_analyze_request` pulls **db + services + experiments types** — **script-like** but one workflow. |
| **`cli/`** | **Entrypoints**; typical for a CLI package. |
| **`datasets/`** | **Cohesive** (acquisition, source specs); `registry` is a **natural hub** (fan-in **20**). |
| **`db/`** | **Split identity:** v1 compat + v2 canonical + **very large** `raw_call_repository` + `models_v2`. |
| **`domain/`** | **Single concept** (representation hierarchy); **isolated** from DB. |
| **`execution/`** | **Remote/local runners** (docker/ssh/vastai); small, **cohesive**. |
| **`experiments/`** | **Grab-bag:** sweep workers, MCQ probes, **ingestion**, **runtime_sweeps** — overlaps **`services/jobs`** and **`pipeline`**. `sweep_worker_main` **1354** lines = **orchestration god-module** risk. |
| **`pipeline/`** | **Stage-based**, clear **5-stage** story; `analyze` is **fat** and **calls `services` heavily** (by design, but high coupling to provenance/artifact/method). |
| **`providers/`** | **Cohesive** (LLM/embedding + managers). |
| **`services/`** | **Largest "surface"** — inference, study, artifacts, provenance, sweeps, embeddings, jobs, **etc.**; **necessarily a wide layer**, but many **>600-line** files. |
| **`storage/`** | **Narrow** (backends + factory). |
| **`utils/`** | **Only 6 modules** (logging, text, estela, session, mcq template) — **small grab-bag** but not huge; **`logging_config`** is **omnipresent** (fan-in **24**). |

---

## 6. Top 5 Decoupling Opportunities

1. **What:** `algorithms` type/API coupling to **`MethodService`**.
   **Where:** `algorithms/recipes.py` 47-48; `text_classification_methods.py` 43-44; `register_*` functions taking **`MethodService`**.
   **Why cheap now:** small surface; registration could move or use a **Protocol** defined in `algorithms` or `domain`.
   **Sketch:** Introduce a **`MethodRegistryPort`** (Protocol) in `algorithms` or a neutral module; **implement** it in `services/method_service.py` and pass into registration helpers.

2. **What:** **Panel** reads/writes through **DB + repository** without a **single facade**.
   **Where:** e.g. `panel_app/views/inference.py` 9-10; `helpers.py` 10-11, 53-58.
   **Why cheap now:** few panel modules; a facade prevents UI knowledge of `models_v2` shapes from spreading.
   **Sketch:** Add **`PanelDataService`/`ReadModelService`** returning DTOs; keep ORM in services only.

3. **What:** **`RawCallRepository` monolith** (size **1701**, fan-in **89**).
   **Where:** `src/study_query_llm/db/raw_call_repository.py` (entire file).
   **Why cheap now:** natural seam by **query concern** (calls vs groups vs sweeps) before more callers accrete.
   **Sketch:** **Split** into `repositories/raw_calls.py`, `repositories/groups.py`, or **module-level private helpers** + thin class; or **Protocol per aggregate** for tests.

4. **What:** **`pipeline/analyze.py`** is both **stage** and **provenance/artifact/method** orchestration (**1072** lines).
   **Where:** e.g. top imports in `pipeline/analyze.py` 16-44 (algorithms + db + services).
   **Why cheap now:** `run_stage` already centralizes.
   **Sketch:** Move **provenance-heavy blocks** into **`ProvenanceService`/`MethodService` helpers** with **narrow** pipeline-facing APIs; keep `analyze` as **I/O + sequencing** only.

5. **What:** **`db/__init__.py`** re-exports **v1 and v2** (10-23), encouraging **broad** `from study_query_llm.db import *` usage.
   **Why cheap now:** v1 is narrow; explicit imports reduce **accidental** v1 use.
   **Sketch:** **Split** `db.v1` vs `db.v2` subpackages or **deprecate** v1 from `__all__` with **lint rule**; keep **one** `db.compat` re-export for migration/tests only.
