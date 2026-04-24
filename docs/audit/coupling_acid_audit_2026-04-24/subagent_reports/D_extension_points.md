# Subagent D — Extension Points & Interfaces

Status: raw subagent finding (verbatim, edited only for fenced code blocks)
Authored by: explore subagent D, 2026-04-24
Scope: Protocols / ABCs, providers, source-spec registry, algorithm/method plugins, storage, provider managers.

---

## 1. Protocol / ABC Inventory

| Name | File:line | Purpose | Concrete implementers (file:line) | Type-check vs duck-type |
|------|-----------|---------|----------------------------------|-------------------------|
| **StorageBackend** | `src/study_query_llm/storage/protocol.py` 14 | Artifact read/write/exists/delete + `get_uri` | `LocalStorageBackend` `local.py` 11; `AzureBlobStorageBackend` `azure_blob.py` 20 (neither subclass explicitly) | Structural + **runtime:** tests use `isinstance(x, StorageBackend)` (`tests/test_storage/test_*.py`). |
| **ExecutionBackend** | `execution/protocol.py` 60 | Container job submit/poll/cancel/logs | `LocalDockerExecution` `local_docker.py` 25; `SSHDockerExecution`; `VastAIExecution` (imported in `execution/factory.py` 30-40) | Structural + **runtime:** `isinstance` in tests. |
| **ModelManager** | `providers/managers/protocol.py` 28 | Model lifecycle: `start`/`stop`/`ping`, sync context manager | `ACITEIManager` `aci_tei.py` 42; `LocalDockerTEIManager` `local_docker_tei.py` 52; `OllamaModelManager` `managers/ollama.py` 39 | Structural + **runtime:** `tests/test_scripts/test_model_manager_protocol.py` asserts `isinstance(mgr, ModelManager)`. **No** inheritance. |
| **JobRunner** | `services/jobs/job_runners.py` 41 | Single `run(job_snapshot, context) -> JobRunOutcome` | `RunKTryRunner` 49; `ReduceKRunner` 90; `FinalizeRunRunner` 119; `McqRunRunner` 148; `AnalysisRunRunner` 184; `LangGraphJobRunner` `langgraph_job_runner.py` 59 | **Not** `@runtime_checkable`. Duck-typed at runtime; `create_job_runner` `job_runner_factory.py` 20-70 returns by `if/elif` on `job_type`. |
| **WorkerOrchestrator** | `services/worker_orchestrator.py` 8 | `run() -> int` worker loop | `StandaloneWorkerOrchestrator` 16; `ShardedWorkerOrchestrator` 27 | **Not** `runtime_checkable`. Factory `create_worker_orchestrator` returns concrete classes typed as `WorkerOrchestrator`. |
| **SupervisorMode** | `services/supervisor_mode.py` 60 | `engine_work_remaining`, `before_progress_poll` | `StandaloneSupervisorMode` 81; `ShardedSupervisorMode` 100+ | **Not** `runtime_checkable`. |
| **SweepTypeAdapter** | `experiments/sweep_request_types.py` 69 | Typed sweep expansion: `sweep_type`, group types, `build_targets`, `analysis_definitions` | `ClusteringSweepAdapter` 205; `McqSweepAdapter` 252; registered in `_SWEEP_TYPE_REGISTRY` 370 | **Not** `runtime_checkable`. Registry holds instances implementing the structural contract. |
| **BaseLLMProvider** (ABC) | `providers/base.py` 102 | Chat: `complete`, `get_provider_name` | `AzureOpenAIProvider` `azure_provider.py` 40; `OpenAICompatibleChatProvider` `openai_compatible_chat_provider.py` 26 | **Explicit inheritance** + `@abstractmethod`. Type hints use `BaseLLMProvider`, not `Protocol`. |
| **BaseEmbeddingProvider** (ABC) | `providers/base_embedding.py` 27 | `create_embeddings`, `get_provider_name`, `close` + optional `validate_model` | `AzureEmbeddingProvider` `azure_embedding_provider.py` 19; `OpenAICompatibleEmbeddingProvider` `openai_compatible_embedding_provider.py` 19; `ManagedTEIEmbeddingProvider` `managed_tei_embedding_provider.py` 81 (extends OpenAI-compatible) | Same: **ABC + inheritance**. |
| **Parser "protocol"** | `datasets/source_specs/parser_protocol.py` | *No* `class ...(Protocol)`.* `ParserContext` is a **dataclass** (22); `ParserCallable` is `Callable[[ParserContext], Iterable[SnapshotRow]]` (33) | Parsers: e.g. `parse_banking77_snapshot` `banking77.py` 73, `parse_estela_snapshot` `estela.py` 102, etc. | **Static typing only:** `ParserCallable` annotates `DatasetAcquireConfig` in `registry.py` and `parse.py` — no `Protocol` class and no `isinstance` against a parser protocol. |

**`@runtime_checkable` summary:** used on `StorageBackend`, `ExecutionBackend`, and `ModelManager` only (plus `ParserCallable` is not a Protocol class).

**Repo-wide `ABC` search:** only `BaseLLMProvider` and `BaseEmbeddingProvider` under `src/` (no other `class ...(ABC)`).

---

## 2. Provider Layer Quality

- **Chat base:** `BaseLLMProvider` is **ABC only**, not a `Protocol` (`providers/base.py` 102-164).
- **Required vs optional:** `complete` and `get_provider_name` are **abstract**. No separate optional section on the chat ABC (unlike `BaseEmbeddingProvider`, which documents optional `validate_model` and default `True` in `base_embedding.py` 71-77).
- **Factory:** `ProviderFactory` uses **string `if/elif` chains** for `create` (lines 67-80), `create_embedding_provider` (148+), `create_chat_provider` (375-384), and `list_provider_deployments` (421-437). `get_available_*` return **hardcoded lists** (e.g. chat 387-389). **No** pluggable name->class registry in `factory.py`.
- **Adding a 5th chat provider (rough touch list):** (1) new module implementing `BaseLLMProvider`; (2) `providers/factory.py` — at minimum `create_chat_provider`, and likely `list_provider_deployments` / `create` if the provider should be first-class; (3) `config.py` / env if new config keys; (4) `providers/__init__.py` if re-exporting; (5) any docs/tests. **~4-6 files** is a reasonable estimate, dominated by the factory and config.
- **Embedding vs chat symmetry:** both use **ABC** and `get_provider_name()`. **Asymmetry:** `BaseEmbeddingProvider` has `close()` (abstract) and async context manager; chat has no `close` on the ABC. `create()` only instantiates `Azure` for "legacy" path while `create_chat_provider` supports Azure + OpenAI-compatible; `create_embedding_provider` special-cases Azure then everything else as `OpenAICompatibleEmbeddingProvider` (`factory.py` 127-158). `get_available_providers()` for `create` returns `["azure", "local_llm", "ollama"]` (110) but `create` only implements `"azure"` in the if-chain — **drift** between listing and `create` implementation (72-80).

---

## 3. Source-Spec Registry Quality

- **Registry style:** `ACQUIRE_REGISTRY` in `registry.py` 80 is a **module-level hardcoded `dict`**, not a decorator or lazy plugin system. New datasets require editing this file and adding imports.
- **`parser_protocol.py`:** defines **dataclass + `Callable` alias**, not a `Protocol`. It is used for **type annotations** and consistency (`parse.py`, `registry.py`); not runtime `Protocol` checking.
- **Adding a dataset (typical steps):** new module (slug constants, `*_file_specs`, `*_source_metadata`, `parse_*_snapshot`); one **`ACQUIRE_REGISTRY` entry** with slug as key; wire **`__init__.py` exports** if public API. **String literals** must match at least: slug constant, dict key, and any callers using the string — **no** slug `Enum` in `registry.py` (plain `str`). Parser id/version are separate string constants for idempotency.
- **Per-dataset modules:** no shared `SourceSpec` ABC; each is a self-contained "snowflake" with the same *shape* (specs + metadata + parser) converging on `ParserContext` and `FileFetchSpec`.

---

## 4. Algorithm / Method Plugin Quality

- **`recipes.py`:** Versioned **JSON recipe specs** + `COMPOSITE_RECIPES`, `canonical_recipe_hash`, `build_composite_recipe`, and DB hooks `register_clustering_components` / `ensure_composite_recipe` (metadata + provenance; docstring says execution stays in algorithm modules) (`recipes.py` 1-35, 273-400).
- **`method_plugins.py`:** Small **runtime plugin envelope** for sweep-style `fixed_k` / `unknown_k` with `run_fixed_k_plugin` / `run_unknown_k_plugin` and `MethodPlugin` dataclass metadata `available_method_plugins()` (`method_plugins.py` 22-135). This is **not** the same as DB method definitions.
- **`canonical_configs.py`:** **Unused** in production per module docstring (lines 1-15): `CANONICAL_CONFIG_BUILDERS` maps `(name, version)` to pure normalizers; intended for future fingerprint stability.

**Overlap / confusion:** three parallel concepts — **recipe JSON** (provenance), **MethodPlugin** keys (`fixed_k`/`unknown_k`), and **canonical_config builders** (dormant) — without a single "one registry" for "how to run a method name."

**Clustering "canonical 4" (recipes):** `COMPOSITE_RECIPES` in `recipes.py` 273-277: `cosine_kllmeans_no_pca`, `kmeans+silhouette+kneedle`, `gmm+bic+argmin`, `hdbscan`. **DB registration** via `register_clustering_components` + `ensure_composite_recipe` from `analyze._resolve_method_definition_id` when `method_name` is in `COMPOSITE_RECIPES` (`analyze.py` 365-388).

**Runtime wiring:** `_resolve_builtin_method_runner` only maps **three** normalized names to functions: `hdbscan`, `kmeans+silhouette+kneedle`, `gmm+bic+argmin` (`analyze.py` 455-463). **`cosine_kllmeans_no_pca` is not** dispatched there — it falls through to `_default_method_runner` unless an external `method_runner` is supplied. **v1** rule resolution in `schema.py` uses `V1_CLUSTERING_METHODS` with **three** names (hdbscan, kmeans..., gmm...) — **not** `cosine_kllmeans` (`schema.py` 15-20).

**New clustering method without `analyze.py`?** **No** for the built-in path: you must extend `_resolve_builtin_method_runner` (and likely provenance/repair paths) and possibly `is_v1_clustering_method` / rules. Recipe-only registration in `recipes` alone does not add execution. **Exception:** a caller can inject a custom `method_runner` into the analyze entrypoint, bypassing the builtin table.

**Algorithms vs DB (`.cursorrules` "minimal dependencies"):** Grep in `algorithms/` shows **no** `psycopg`, `sqlalchemy`, or `db` imports. **Decoupling holds** for `algorithms/`. The **pipeline** (`pipeline/analyze.py`) intentionally couples to DB, services, and method registration.

---

## 5. Storage Layer

- **`storage/factory.py`:** `StorageBackendFactory.create` is **if/elif** on `"local"` / `"azure_blob"` (31-38) — same "strategy by string" as providers, not a self-registering plugin map. Returns type **`StorageBackend` (Protocol) in the annotation** only (`factory.py` 17, TYPE_CHECKING import 10).
- **Protocol vs concrete:** `LocalStorageBackend` and `AzureBlobStorageBackend` **do not** inherit from a base class; they **structurally** satisfy `StorageBackend(Protocol)`.
- **Leaky call sites:** `panel_app/views/storage_stats.py` imports **`AzureBlobStorageBackend` concretely** and branches `if not isinstance(backend, AzureBlobStorageBackend):` (lines 15-16, 304-305) for Azure-only probing — bypassing the `StorageBackend` abstraction for that feature.

---

## 6. Provider Manager Layer

- **`ModelManager(Protocol)`** in `managers/protocol.py` is a **clean, `runtime_checkable` structural contract**; `ACITEIManager`, `LocalDockerTEIManager`, and `OllamaModelManager` **do not** subclass it; comments say TEI managers match duck typing for `ManagedTEIEmbeddingProvider`.
- **Relation to `base_embedding.py`:** **No** inheritance link. `ManagedTEIEmbeddingProvider` extends **`OpenAICompatibleEmbeddingProvider` -> `BaseEmbeddingProvider`**. The manager supplies **`endpoint_url`** after `start()`; the embedding client uses the HTTP API. **Orthogonal concerns:** `ModelManager` = infra lifecycle; `BaseEmbeddingProvider` = request/response embedding API.

---

## 7. Top 5 Interface Improvements

1. **What's missing:** `ProviderFactory` "available providers" and `create`/`create_chat_provider` **lists and branches disagree**; easy to list a name that cannot be constructed.
   **Where:** `providers/factory.py` 67-80, 102-110, 351-384.
   **Why now:** One source of truth prevents subtle runtime `ValueError`s as providers are added.
   **Sketch:** A small `dict[str, Callable[..., BaseLLMProvider]]` (or two dicts: legacy vs chat) filled once, with `get_available_*` derived from keys.

2. **What's missing:** **Recipe/composite method registration in DB vs execution in `analyze`** is split; `cosine_kllmeans_no_pca` gets `ensure_composite_recipe` but **no** builtin runner.
   **Where:** `analyze.py` 365-388 vs 455-463; `recipes.py` `COMPOSITE_RECIPES`.
   **Why now:** Operators see a "registered" method that runs the default embedding-summary path unless a custom runner is injected — **provenance/behavior mismatch risk**.
   **Sketch:** Either document "composite-only metadata in analyze" or add explicit dispatch / raise if recipe implies a runner that is not implemented.

3. **What's missing:** `ParserCallable` is a **`Callable` alias**, not a `Protocol` — no way to `isinstance` or attach default helpers.
   **Where:** `parser_protocol.py` 33.
   **Why now:** Tighter contracts as more parsers land.
   **Sketch:** `class SnapshotParser(Protocol): def __call__(self, ctx: ParserContext) -> Iterable[SnapshotRow]: ...` (or `@runtime_checkable` only if needed).

4. **What's missing:** **Chat vs embedding** bases diverge (no `close` on `BaseLLMProvider`, different factory shapes).
   **Where:** `base.py` 102+; `base_embedding.py` 27+.
   **Why now:** Symmetric resource cleanup and testing doubles.
   **Sketch:** Optional `close` or shared `BaseAsyncProvider` with `get_provider_name` + `close` default no-op for chat.

5. **What's missing:** **Storage** panel path **fixes on Azure concrete type** instead of optional protocol for "list stats / probe."
   **Where:** `panel_app/views/storage_stats.py` 304-305.
   **Why now:** Blocks alternate backends in UI or double-tests.
   **Sketch:** `if backend.backend_type == "azure_blob":` and narrow cast, or a small `AzureLikeStorage(Protocol)` with only the probe surface.
