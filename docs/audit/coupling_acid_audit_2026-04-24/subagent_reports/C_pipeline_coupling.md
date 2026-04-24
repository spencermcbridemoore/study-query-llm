# Subagent C — Five-Stage Pipeline Coupling

Status: raw subagent finding (verbatim, edited only for fenced code blocks)
Authored by: explore subagent C, 2026-04-24
Scope: stage-to-stage coupling, `run_stage` contract, stage cohesion, hardcoded assumptions, test seams, clustering sub-pipeline.

---

## 1. Stage-to-Stage Coupling Map

### acquire -> parse
- **Downstream needs:** `dataset_group_id` (int) as the only pipeline handle; in-DB `Group` row type `dataset` with `metadata_json` including `dataset_slug` and `content_fingerprint`; acquisition artifacts (blob URIs) for bytes on disk.
- **How it gets it:** Caller passes the id. `parse` opens a DB session, loads the group, reads `dataset_slug` from metadata, then lists `dataset_acquisition_file` / `dataset_acquisition_manifest` artifacts and materializes them via `ArtifactService` (`parse.py` session query + `_collect_acquisition_artifact_uris` / `_materialize_acquisition_files`).
- **Coupling to upstream implementation:** **Moderate.** Parsing logic is not hard-wired to a single parser class, but default parser **identity and callable** are resolved through `ACQUIRE_REGISTRY` keyed by `dataset_slug` when the caller omits `parser` / ids (`_resolve_parser_and_identity` in `parse.py`). That ties parse behavior to the **source-spec registry**, not to acquire's file-layout implementation.

### parse -> snapshot
- **Downstream needs:** `dataframe_group_id` (int); optional in-memory / call-time `SubquerySpec`.
- **How it gets it:** DB: verify `Group` is `dataset_dataframe`; `find_dataframe_parquet_uri` resolves canonical parquet; parquet bytes are read from storage (`_load_dataframe_frame` in `snapshot.py`). Snapshot metadata (e.g. `dataset_slug`) is read from the dataframe group.
- **Coupling to upstream implementation:** **Low for format.** Snapshot operates on the **canonical parquet schema** and group metadata, not on parser class internals. The contract is the stable column set and `SnapshotRow`-derived table.

### parse -> embed
- **Downstream needs:** `dataframe_group_id` (int); embedding `deployment` and `provider` (and related key fields); in-storage canonical parquet for the `text` column.
- **How it gets it:** Same as snapshot for locating parquet: `find_dataframe_parquet_uri` + `ArtifactService` read (`embed._load_dataframe_texts`). Idempotent reuse uses `ArtifactService.find_embedding_matrix_artifact` with a constructed `dataset_key` (`embed.py`).
- **Coupling to upstream implementation:** **Low** for parse specifics; **shared** with snapshot on the same dataframe id and parquet artifact. There is no direct `parse` -> `embed` data object; the link is **the dataframe group and artifacts**.

### snapshot -> analyze
- **Downstream needs:** `snapshot_group_id` (int) plus (with embed) shared dataframe lineage. Inputs are loaded as: snapshot JSON artifact (`resolved_index`, spec), dataframe parquet sliced by positions, and embedding matrix rows indexed by those positions.
- **How it gets it:** DB + blob: `analyze` calls `_load_snapshot_subquery`, `_load_dataframe_slice`, and alignment checks in `analyze.py` (e.g. `snapshot_group_id` / `resolved_positions` / `embedding_matrix`).

### embed -> analyze
- **Downstream needs:** `embedding_batch_group_id` (int) for the `embedding_matrix` artifact; must match the snapshot's `source_dataframe_group_id` in group metadata.
- **How it gets it:** DB + blob: `_load_embedding_matrix` and metadata checks comparing `source_dataframe_group_id` from snapshot vs embedding groups (`analyze.py`).

**Dual-input lineage (snapshot + embedding):** `analyze` requires both ids and enforces a single `dataframe_group_id` derived from `snapshot` and `embedding` metadata before slicing (`analyze.py` ~664-675, 677-714).

## 2. `run_stage` Contract Surface

From `src/study_query_llm/pipeline/runner.py`:

### Signature

```python
def run_stage(
    *,
    db: DatabaseConnectionV2,
    stage_name: str,
    group_type: str,
    group_name: str,
    group_description: str | None = None,
    group_metadata: dict[str, Any] | None = None,
    request_group_id: int | None = None,
    source_group_id: int | None = None,
    run_key: str | None = None,
    run_kind: str = "execution",
    run_metadata: dict[str, Any] | None = None,
    depends_on_group_ids: Sequence[int] | None = None,
    contains_parent_group_ids: Sequence[int] | None = None,
    artifact_dir: str = "artifacts",
    write_artifacts: StageArtifactWriter | None = None,
    finalize_db: StageFinalizeHook | None = None,
) -> StageResult:
```

(`pipeline/runner.py` 65-83; implementation continues through ~187.)

### How a stage function declares an idempotency key
**It does not.** `run_stage` always **`create_group`s a new group** in the first session (`runner.py` ~98-108). **Per-stage idempotency** is implemented **outside** `run_stage` by:
- pre-querying for an existing group/artifact and **returning `StageResult` without calling `run_stage`** (acquire, parse, snapshot, embed), or
- for **analyze**, pre-checking `get_provenanced_run_by_request_and_key` and returning early when a completed run exists (`analyze.py` ~760-786).

**Per docs:** idempotency keys (fingerprint, `(source_dataset_group_id, parser_id, ..., dataframe_hash)`, etc.) are enforced via **metadata equality lookups** in those pre-paths, not as parameters to `run_stage`.

**Provenance runs:** if `request_group_id` and `run_key` are set, `run_stage` creates a `provenanced_run` in `RUN_STATUS_RUNNING` before artifacts (`runner.py` ~123-135). `analyze` uses this for the analysis stage (`analyze.py` `run_kind="analysis_execution"`, `run_key=run_key`).

### What is enforced at runtime vs by lint
- **Runtime (`run_stage`):** ordered steps: create group and links -> optional artifact writes -> optional `finalize_db` -> mark provenanced run **completed** (or **failed** with `stage_failure` in metadata) (`runner.py` ~98-179).
- **Lint (`scripts/check_persistence_contract.py`):**
  - Top-level public functions in `pipeline/*.py` (except `__init__.py`, `types.py`, `runner.py`) must **contain a `run_stage` call** or be decorated with `@allow_no_run_stage` (`lint_file` ~88-108).
  - **`create_group`** with a stage `group_type` is restricted to allowlisted path prefixes; otherwise reported as unauthorized (`lint_group_type_boundaries` ~141-165).

**Idempotency and artifact contracts** from `docs/DATA_PIPELINE.md` are **not** fully enforced at runtime by `run_stage` alone; they rely on each stage's logic + tests.

### Where `run_stage` does not abstract well / duplication
- **Idempotent reuse** bypasses `run_stage` entirely (separate code paths, duplicated `StageResult` shaping).
- **Five copies** of near-identical `_resolve_db` in `acquire`, `parse`, `snapshot`, `embed`, `analyze` (e.g. `acquire.py` ~28-40).
- **Repeated** `_call_artifact_uri_by_id` helpers across stage modules.
- **analyze** couples heavy logic inside `write_artifacts` (method runner, clustering envelope, then artifact writes) and `finalize_db` (MethodService, ProvenancedRunService), so `run_stage`'s "write then finalize" boundary is used but the **orchestration is very large** in one file.
- **embed** reuse path: `_collect_embedding_artifact_uris` scans `CallArtifact` broadly (`embed.py` ~54-63), which is a **leaky pattern** relative to a narrow repository query.

## 3. Stage-Internal Cohesion

| Stage | Orchestration vs algorithm | Reaches algorithms / providers / clustering | File size (approx.) | Obvious split points |
|--------|----------------------------|---------------------------------------------|------------------------|----------------------|
| **acquire** | Orchestration: fetch, manifest, fingerprint, store. | `datasets.acquisition` + `DatasetAcquireConfig`; no ML. | ~197 lines (`acquire.py`) | Extraction of `_resolve_db` / shared artifact URI helpers. |
| **parse** | Mixed: IO + **canonical PyArrow table build** (`_build_dataframe_payload`) is fixed schema logic. | Registry dispatch for parser; no embedding. | ~376 lines (`parse.py`) | Move table/manifest building to a small `canonical_dataframe` module. |
| **snapshot** | Mixed: **pandas `query`**, category filter, sampling — real algorithmic surface. | No clustering; no providers. | ~351 lines (`snapshot.py`) | Extract `_apply_subquery` / filter helpers for unit tests without DB. |
| **embed** | Orchestration around **embedding fetch**; matrix validation. | `fetch_embeddings_async` default path; overridable `embedding_fetcher`. | ~268 lines (`embed.py`) | Narrow artifact listing for reuse; optional split of "matrix find vs write." |
| **analyze** | Orchestration + **large** inline policy: representation derivation, v1 clustering hooks, HDBSCAN dispatch, default runner, DB services. | **Clustering** package + `hdbscan_runner` + `algorithms.recipes` + Method/Provenance services. | **~1072+ lines** (`analyze.py`) | Split: representation slicing, method registry, `finalize_db` + provenance, artifact typing. |

## 4. Hardcoded Assumptions

- **Source-spec / dataset slugs in pipeline:** **No** `banking77` / `twenty_newsgroups` string literals under `src/study_query_llm/pipeline/` (grep). Slugs in tests and `datasets/source_specs` are expected; the pipeline reads **`dataset_slug` from group metadata** or the acquire config.
- **Provider / deployment:** `embed(..., provider: str = "azure")` — default **provider name** is hardcoded (`embed.py` ~118).
- **Representation names:** `full`, `label_centroid`, `intent_mean` (aliases) appear as string constants and checks in `embed.py` (`LEGACY_NON_FULL_ALIASES`, `_normalize_representation`) and `analyze.py` (`REPRESENTATION_*`, `_resolve_representation`, `_derive_representation_view`).
- **Clustering v1 config path:** `analyze.py` **hardcodes** `_CLUSTERING_RULES_PATH` to `config/rules/clustering/rules-v1.0.0.yaml` (~56-61).
- **String dispatch instead of a registry object:**
  - `analyze._resolve_builtin_method_runner` matches **method name strings** to callables (`analyze.py` ~455-463).
  - `clustering/schema.py` `V1_CLUSTERING_METHODS` and `is_v1_clustering_method` (frozenset of method name strings) (~11-21, 36-38).
  - `runner_common.preprocess_for_effective_pipeline` branches on `stage` strings in `pipeline_effective` (~42-57).

## 5. Test Seams

- **acquire / parse / snapshot / embed / analyze (existing tests):** `tests/pipeline/test_*.py` predominantly use **SQLite** `tmp_path` databases and **local** artifact storage (`monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")`). They are **integration-style**, not DB-free.
- **Cheapest DB-free unit tests (if APIs were factored or tested indirectly):**
  - **snapshot:** Pure functions `_apply_subquery`, `_apply_category_filter`, `_hash_payload` in `snapshot.py` (in-memory `pd.DataFrame` + spec dicts) — *no public test module dedicated to this today*.
  - **parse:** `_build_dataframe_payload(rows)` from `SnapshotRow` lists — *same*.
  - **embed:** `embedding_fetcher` is injectable; **DB is still required** for group + artifact lookup.
  - **analyze:** `method_runner` override allows a pure numpy/json runner; **tests still build full chain** in `_prepare_inputs` (`test_analyze.py`) — *DB required*.
- **Repository fakes:** `test_runner.py` uses real `DatabaseConnectionV2` + `RawCallRepository` with explicit `create_group` — **no** in-memory repository fake in pipeline tests.
- **Evidence of mocking:** `test_embed.py` fakes `embedding_fetcher`; `test_analyze.py` supplies custom `method_runner` and (elsewhere) may patch modules — the **persistence and artifact layers remain real** for stage tests.

**Conclusion:** For most stages, **"cheapest test without DB"** is *not* available without extracting pure helpers or introducing repository protocols; the current test suite's pattern **confirms strong DB/artifact coupling**.

## 6. Clustering Sub-pipeline (`pipeline/clustering/`)

- **Pattern:** **Strategy-style free functions** per method, not a class hierarchy. `kmeans_runner.run_kmeans_silhouette_kneedle_analysis` and `gmm_runner.run_gmm_bic_argmin_analysis` use `@allow_no_run_stage` and share `runner_common` / `selection` / `hashing` / `resolver` / `validators` (`pipeline/clustering/__init__.py` exports).
- **Where new methods plug in:** (1) Add method name to `V1_CLUSTERING_METHODS` and `base_algorithm_for_method` in `schema.py` if part of v1; (2) extend YAML rule resolution in `resolver.py` / `rules-v1.0.0.yaml`; (3) add a runner module and wire it in `analyze._resolve_builtin_method_runner` and `clustering.__init__` as needed. **HDBSCAN is wired from `pipeline/hdbscan_runner.py`, not from `clustering/`.**
- **hdbscan vs kmeans placement:** `src/study_query_llm/pipeline/hdbscan_runner.py` is **sibling to `clustering/`**, while kmeans/gmm live **under** `clustering/`. It still follows the same **callable signature** as other runners and uses `@allow_no_run_stage` (`hdbscan_runner.py` ~93-104). **Likely reasons for split:** optional **`hdbscan` import** and larger parameter surface kept as a top-level module; kmeans/gmm grouped with the **v1 pipeline provenance** stack (`runner_common`, resolver-injected `_v1_pipeline_resolved` in `kmeans_runner.py` ~21-31). The layout is **consistent in interface** but **inconsistent in packaging** (one family in subpackage, HDBSCAN outside).

## 7. Top 5 Pipeline Decoupling Opportunities

1. **What:** **Duplicated `_resolve_db` and artifact URI helpers** across all five stage modules.
   **Where:** e.g. `pipeline/acquire.py` ~28-40, `parse.py` ~31-43, `snapshot.py` ~25-37, `embed.py` ~30-42, `analyze.py` ~78-90 (and parallel `_call_artifact_uri_by_id` blocks).
   **Cheap-now reason:** One place to fix connection policy, test doubles, and Windows/env behavior.
   **Sketch:** `pipeline/db.py` with `get_connection(...)`; `pipeline/artifacts.py` for `uri_for_artifact_id`.

2. **What:** **Builtin `method_name` string dispatch** for analysis.
   **Where:** `pipeline/analyze.py` ~455-463.
   **Cheap-now reason:** New clustering methods won't need to edit a central if-chain; names stay consistent with `schema.V1_CLUSTERING_METHODS`.
   **Sketch:** `dict[str, AnalysisRunner]` registry (or `Mapping`) built at import from `clustering` + `hdbscan_runner`.

3. **What:** **HDBSCAN runner lives outside `clustering/`** while kmeans/gmm are inside — awkward discovery and documentation of "one place" for methods.
   **Where:** `pipeline/hdbscan_runner.py` vs `pipeline/clustering/kmeans_runner.py`.
   **Cheap-now reason:** Unifies extension point and imports for operators and tests.
   **Sketch:** `clustering/hdbscan_runner.py` (or `clustering/methods/hdbscan.py`) + thin re-export in `hdbscan_runner.py` for compatibility.

4. **What:** **`embed` reuse lists artifacts with a global `CallArtifact` query** (`session.query(CallArtifact).order_by...`) with manual `group_id` filter.
   **Where:** `pipeline/embed.py` ~54-63.
   **Cheap-now reason:** Correctness, performance, and a single query API aligned with `find_embedding_matrix_artifact`.
   **Sketch:** `RawCallRepository.list_group_artifacts(group_id=..., artifact_types=[...])` (same as parse/snapshot list helpers).

5. **What:** **`analyze.py` is a monolith** (representation math, v1 provenance, method execution, MethodService, ProvenancedRunService, locking).
   **Where:** `pipeline/analyze.py` (module spans ~1k+ lines; e.g. `analyze` function ~631-1072, helpers above).
   **Cheap-now reason:** Smaller files enable DB-free unit tests of pure functions and a stable **seam** for `method_runner` without loading all persistence.
   **Sketch:** `analyze_prepare.py` (load + slice + representation), `analyze_persist.py` (finalize), `analysis_methods.py` (registry); keep public `analyze()` as thin composition.
