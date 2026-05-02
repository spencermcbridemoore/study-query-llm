# Method Recipes

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-05-02

## Purpose

A **method recipe** describes a composite/pipeline analysis method as an
ordered list of component stages. Each stage references an existing
`method_definitions` row by `(name, version)` and carries stage-local
default parameters. The recipe is **descriptive metadata** — execution
still lives in the relevant algorithm module (e.g. `run_sweep` in
`src/study_query_llm/algorithms/sweep.py`). The recipe exists so that:

1. Provenance reads can answer *"what pipeline was this run?"* in one
   place, rather than inferring it from worker/code paths.
2. Two runs can be compared at the pipeline level: if their `recipe_hash`
   differs, the pipelines are structurally different.
3. Future work on intermediate-artifact caching and inner-run DAG execution
   has a stable, versioned pipeline shape to reference.

This document is the v0 spec.

## Scope and non-goals

- **Descriptive, not executable.** This phase does not introduce a runtime
  recipe executor. `run_sweep` continues to run the pipeline monolithically.
- **Clustering family only (for now).** Registered clustering composites are
  `cosine_kllmeans_no_pca`, `hdbscan+fixed`,
  `kmeans+normalize+pca+sweep`, and `gmm+normalize+pca+sweep`. Adding
  more composites is additive.
- **No fingerprint tuple shape change.** The recipe hash enters the
  canonical run fingerprint via `config_json["recipe_hash"]`, which
  `canonical_run_fingerprint` already hashes (minus scheduling-only keys).
  The fingerprint JSON schema itself is unchanged.

## Bundled Clustering Subsystem

### Subsystem definition

- **Name:** Bundled Clustering Subsystem.
- **Module home:** `src/study_query_llm/pipeline/clustering/`.
- **Registry:** `src/study_query_llm/pipeline/clustering/registry.py`
  registers only bundled clustering methods.
- **Output schema contract:** every bundled method emits
  `cluster_labels`, `summary_metrics`, and `recipe_hash`. Methods producing
  non-cluster-label artifacts (for example transformed embeddings) do not
  belong in this subsystem.

### Embedding representation (Slice 1.6)

- Bundled registry specs declare `allowed_representations=frozenset({"full"})` only.
- The analyze stage rejects `representation_type` / `embedding_representation` values `label_centroid` and `intent_mean` with a migration `ValueError` (per-label aggregate vectors are not used as clustering inputs).

### Naming grammar (locked)

- **Pattern:** `<algorithm>+<preprocessing-chain-tokens>+<fit-mode>`.
- **Algorithm tokens (closed set):** `kmeans`, `spherical-kmeans`, `gmm`,
  `hdbscan`, `agglomerative`, `dbscan`, `spectral`, `leiden`, `louvain`.
- **Preprocessing tokens (closed set):** `normalize`, `pca`, `umap`,
  `umap-graph`, `knn-graph`, `similarity`. Tokens are listed
  left-to-right in the method name in the order they are applied to
  embeddings. Adding a new preprocessing token requires an explicit grammar
  update in this document.
- **Fit-mode tokens:** `fixed-k`, `fixed-eps`, `fixed`, `sweep`.
  `sweep` is reserved for existing sweep-select methods.
- **Special annotation:** `+approx+` annotates the approximation variant
  of `spherical-kmeans` (`normalize -> sklearn KMeans`); the true
  spherical objective is deferred and out of scope.
- **Hyperparameters are parameters, not name tokens.** `k`, `eps`,
  `min_samples`, `linkage`, `init`, `random_state`, `n_components`,
  `n_neighbors`, `resolution`, `affinity`, and `metric` are passed as
  method parameters.

### Examples

- `kmeans+fixed-k`
- `kmeans+normalize+pca+fixed-k`
- `spherical-kmeans+approx+fixed-k`
- `hdbscan+normalize+pca+fixed`
- `leiden+knn-graph+fixed`

### Bundled-vs-composed coexistence rule

- Bundled methods are permanently self-contained.
- Bundled methods MUST NOT consume output from a future standalone
  DR-as-method (transform) run.
- Bundled methods are not transitional; they continue to exist after
  DR-as-method ships.

### Forward reservation for transforms

- `src/study_query_llm/pipeline/transforms/` is reserved for future
  DR-as-method work (transformed-embedding artifacts as first-class outputs).
- No files exist there in this rollout.
- Adding code to `src/study_query_llm/pipeline/transforms/` requires a
  separate explicitly scoped design and is out of scope for the current
  bundled clustering rollout.

### Legacy names retired in Slice 1.5

The pre-grammar names `hdbscan`, `kmeans+silhouette+kneedle`, and
`gmm+bic+argmin` were re-registered under the bundled grammar in Slice 1.5
with algorithmic identity preserved (the runner functions did not change):

| Legacy name (deprecated)       | Bundled-grammar name (current)  |
|--------------------------------|---------------------------------|
| `hdbscan`                      | `hdbscan+fixed`                 |
| `kmeans+silhouette+kneedle`    | `kmeans+normalize+pca+sweep`    |
| `gmm+bic+argmin`               | `gmm+normalize+pca+sweep`       |

A loud-fail deprecation guard
(`raise_if_deprecated_clustering_method` in
`pipeline/clustering/registry.py`) rejects the legacy names at the top of
`pipeline.analyze()` so neither explicit `method_runner` injection nor
notebook scripts can land new rows under deprecated names; the guard fires
*before* runner resolution. Historical `provenanced_runs` rows under the
legacy names remain queryable; only new write paths are blocked.

BANK77 strategy CLI tokens (`hdbscan`, `kmeans_silhouette_kneedle`,
`gmm_bic_argmin`) are kept stable for operator continuity by attaching
them as `strategy_aliases` on the new bundled-grammar specs in the
registry; the alias index resolves them to the bundled-grammar method
names. `agglomerative+fixed-k` already conformed to the grammar before
Slice 1.5 and is unchanged.

The bundled `kmeans+normalize+pca+sweep` and `gmm+normalize+pca+sweep`
methods always include `normalize -> pca -> <algorithm>` in the effective
pipeline; the legacy `<= 200` embedding-dim PCA-skip branch from the
retired YAML resolver was dropped as part of the rename. `pca_n_components`
is a method parameter (default 100, clamped at runtime to
`max(1, min(value, embedding_dim, max(1, n_samples - 1)))`).

## Storage

Recipes live on `method_definitions.recipe_json` (JSON, nullable), added by
`src/study_query_llm/db/migrations/add_recipe_json_column.py`. Composite
methods SHOULD populate this column; non-composite method rows SHOULD leave
it null.

## Recipe JSON shape (v0)

```json
{
  "recipe_version": "v0",
  "stages": [
    {
      "name": "<component MethodDefinition.name>",
      "version": "<component MethodDefinition.version>",
      "role": "<pooling | projection | initialization | clustering | ...>",
      "params": { "<stage-local defaults>": "..." }
    }
  ],
  "notes": "<optional free text>"
}
```

Rules:

- `stages` is ordered; stage order is semantically significant.
- Each stage's `(name, version)` MUST resolve to an existing
  `method_definitions` row. Use the helper
  `register_clustering_components` (or its family equivalent) to keep
  component rows in sync with code.
- `params` at the recipe level are *defaults* that describe the canonical
  pipeline. Actual per-run values live in run `config_json` and MAY differ
  (within the composite method's `parameters_schema`).
- Bump the composite's `version` when the recipe changes shape or swaps
  components; attaching a recipe to an existing row that lacked one does
  not warrant a version bump and should use
  `MethodService.update_recipe` / `ensure_composite_recipe`.

## Canonical recipe hash

`canonical_recipe_hash(recipe)` = SHA-256 of
`json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
The hash is deterministic (dict authoring order does not matter) and
sensitive to stage order, stage `version`, and stage `params` changes.

Canonical definitions live in
`src/study_query_llm/algorithms/recipes.py`:

- `RECIPE_VERSION = "v0"`
- `CLUSTERING_COMPONENT_METHODS` — component specs for
  `mean_pool_tokens`, `pca_svd_project`, `kmeanspp_init`, `k_llmmeans`,
  `umap_project`.
- Canonical clustering composite recipes:
  - `COSINE_KLLMEANS_NO_PCA_RECIPE` (3 ordered stages:
    `mean_pool_tokens` -> `kmeanspp_init` -> `k_llmmeans`)
  - `HDBSCAN_FIXED_RECIPE` (registered under `hdbscan+fixed`).
  - `KMEANS_NORMALIZE_PCA_SWEEP_RECIPE` (registered under
    `kmeans+normalize+pca+sweep`).
  - `GMM_NORMALIZE_PCA_SWEEP_RECIPE` (registered under
    `gmm+normalize+pca+sweep`).
  - Legacy constants `HDBSCAN_V1_RECIPE`,
    `KMEANS_SILHOUETTE_KNEEDLE_RECIPE`, `GMM_BIC_ARGMIN_RECIPE` are
    intentionally preserved at module scope as inputs to the permanent
    CONSTANT-vs-CONSTANT recipe-hash regression
    (`tests/test_services/test_recipe.py::test_bundled_recipe_constants_match_legacy_constants`)
    but are no longer registered in `COMPOSITE_RECIPES`; resolution under
    the legacy names raises `KeyError`.
- `COMPOSITE_RECIPES` — name → recipe registry. After Slice 1.5 it holds
  `cosine_kllmeans_no_pca`, `hdbscan+fixed`, `kmeans+normalize+pca+sweep`,
  and `gmm+normalize+pca+sweep`.
- `build_composite_recipe(name)` — returns a deep copy.
- `register_clustering_components(method_service)` — idempotent component
  registration.
- `ensure_composite_recipe(method_service, name, ...)` — idempotent
  composite registration with recipe attached (or back-filled in place when
  the row already exists without one).

## Fingerprint integration

Ingestion (`src/study_query_llm/experiments/ingestion.py`) injects
`recipe_hash = canonical_recipe_hash(recipe)` into the `config_json`
passed to `ProvenancedRunService.record_method_execution`. The fingerprint
code path is unchanged:

1. `record_method_execution` passes `config_json` to
   `canonical_run_fingerprint`.
2. `_strip_scheduling_keys` removes only orchestration/lease keys;
   `recipe_hash` survives.
3. `canonical_config_hash` hashes the remaining config.
4. The resulting `fingerprint_hash` therefore differs whenever `recipe_hash`
   differs, giving runs from different pipelines distinct semantic
   identities.

Two runs with identical scheduling history but different recipes will have
different fingerprints. Two runs with different scheduling but the same
recipe, method, config, and input snapshot will have identical
fingerprints.

## Operator workflow

Initial setup (one-time; idempotent on re-run):

```
python -m study_query_llm.db.migrations.add_recipe_json_column
python scripts/register_clustering_methods.py
```

From then on, `ingest_result_to_db` for composite methods will:

- Ensure the 5 registered component methods exist (idempotent no-ops after first run).
- Ensure the composite row exists with `recipe_json` populated (attaching
  in place when a prior row lacked the recipe).
- Inject `recipe_hash` into the run's `config_json` before recording the
  `provenanced_runs` row.

Stage-5 `analyze` persists the canonical composite `recipe_hash` in
analysis execution `config_json` for every composite method registered in
`COMPOSITE_RECIPES`, including the bundled-grammar clustering methods
(`hdbscan+fixed`, `kmeans+normalize+pca+sweep`,
`gmm+normalize+pca+sweep`). All bundled clustering specs ship with
`provenance_envelope="none"`; the legacy v1 envelope (and its
resolver/validators/identity decoration in `clustering_summary`) was
retired in Slice 1.5, so no v1 envelope fields land in `config_json` for
new write paths.

## Adding a new composite

1. Register components in `CLUSTERING_COMPONENT_METHODS` (or a new family
   list) if they are not already there.
2. Define the canonical recipe dict and add it to `COMPOSITE_RECIPES`.
3. Route it through `ensure_composite_recipe` at the ingestion site.
4. Add tests mirroring the pattern in `tests/test_services/test_recipe.py`.

## Cross-references

- [ARCHITECTURE_CURRENT.md](ARCHITECTURE_CURRENT.md) — recipe layer sits
  inside the `method_definitions` contract.
- [SCHEDULING_PROVENANCE_BOUNDARY.md](SCHEDULING_PROVENANCE_BOUNDARY.md) —
  the recipe does not drive scheduling; stages are not orchestration jobs
  in this phase.
- Bundled Clustering Subsystem — see
  [Bundled Clustering Subsystem](#bundled-clustering-subsystem); module home
  `src/study_query_llm/pipeline/clustering/`.
- `docs/STANDING_ORDERS.md` — Method Definitions and Provenance section
  cites this recipe rule.
