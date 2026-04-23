# Method Recipes

Status: living  
Owner: documentation-maintainers  
Last reviewed: 2026-04-14

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
- **Clustering family only (for now).** The only composite recipe currently
  registered is `cosine_kllmeans_no_pca`. Adding more composites is
  additive.
- **No fingerprint tuple shape change.** The recipe hash enters the
  canonical run fingerprint via `config_json["recipe_hash"]`, which
  `canonical_run_fingerprint` already hashes (minus scheduling-only keys).
  The fingerprint JSON schema itself is unchanged.

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
- `COSINE_KLLMEANS_NO_PCA_RECIPE` — the canonical recipe for that
  composite (3 ordered stages: `mean_pool_tokens` -> `kmeanspp_init` ->
  `k_llmmeans`).
- `COMPOSITE_RECIPES` — name → recipe registry.
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
- `docs/STANDING_ORDERS.md` — Method Definitions and Provenance section
  cites this recipe rule.
