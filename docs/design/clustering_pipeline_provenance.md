# Clustering Pipeline Provenance (Design)

Status: design draft  
Owner: pipeline maintainers  
Last reviewed: 2026-04-22

> **Design document** — captures decisions about how clustering-pipeline
> provenance should be structured before the second clustering runner is
> built. Nothing here is implemented yet; the existing
> `src/study_query_llm/pipeline/hdbscan_runner.py` is the only production
> clustering runner today.

## Purpose

Define the shape that clustering-pipeline provenance should take once
multiple clustering methods (k-sweep based, HDBSCAN, Leiden/Louvain, etc.)
coexist in the analyze stage. Specifically:

1. How to differentiate clustering operations from other future
   operation kinds at the artifact level.
2. How to decide method parameters consistently across pipelines, with
   provenance that lets a reader recover *why* each parameter has its
   value.
3. How to compose preprocessing + terminal-clusterer stages without
   per-pipeline parameter tuning.
4. How to record sweep-and-select methods (e.g.
   `kmeans + silhouette + kneedle`) so their selection evidence is
   first-class, not hidden behind a "chosen k" scalar.

## Glossary

- **Operation type**: provenance contract that defines artifact shape,
  valid stage vocabulary, and validation boundaries for a class of
  computations (initially: `cluster_pipeline`).
- **Rule set**: versioned parameter-decision policy within one
  operation type.
- **Declared pipeline**: stage sequence requested by the study design.
- **Resolved pipeline**: declared pipeline with concrete parameters
  derived from rule tiers and context.
- **Effective pipeline**: resolved pipeline after skip rules are
  applied.
- **Selection evidence**: sweep range, metric curve, and rule output
  used to choose a Tier-4 parameter value.
- **Single-fit method**: method with no sweep over cluster count-like
  parameters (for example HDBSCAN in fixed-parameter mode).
- **Sweep-and-select method**: method that evaluates a parameter range
  and chooses one value via a scoring rule.
- **Hard constraint**: invariant that must hold; violations raise.
- **Soft heuristic**: preferred rule outcome that may be overridden by
  explicit pipeline parameters.
- **Alias**: two declared pipelines that resolve to the same effective
  pipeline hash for a given context and rule set.
- **Rule inputs**: context keys that rules are allowed to consult
  (initially `embedding_dim`, `n_samples`).
- **Input audit metadata**: optional provenance metadata recorded for
  traceability but not consulted by rules (for example
  `embedding_model_id` before model-specific rules exist).

## Status and scope

- **Decided.** The conceptual model in this document (operation type,
  rule set, parameter tiers, declared/resolved/effective pipeline,
  selection-evidence schema, stage-skip semantics) is the agreed shape.
- **Deferred.** No code changes are part of this design pass. The
  rule-resolver, schema migrations, and any new runners come later.
- **Trigger to begin implementation.** When a second clustering runner
  is added (e.g. a k-sweep runner for k-means + silhouette + kneedle),
  the rule-resolver and dual `pipeline_declared`/`pipeline_effective`
  fields become worth building. Until then, this is a forward
  reference.

## V1 Scope and Non-goals

### In scope (v1)

- **Methods / composites:** `hdbscan`, `kmeans+silhouette+kneedle`,
  `gmm+bic+argmin`.
- **Capabilities:** tiered rule resolution, declared/resolved/effective
  pipeline recording, `pipeline_effective_hash`, and alias-detection
  primitives.
- **Artifact model:** clustering artifacts carry operation identity,
  rule-set provenance, resolved/effective pipeline identity, and
  selection evidence (for sweep methods).

### Out of scope (v1)

- Generic multi-operation-type framework beyond `cluster_pipeline`.
- Advanced skip-expression language (OR/NOT/arbitrary expression trees).
- Cross-rule-set diff tooling or UI.
- Full `input_audit_metadata` taxonomy.
- Broad catalog expansion beyond v1 methods/composites.

## Relationship to existing pipeline docs

This design layers on top of, and does not replace, the following:

- [`docs/DATA_PIPELINE.md`](../DATA_PIPELINE.md) — defines the
  canonical five-stage pipeline (`acquire -> parse -> snapshot -> embed
  -> analyze`). Everything in this document lives inside the `analyze`
  stage.
- [`docs/living/METHOD_RECIPES.md`](../living/METHOD_RECIPES.md) —
  defines the existing `recipe_json` / `recipe_hash` mechanism for
  composite method identity. The **resolved pipeline** in this design
  is materially equivalent to a recipe; the **rule set** is the layer
  that *produces* the resolved recipe from a declared pipeline plus
  context. Recipes describe pipelines; rule sets choose them.
- [`docs/living/SCHEDULING_PROVENANCE_BOUNDARY.md`](../living/SCHEDULING_PROVENANCE_BOUNDARY.md)
  — this design lives entirely on the provenance side of that
  boundary. Sweep ranges, restart counts, and selection rules are
  provenance content, not orchestration content.
- `src/study_query_llm/pipeline/hdbscan_runner.py` — the existing
  single-fit clustering runner; treated below as the degenerate case
  of the general selection-pipeline framework.

## Concepts

### Operation type vs rule set

Two orthogonal axes of provenance, recorded as separate fields on every
artifact:

- **Operation type** — the *contract*: artifact schema, valid stage
  vocabulary, validation rules, evaluation framework. Changes rarely.
  Initial value: `cluster_pipeline`. Future operation types
  (`embedding_generation`, `dataset_ingestion`, `evaluation`, etc.)
  are peer entries.
- **Rule set** — a versioned configuration that decides parameters
  *within* an operation type. Changes frequently. Multiple rule sets
  can coexist for the same operation type (e.g. "small-data defaults"
  vs "large-data defaults").

The operation type bounds what rules can exist (a `pca.n_components`
rule has no meaning in a `dataset_ingestion` operation type). The rule
set chooses values for the blanks the operation type's schema leaves
open.

### Parameter tiers

Every parameter in a clustering pipeline falls into exactly one of
four tiers, distinguished by *what determines its value*:

1. **Global hygiene** — identical across all pipelines and datasets
   (e.g. `random_state=42`, `kmeans.n_init=20`,
   `hdbscan.cluster_selection_method="eom"`). Lives in one place.
2. **Adjacency-derived** — depends on neighbor stages in the pipeline
   (e.g. `pca.n_components` depends on what the downstream consumer
   is; `agglomerative.linkage` depends on the metric the upstream
   normalizer enables). Determined by adjacency rules.
3. **Dataset constants** — one value per dataset, identical across all
   pipelines for that dataset (e.g. `umap.n_neighbors`,
   `knn_graph.k`, `hdbscan.min_cluster_size` defaults).
4. **Swept** — the only true free parameter per pipeline, resolved by
   a selection method (`k`, `min_cluster_size` when treated as a
   sweep, `eps`, `resolution`).

Tiers 1–3 are deterministic given the rule set + dataset + pipeline
shape. Tier 4 requires a selection mechanism and produces sweep
evidence.

### Stage vocabulary and tier precedence (v1)

#### Stage vocabulary (v1)

Preprocessing stages:

- `embed`
- `normalize`
- `pca`
- `umap`

Terminal clustering stages:

- `hdbscan`
- `kmeans`
- `gmm`

Composite method identifiers (for example
`kmeans+silhouette+kneedle`) represent a terminal stage plus a
selection policy binding. In v1, declared pipelines MAY reference these
composite identifiers directly.

#### Tier precedence (v1)

When multiple sources can set a parameter, resolve in this order:

1. **Hard constraints** (always enforced; cannot be overridden).
2. **Explicit declared override** (if provided on declared pipeline for
   non-swept parameters).
3. **Tier 4 selection result** (for swept parameters).
4. **Tier 3 dataset constants**.
5. **Tier 2 adjacency-derived defaults**.
6. **Tier 1 global hygiene defaults**.

Notes:

- Tier-4 selection applies only to parameters classified as swept for
  the method.
- Explicit declared overrides do not bypass hard constraints.

### Pipeline lifecycle: declared, resolved, effective

A pipeline exists in three layers across its lifecycle. All three MUST
be recorded on every clustering artifact:

| Layer | What it is | When it's known |
|---|---|---|
| **Declared** | Static stage sequence: `[embed, normalize, pca, hdbscan]` | Study design time |
| **Resolved** | Same shape with concrete tier-1/2/3 parameters bound; tier-4 sweep evidence attached | After rules fire |
| **Effective** | Resolved pipeline with skipped stages removed | After conditional skips apply |

These coincide when no skip rule triggers; they diverge when one
does. The `pipeline_effective_hash` (a content hash over the
effective stage list with bound parameters) is the canonical identity
for "what computation actually ran" and is the basis for alias
detection across pipelines.

### Selection pipeline vs single-fit (HDBSCAN as degenerate case)

Clustering methods split into two execution shapes:

- **Single-fit** — one fit, no `k` to choose. HDBSCAN, DBSCAN, Leiden,
  Louvain. The cluster count emerges from density / resolution
  parameters. Selection evidence is empty.
- **Sweep-and-select** — fit at multiple candidate `k` (or
  `resolution` etc.), score each with a metric curve, apply a
  selection rule to choose one. K-means, GMM, agglomerative,
  spectral, plus the various "+ silhouette + kneedle" or
  "+ stability + threshold" composites.

Single-fit is the degenerate case of sweep-and-select where
`sweep_range` has length 1, `selection_metric` is null, and
`metric_curve` is empty. The artifact schema (below) is designed for
the sweep-and-select case; single-fit runners just leave the
selection-evidence block empty or absent.

### Method naming convention

The selection step is part of a method's identity, not a hidden
post-processing decoration. Two artifacts with the same base
algorithm but different selection rules are *different methods* —
they will produce different labelings on identical inputs.

Naming pattern: `<base>+<metric>+<rule>`. Examples:

- `hdbscan` — single-fit, no selection.
- `kmeans+silhouette+kneedle` — k-means swept across a range,
  silhouette computed per k, kneedle picks the inflection.
- `kmeans+stability_ari+threshold` — k-means swept, bootstrap-ARI
  per k, largest k above threshold wins.
- `gmm+bic+argmin` — GMM swept, BIC per k, k with minimum BIC wins.

The base algorithm and selection rule together drive the
`method_name` field; both must be considered when comparing across
artifacts.

### Selection metric and rule catalog

The catalog below defines the allowed building blocks for
`<base>+<metric>+<rule>` method naming and selection-evidence payloads.

#### Metric families

| Metric key | Family | Typical terminal methods | Optimize |
|---|---|---|---|
| `silhouette` | Internal validation | k-means, spherical k-means, agglomerative, spectral | `argmax` / `kneedle` on curve |
| `calinski_harabasz` | Internal validation | k-means, agglomerative, spectral | `argmax` |
| `davies_bouldin` | Internal validation | k-means, agglomerative | `argmin` |
| `gap` | Reference-based internal validation | k-means | first `k` meeting gap rule |
| `bic` | Information criterion | GMM | `argmin` |
| `aic` | Information criterion | GMM | `argmin` |
| `icl` | Information criterion + assignment entropy | GMM | `argmin` |
| `stability_ari` | Resampling stability | k-means, spherical k-means, agglomerative | threshold or `argmax` |
| `prediction_strength` | Resampling stability | k-means, agglomerative | threshold |
| `eigengap` | Structural spectral heuristic | spectral | `argmax_gap` |
| `dendrogram_jump` | Structural hierarchy heuristic | agglomerative | `argmax_jump` |
| `dbcv` | Density-cluster validation | HDBSCAN / DBSCAN sweeps | `argmax` |

#### Selection-rule families

| Rule key | Description | Typical use |
|---|---|---|
| `argmax` | Choose parameter with highest score | silhouette, CH, DBCV |
| `argmin` | Choose parameter with lowest score | BIC, AIC, DB |
| `kneedle` | Detect elbow/knee of monotone-ish curve | inertia, silhouette plateau |
| `threshold_max_k` | Largest parameter whose score passes threshold | stability ARI, prediction strength |
| `argmax_gap` | Largest adjacent eigenvalue gap index | spectral eigengap |
| `argmax_jump` | Largest merge-cost jump index | agglomerative dendrogram |
| `first_satisfying` | First parameter satisfying rule predicate | gap rule variants |

#### Canonical composite examples

- `kmeans+silhouette+kneedle`
- `kmeans+stability_ari+threshold_max_k`
- `gmm+bic+argmin`
- `spectral+eigengap+argmax_gap`
- `agglomerative+dendrogram_jump+argmax_jump`

#### Composite status (v1 gate)

| Composite / method | Status |
|---|---|
| `hdbscan` | v1 |
| `kmeans+silhouette+kneedle` | v1 |
| `gmm+bic+argmin` | v1 |
| other catalog composites | deferred |

## Schemas

### Artifact schema

Every clustering artifact carries the following structure. Fields
marked OPTIONAL are absent for single-fit methods.

```jsonc
{
  // --- Identity ---
  "operation_type":      "cluster_pipeline",
  "operation_version":   "v1",
  "method_name":         "kmeans+silhouette+kneedle",   // see naming convention
  "base_algorithm":      "kmeans",                       // for grouping / comparison

  // --- Rule set provenance ---
  "rule_set_version":    "rules-v1.2.3",
  "rule_set_hash":       "sha256:abc...",
  "rule_inputs": {                                       // context the rules consulted
    "embedding_dim": 768,
    "n_samples": 18234
  },
  "input_audit_metadata": {                              // optional metadata not consulted by rules
    "embedding_model_id": "all-mpnet-base-v2"
  },

  // --- Pipeline lifecycle ---
  "pipeline_declared":   ["embed", "normalize", "pca", "kmeans"],
  "pipeline_resolved":   [
    { "stage": "embed", "params": {} },
    { "stage": "normalize", "params": {} },
    { "stage": "pca", "params": { "n_components": 100, "random_state": 42 } },
    { "stage": "kmeans", "params": { "n_init": 20, "metric": "cosine", "k": 10 } }
  ],
  "pipeline_effective":  ["embed", "normalize", "pca", "kmeans"],
  "pipeline_effective_hash": "sha256:def...",            // hash over effective stage sequence + bound stage params
  "skipped_stages":      [],                             // [{ stage, reason }] when non-empty
  "aliases":             [],                             // declared pipelines that resolved identically

  // --- Selection evidence (OPTIONAL: present iff method has a sweep) ---
  "selection_evidence": {
    "sweep_range":            [2, 3, 5, 8, 10, 15, 20, 30, 50],
    "selection_metric":       "silhouette",
    "selection_rule":         "kneedle",
    "selection_rule_params":  { "S": 1.0, "curve": "concave" },
    "selection_curve_artifact_ref": "kmeans_selection_curve.json",
    "chosen_k":               10,
    "chosen_k_rationale":     "kneedle inflection at k=10; silhouette plateaus for k>=10"
  },

  // --- Cluster summary (always present, same shape regardless of method) ---
  "n_samples":                 18234,
  "n_features":                100,
  "cluster_count":             10,
  "cluster_ids":               [0, 1, ..., 9],
  "cluster_sizes":             { "0": 1523, "1": 980, "...": "..." },
  "noise_count":               0,                          // 0 for non-density methods
  "noise_fraction":            0.0,
  "labels_artifact_ref":       "kmeans_labels.json"
}
```

### Hash identity contract (v1)

`pipeline_effective_hash` is the canonical semantic identity of the
effective clustering computation.

Decision (v1):

- V1 in-scope methods (`hdbscan`, `kmeans+silhouette+kneedle`,
  `gmm+bic+argmin`) MUST materialize effective pipeline identity through
  the existing recipe mechanism, and persist `recipe_hash`.
- If a canonical effective recipe payload exists, `pipeline_effective_hash`
  MUST equal that payload's canonical recipe hash.
- If a separate `recipe_hash` field is materialized on the same run for
  the effective pipeline identity, it MUST match `pipeline_effective_hash`.
- Hash identity is computed over the effective pipeline shape (ordered
  stages) plus bound stage parameters.

Canonical serialization (v1):

1. Build payload:
   - `stages`: ordered effective stage list.
   - Each stage entry includes `stage` and normalized `params`.
2. Normalize values:
   - Object keys sorted lexicographically.
   - `null`/boolean/integer values serialized as native JSON types.
   - Floating values normalized to canonical decimal string form via
     `format(value, ".12g")` before final JSON serialization.
3. Serialize with:
   - `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
4. Hash:
   - SHA-256 over UTF-8 bytes of canonical serialized payload.

Selection-evidence persistence (v1):

- Keep compact selection summary inline in `selection_evidence`.
- Persist the full metric curve in a sibling artifact referenced by
  `selection_curve_artifact_ref`.

### Rule-set file shape

Rule sets live as versioned files under (proposed)
`config/rules/clustering/rules-v<version>.yaml`. A single file holds
all four parameter tiers. `rule_set_hash` is computed from the
canonicalized parsed YAML (sorted-key JSON serialization + UTF-8
SHA-256), not raw file bytes, so non-semantic YAML edits do not churn
semantic identity. The raw file digest MAY be recorded separately as
`rule_set_file_digest` under `input_audit_metadata` for forensic
traceability.

```yaml
version: "1.2.3"
context_keys: [embedding_dim, n_samples]   # what rules are allowed to consult

global_hygiene:                            # Tier 1
  random_state: 42
  kmeans: { n_init: 20, max_iter: 300, init: "k-means++" }
  hdbscan: { cluster_selection_method: "eom", alpha: 1.0, min_samples: null }
  gmm: { reg_covar: 1.0e-6, n_init: 10 }

adjacency_rules:                           # Tier 2
  pca:
    n_components_for_downstream:
      hdbscan:
        - { input_dim_le: 200,  value: null }       # null => skip
        - { input_dim_le: 768,  value: 30 }
        - { input_dim_le: 4096, value: 100 }
      kmeans:
        - { input_dim_le: 200,  value: null }
        - { input_dim_le: 4096, value: 100 }
    hard_cap: "min(value, input_dim, n_samples - 1)"
    skip_when: [ { input_dim_le: 200 } ]

  umap:
    n_components_for_downstream: { hdbscan: 10, kmeans: 10, gmm: 10 }
    metric_when_normalized_upstream: "euclidean"
    metric_otherwise: "cosine"
    min_dist_for_clustering: 0.0

  agglomerative:
    linkage_when_metric:
      euclidean: ["ward", "average", "complete"]
      cosine:    ["average", "complete"]            # ward forbidden

  hdbscan:
    metric_when_normalized_upstream: "euclidean"
    metric_otherwise: "cosine"

dataset_constants:                         # Tier 3
  twenty_newsgroups:
    embedding_dim: 768
    n_samples: 18846
    umap_n_neighbors: 15
    knn_graph_k: 15
    hdbscan_min_cluster_size_default: 50
    dbscan_min_samples: 5

selection_policies:                        # Tier 4
  "kmeans+silhouette+kneedle":
    base_algorithm: "kmeans"
    k_range: [2, 3, 5, 8, 10, 15, 20, 30, 50]
    selection_metric: "silhouette"
    selection_rule: "kneedle"
  "gmm+bic+argmin":
    base_algorithm: "gmm"
    k_range: [2, 3, 5, 8, 10, 15, 20, 30, 50]
    selection_metric: "bic"
    selection_rule: "argmin"

cross_stage_skip_rules:                    # cross-cutting skip rules
  - skip_stage: "normalize"
    when_downstream_contains: "umap"
    when_downstream_param: { umap.metric: "cosine" }
    rationale: "UMAP with cosine metric normalizes internally"
```

## Rules layer in detail

### Lookup tables keyed by context

Adjacency and dataset-constant rules are expressed as lookup tables
whose rows are predicates over the rule-input context (`embedding_dim`,
`n_samples`). The resolver picks the first matching row. Rules MUST
NOT consult any context key not declared in `context_keys` at the top
of the rule file (so artifacts can record exactly the inputs the
rules saw).

A thin Python resolver applies hard caps after the table lookup
(e.g. `min(value, input_dim, n_samples - 1)` for PCA's
`n_components`). Caps are encoded as expression strings in the rule
file, evaluated in a restricted namespace containing only the
declared context keys plus `value`.

### Input audit metadata contract (v1)

`input_audit_metadata` is optional provenance metadata recorded for
traceability and operator diagnostics.

Rules:

- It MUST NOT be consulted by rule resolution in v1.
- Runners MAY emit it when information is available.
- Missing keys do not invalidate an artifact.
- Values are free-form strings in v1; strict per-key schema is deferred
  until promotion trigger (first downstream hard-failure on missing key).

Minimal recommended keys (v1):

- `embedding_model_id`
- `embedding_provider`
- `embedding_engine`
- `rule_set_file_digest` (optional raw SHA-256 over rule-file bytes,
  for byte-level forensics when canonical `rule_set_hash` is used for
  semantic identity)

### Resolver determinism contract (v1)

Resolver behavior MUST be deterministic: identical
`(rule_set_hash, pipeline_declared, rule_inputs)` MUST produce
byte-identical `pipeline_resolved`, `pipeline_effective`, and
`pipeline_effective_hash`.

### Resolver test-fixture commitment (v1)

Resolver implementations MUST ship with golden fixtures exercised in CI:

`(declared_pipeline, rule_set, context) -> (resolved, effective, hash)`

These fixtures are the primary guardrail against silent drift in stage
ordering, serialization shape, and hash identity.

### Validation phasing (v1)

Hard-constraint validation runs in two phases tied to artifact
assembly:

- **Pre-run validation**: executes after Tier 1/2/3 binding and before
  runner execution; covers structural and bound-parameter invariants.
- **Post-selection validation**: executes after runner selection
  outcomes are known (Tier 4 bound values) and before final artifact
  persistence; covers selection-dependent invariants.

`pipeline_effective_hash` MUST be finalized only after post-selection
validation passes so the persisted hash reflects a fully validated
effective pipeline.

### Hard constraints vs soft heuristics

Two kinds of rules, distinguished by enforcement:

- **Hard constraints** are mathematical or library-level invariants
  (`pca.n_components <= min(input_dim, n_samples - 1)`,
  `umap.n_components < input_dim`, ward linkage requires Euclidean).
  Violations raise; they are not overridable.
- **Soft heuristics** are preferred values per context range
  (`pca.n_components_for_downstream.hdbscan = 30 when input_dim_le 768`).
  They are advisory and can be overridden by an explicit per-pipeline
  parameter on the declared pipeline.

Both are recorded the same way in artifacts (resolved value + rule
set version); the validator treats them differently.

#### Hard constraints (normative list)

The following constraints are mandatory in this design. Most are
checked pre-run; constraints that depend on selection outcomes are
checked post-selection before artifact persistence. Constraint failures
are hard errors unless the enforcement column says `warning`.

| Constraint | Enforcement | Why |
|---|---|---|
| `pca.n_components <= min(input_dim, n_samples - 1)` | error | PCA rank cannot exceed matrix rank bounds |
| `umap.n_components < input_dim` | error | UMAP projection must strictly reduce dimensionality in this design |
| `dbscan.eps > 0` and `dbscan.min_samples >= 2` | error | DBSCAN parameter domain validity |
| `hdbscan.min_cluster_size >= 2` | error | HDBSCAN parameter domain validity |
| `agglomerative.linkage == "ward"` implies metric is Euclidean-compatible | error | Ward objective is variance-minimization in Euclidean geometry |
| Unknown declared stage name for v1 stage vocabulary | error | Prevents unsupported pipeline shapes |
| Unknown method/composite name in v1 allowlist | error | Prevents unregistered selection behavior |
| Rule references an undeclared `context_keys` input | error | Preserves deterministic and auditable rule evaluation |
| If `selection_evidence` exists: chosen parameter must be in sweep range and represented in full selection curve artifact (**post-selection check**) | error | Selection trace must be reproducible and auditable |
| Unsupported skip predicate form (outside v1 vocabulary) | error | Bounds resolver complexity and behavior in v1 |
| `pipeline_effective_hash` is computed from effective stage sequence **and** bound stage params | error | Distinguishes semantically different computations that share stage names |
| No dim-reduction stage with `input_dim > 100` for DBSCAN/HDBSCAN | warning | Quality guardrail against high-dim distance concentration |

### Stage skip semantics (Option B)

Rules MAY remove stages from the effective pipeline. Skip is a
**first-class rule property**, not an emergent `null` from a
parameter resolver. When a skip fires:

- The declared pipeline retains the stage.
- The resolved pipeline marks the stage with `"skipped": true` and a
  `"skip_reason"` string.
- The effective pipeline omits the stage.
- An entry is added to `skipped_stages` with the rule that fired.

When two distinct declared pipelines resolve to the same effective
pipeline (an alias), both artifacts MUST list each other in
`aliases`. Alias detection runs at study-design time (to surface
duplicates before running) and at study-analysis time (to correctly
merge bit-identical methods).

The alternative ("rules adapt parameters but never remove stages")
was considered and rejected because it forces real work in cases
where the stage is provably useless (e.g. PCA on a 100-dim embedding
to 100 components), and because the transparency cost of recording
declared/resolved/effective separately is small once the skip
behaviour is first-class.

#### Skip predicate vocabulary (v1)

Allowed predicate forms:

- `input_dim_le`
- `input_dim_gt`
- `n_samples_le`
- `n_samples_gt`
- `when_downstream_contains`
- `when_downstream_param`
- `all_of` (conjunction-only composition)

Not supported in v1:

- Disjunction (`any_of` / OR)
- Negation (NOT)
- Arbitrary expression strings or embedded code

Any future expansion of skip-expression power MUST satisfy the
decision-log gate: deterministic AST representation, no eval strings,
bounded complexity, and explicit safety review.

#### Normalize no-op policy (v1)

- Declared `normalize` is executed unless an explicit skip rule fires.
- Unit-norm detection may be recorded as diagnostics but MUST NOT
  implicitly skip `normalize` in v1.
- Alias behavior remains tied to explicit skip rules, not implicit
  runtime auto-no-op logic.

### Embedding-dim awareness

Initial context keys are `embedding_dim` and `n_samples`. These cover
the rules that genuinely depend on input characteristics:

- PCA `n_components` (hard cap on `input_dim`; skip-when threshold)
- Whether to PCA at all (skip when `input_dim_le 200`)
- GMM `covariance_type` (`full` if d ≤ 20, `tied` if d ≤ 50,
  `diag` otherwise — applied to post-reduction dim)
- HDBSCAN/DBSCAN dim-reduction prerequisite (warn or block when
  `input_dim > 100` and no reduction stage is present)
- UMAP `n_components` (hard cap on `input_dim`)

Additional context keys (`embedding_model_id`, `sparsity`,
`embedding_norm_distribution`) are deferred until a specific pipeline
forces the issue.

### Per-method parameter-tier mapping (normative)

The table below is the normative parameter-tier assignment for terminal
clustering methods. A parameter should appear in exactly one tier for a
given method.

| Method | Tier 1: global hygiene | Tier 2: adjacency-derived | Tier 3: dataset constants | Tier 4: swept/selected |
|---|---|---|---|---|
| `kmeans` | `random_state`, `init`, `n_init`, `max_iter`, `tol` | `distance_metric` (from upstream normalize/metric geometry) | none by default | `k` |
| `spherical_kmeans` | `random_state`, `n_init`, `max_iter`, `tol` | requires normalized geometry (explicit normalize stage or equivalent) | none by default | `k` |
| `gmm` | `random_state`, `n_init`, `max_iter`, `reg_covar` | `covariance_type` from post-reduction dimensionality/geometry | none by default | `k` |
| `hdbscan` | `cluster_selection_method`, `alpha`, `allow_single_cluster`, `core_dist_n_jobs`, `approx_min_span_tree`, `random_state` | `metric` from upstream geometry/normalization | default `min_cluster_size` baseline | `min_cluster_size` (when swept), optionally `min_samples` |
| `dbscan` | algorithm defaults, deterministic seed policy where applicable | `metric` from upstream geometry/normalization | default `min_samples` baseline | `eps` (and optionally `min_samples`) |
| `agglomerative` | implementation defaults (`compute_full_tree`, deterministic settings) | `linkage`/`metric` compatibility from upstream geometry | none by default | `n_clusters` or `distance_threshold` |
| `spectral` | `assign_labels`, `random_state`, eigensolver defaults | `affinity` type (`rbf`/`nearest_neighbors`/`precomputed`) and affinity params | `n_neighbors` when using NN affinity | `k` |
| `leiden` | `random_state`, `n_iterations`, `objective_function` | graph weighting/metric rules inherited from upstream graph builder | kNN graph neighbor count baseline | `resolution_parameter` |
| `louvain` | `random_state`, implementation defaults | graph weighting/metric rules inherited from upstream graph builder | kNN graph neighbor count baseline | `resolution_parameter` |

Notes:

- Single-fit runs can hold Tier-4 parameters fixed (no sweep). They
  remain Tier-4 semantically because they are the parameter class whose
  sweep would change cluster-count behavior.
- Where a method row says "none by default" in Tier 3, projects may add
  dataset constants later if repeated empirical tuning justifies it.

## Worked example (end-to-end resolution trace)

This worked example shows the full lifecycle for one sweep-and-select
pipeline on twenty_newsgroups.

### Inputs

- Rule set: `rules-v1.2.3`
- Dataset: `twenty_newsgroups`
- Context keys: `embedding_dim=768`, `n_samples=18846`
- Declared pipeline:
  `["embed", "normalize", "pca", "kmeans+silhouette+kneedle"]`

### Resolution

1. **Tier 1 (global hygiene)** binds:
   - `random_state=42`
   - `kmeans.n_init=20`
   - `kmeans.max_iter=300`
   - `kmeans.init="k-means++"`
2. **Tier 2 (adjacency-derived)** binds:
   - `pca.n_components=100` (`kmeans` downstream + `embedding_dim=768`)
   - `kmeans.distance_metric="cosine"` (normalize stage upstream)
3. **Tier 3 (dataset constants)** contributes no additional k-means
   parameters for this rule set.
4. **Tier 4 (swept)** binds selection policy:
   - `k_range=[2,3,5,8,10,15,20,30,50]`
   - metric=`silhouette`, rule=`kneedle`

### Effective pipeline

No skip rule fires (`embedding_dim=768` does not satisfy
`pca.skip_when input_dim_le 200`), so:

- `pipeline_declared == pipeline_effective`
- `pipeline_effective_hash` hashes stage sequence + bound stage params
  for `[embed, normalize, pca(n_components=100), kmeans(metric=cosine)]`

### Selection evidence (abbreviated)

```json
{
  "sweep_range": [2, 3, 5, 8, 10, 15, 20, 30, 50],
  "selection_metric": "silhouette",
  "selection_rule": "kneedle",
  "selection_curve_artifact_ref": "kmeans_selection_curve.json",
  "chosen_k": 10
}
```

### Final artifact highlights

- `operation_type="cluster_pipeline"`
- `method_name="kmeans+silhouette+kneedle"`
- `rule_set_version="rules-v1.2.3"`
- `rule_inputs={"embedding_dim":768,"n_samples":18846}`
- `pipeline_effective_hash=<sha256 over effective sequence+params>`
- `cluster_count=10`

### Alias contrast under different context

If the same declared pipeline is run with `embedding_dim=128`, the
PCA skip rule can fire. Then:

- `pipeline_declared` still includes `pca`
- `pipeline_effective` becomes `[embed, normalize, kmeans]`
- it may alias another declared pipeline that omitted PCA explicitly
  (same effective hash)

## Pipeline taxonomy

The full enumeration of preprocessing × terminal-clusterer
combinations yields ~45 candidate pipelines. The rule-set + alias
detection collapse this to a smaller set of distinct effective
pipelines per dataset. The taxonomy guidance:

- **Drop outright** (mathematically weak or untunable):
  - `Embedding → normalize → DBSCAN` (DBSCAN `eps` untunable in high-D
    even on the unit sphere)
  - Any density-based method on raw high-D embeddings without a
    dim-reduction stage and without a cosine metric
- **Treat as ablation only** (not as competitive method):
  - The "no normalize" variant of any pipeline that goes through a
    cosine-based step downstream — keep only when the study is
    explicitly about the normalize ablation
- **Detect as alias under rules**:
  - `Embedding → PCA → HDBSCAN` aliases `Embedding → HDBSCAN` for
    `input_dim ≤ 200` (PCA skipped)
  - `Embedding → normalize → UMAP(metric=cosine) → ...` aliases
    `Embedding → UMAP(metric=cosine) → ...` (cross-stage skip rule)
- **Method-name discipline**:
  - `Embedding → normalize → spherical_kmeans` and
    `Embedding → normalize → kmeans+cosine_assignment` are *different
    methods* (different centroid update rule); both are kept and
    named distinctly.

The canonical menu (the set most clustering benchmarks report)
collapses to roughly: kmeans / spherical kmeans / GMM (with optional
PCA), HDBSCAN (with PCA or UMAP), agglomerative (ward post-PCA,
average post-normalize), spectral, Leiden/Louvain on kNN graph, and
the BERTopic-style UMAP→HDBSCAN.

## Decision log

| Date | Decision | Alternatives | Rationale |
|---|---|---|---|
| 2026-04-22 | V1 scope is frozen to `hdbscan`, `kmeans+silhouette+kneedle`, and `gmm+bic+argmin`; broader catalog entries are deferred. | Treat full catalog as in-scope for first implementation. | Keeps implementation bounded while preserving an extensible catalog design. |
| 2026-04-22 | Rule-set versioning is single and shared across `global_hygiene`, `adjacency_rules`, `dataset_constants`, and `selection_policies` in v1. | Independent per-tier sub-versioning. | Preserves atomic provenance and rollback behavior for v1. |
| 2026-04-22 | Treat HDBSCAN as a special case alongside k-sweep methods rather than retrofitting it into a generalized envelope. | Force HDBSCAN into a `selection_evidence: null` envelope from day one. | One example is overfitting; defer unification until ≥2 sweep runners exist and the common shape is empirical, not speculative. |
| 2026-04-22 | Operation type and rule set are separate provenance fields. | Collapse them — rule set version "is" the operation type. | Operation type is the contract; rule set is one configuration within that contract. They version at different cadences and serve different purposes (schema vs. style). |
| 2026-04-22 | Cross-rule-set diff tooling is deferred. Trigger: ≥2 rule-set versions are in active production use. | Build diff tooling in v1. | Contract can ship without dedicated comparison UX; add when operational need appears. |
| 2026-04-22 | Generic multi-operation rule registry is deferred. Trigger: second non-clustering operation requires rule-driven parameterization. | Extract generic registry in v1. | Avoid premature abstraction from a single operation type. |
| 2026-04-22 | Rules MAY skip stages; the artifact records declared, resolved, and effective pipelines separately (Option B). | (A) Rules adapt parameters only, never remove stages. (C) Skip is advisory; user must explicitly accept. | (A) forces real work when a stage is provably useless and creates noisy near-no-op operations. (C) is annoying at scale (many pipelines × many embedding models). (B) keeps adaptation while making aliasing detectable. |
| 2026-04-22 | `pipeline_effective_hash` covers effective stage sequence plus bound stage parameters. | Sequence-only hash over stage names. | Sequence-only hashing collapses semantically distinct computations. Parameter-inclusive hashing preserves alias detection fidelity. |
| 2026-04-22 | `pipeline_effective_hash` and effective `recipe_hash` must match whenever both are materialized. | Keep independent hashes for pipeline and recipe identities. | A single semantic identity avoids drift and duplicate identity contracts. |
| 2026-04-22 | V1 in-scope methods MUST materialize effective identity through the existing recipe mechanism, so hash equality is concretely enforced in production. | Treat recipe linkage as aspirational for v1 methods. | Ensures the hash-identity contract is exercised immediately, not deferred. |
| 2026-04-22 | Canonical serialization for `pipeline_effective_hash` uses normalized payload + sorted-key JSON + UTF-8 SHA-256. | Implementation-defined serialization. | Deterministic cross-environment identity is required for alias detection and provenance comparison. |
| 2026-04-22 | The selection step is part of `method_name` (e.g. `kmeans+silhouette+kneedle`), not a hidden post-processing decoration. | Treat selection as a separate metadata field on a base method name. | Two methods that share a base algorithm but differ in selection rule produce different labelings on identical inputs — they are different methods for comparison purposes. |
| 2026-04-22 | Selection-catalog expansion is staged by infrastructure cost: (i) selection-rule expansion on existing terminals, (ii) terminal-stage expansion without new infra, (iii) infrastructure-prerequisite expansion. | One-step expansion of the full deferred catalog. | Matches rollout order to engineering complexity and validation cost. |
| 2026-04-22 | Artifacts record `rule_set_version` + resolved pipeline parameters (`pipeline_resolved`) + `rule_inputs`, NOT per-parameter rule references. | Tag each parameter with the rule that produced it. | Per-parameter rule references age badly (rule keys get refactored, references go stale). Triple of (rule set + resolved value + inputs the rules saw) is sufficient to re-derive any parameter. |
| 2026-04-22 | Initial rule context keys are `embedding_dim` and `n_samples` only. | Include `embedding_model_id`, `sparsity`, `embedding_norm_distribution` from day one. | These two cover the rules that genuinely depend on input characteristics today. Add others when a specific pipeline forces them; premature context keys add validation surface without delivering value. |
| 2026-04-22 | Stage skip is a first-class rule property (`skip_when` predicate), not an emergent `null` from a parameter resolver returning no value. | Use parameter `null` as an implicit skip signal. | First-class skip surfaces the rationale explicitly in artifacts and lets validation reason about skip rules independently of parameter rules. |
| 2026-04-22 | Selection evidence stores compact summary inline and full metric curves in sibling artifacts. | Keep full metric curve inline in summary payload. | Curves can be large; artifact references keep summary payload compact while preserving full auditability. |
| 2026-04-22 | `input_audit_metadata` is optional and non-resolver in v1; promotion to required schema triggers only when a downstream consumer hard-fails on missing key. | Make audit metadata required or resolver-consultable in v1. | Keeps v1 resolver deterministic while preserving a concrete promotion trigger; avoids premature schema lock-in. |
| 2026-04-22 | Normalize is not implicitly skipped via unit-norm detection in v1. | Auto-skip normalize whenever vectors appear unit-normalized. | Avoids hidden runtime-dependent aliasing; skip behavior remains rule-driven and auditable. |
| 2026-04-22 | Skip predicates are restricted to a small v1 vocabulary (comparison + conjunction). Future expansion gate: deterministic AST, no eval strings, bounded complexity, explicit safety review. | Support full boolean/expression language in v1. | Keeps resolver bounded and testable while preserving key skip use-cases. |
| 2026-04-22 | Resolver output is deterministic for identical `(rule_set_hash, pipeline_declared, rule_inputs)` inputs. | Permit implementation-defined non-determinism. | Determinism is required for reproducibility and stable alias detection. |
| 2026-04-22 | Resolver CI includes golden fixtures: `(declared_pipeline, rule_set, context) -> (resolved, effective, hash)`. | Leave fixture coverage to implementer discretion. | Prevents silent drift in serialization, resolution, and hash behavior. |
| 2026-04-22 | HDBSCAN target-schema migration is forward-write only in v1 (no backfill by default). | Backfill existing HDBSCAN artifacts during v1 rollout. | Reduces migration risk/scope; backfill remains optional if downstream query needs it. |
| 2026-04-22 | `rule_set_hash` is computed from canonicalized parsed YAML (sorted-key JSON, UTF-8 SHA-256), not raw file bytes; raw file digest MAY be persisted as `rule_set_file_digest` in `input_audit_metadata`. | Hash raw YAML bytes directly for `rule_set_hash`. | Canonical hashing is stable across non-semantic formatting edits while optional raw digest preserves byte-level provenance. |
| 2026-04-22 | Hard-constraint validation is phased: pre-run for structural/bound-parameter checks, post-selection for selection-dependent checks. | Single pre-execution validation pass. | Selection-dependent invariants cannot be evaluated until the runner produces chosen Tier-4 values. |
| 2026-04-22 | `pipeline_effective_hash` is finalized only after post-selection validation passes. | Compute and persist hash before post-selection validation. | Prevents persisting identity for an unvalidated effective pipeline. |

## Open questions

None currently open for v1 closure. Future deferrals are captured as
decision-log entries with explicit triggers.

## Implementation triggers

Each piece of this design becomes worth building at a specific
trigger; until then it stays in this document.

| Trigger | What to build |
|---|---|
| First non-HDBSCAN clustering runner is added | (a) `operation_type` field added to existing HDBSCAN artifacts (one string, no new schema); (b) the new runner's artifact uses the schema in this document; (c) minimal rule-resolver covering whichever Tier 2/3 rules the new runner exercises. |
| Second sweep-based runner is added | (a) Extract the rule-resolver into shared infrastructure; (b) formalize rule-set YAML canonical hashing for `rule_set_hash` (with optional raw `rule_set_file_digest` tracking); (c) add `pipeline_declared` / `pipeline_effective` fields to all clustering artifacts. |
| First conditional skip rule fires in production | Add `skipped_stages` and `aliases` fields; build the alias-detection tooling for study-design time. |
| First non-clustering operation type appears | Extract the operation-type registry; introduce per-operation rule-set namespaces; revisit cross-operation hygiene rules (e.g. `random_state` shared across operations). |
| First multi-embedding-model study | Add `embedding_dim`-keyed lookup tables for any rules that currently use a fixed value across all dims; add `embedding_model_id` to `rule_inputs` if model-specific quirks emerge. |
| Two or more rule-set versions are in active production use | Add cross-rule-set diff tooling showing rule-level and resolved-parameter deltas for the same declared pipeline/context. |
| A downstream system hard-fails on missing `input_audit_metadata` fields | Promote `input_audit_metadata` to a versioned schema with required keys and per-key format constraints. |

## Appendix A: HDBSCAN Current-to-Target Mapping (v1)

This appendix maps the existing HDBSCAN runner payload shape
(`src/study_query_llm/pipeline/hdbscan_runner.py`) to the target
clustering provenance schema in this design.

| Current field | Target field | Mapping notes |
|---|---|---|
| `structured_results.hdbscan_summary.method_name` | `method_name` | Direct carry-forward (`hdbscan`). |
| `structured_results.hdbscan_summary.n_samples` | `n_samples` | Direct carry-forward. |
| `structured_results.hdbscan_summary.n_features` | `n_features` | Direct carry-forward. |
| `structured_results.hdbscan_summary.cluster_count` | `cluster_count` | Direct carry-forward. |
| `structured_results.hdbscan_summary.cluster_ids` | `cluster_ids` | Direct carry-forward. |
| `structured_results.hdbscan_summary.cluster_sizes` | `cluster_sizes` | Direct carry-forward. |
| `structured_results.hdbscan_summary.noise_count` | `noise_count` | Direct carry-forward. |
| `structured_results.hdbscan_summary.noise_fraction` | `noise_fraction` | Direct carry-forward. |
| `structured_results.hdbscan_summary.parameters` | `pipeline_resolved` stage params | Move from algorithm-local `parameters` bag into resolved stage-parameter entries. |
| `structured_results.hdbscan_cluster_labels.cluster_labels` | labels artifact payload | Continue as label artifact content. |
| `hdbscan_summary.json` | summary artifact | May remain, but schema contents are upgraded to target fields. |
| `hdbscan_labels.json` | labels artifact | May remain with equivalent semantic payload. |

New required fields in target schema (not currently in HDBSCAN summary):

- `operation_type`
- `operation_version`
- `rule_set_version`
- `rule_set_hash`
- `rule_inputs`
- `pipeline_declared`
- `pipeline_resolved`
- `pipeline_effective`
- `pipeline_effective_hash`

Cutover policy (v1):

- Target-schema migration is forward-write only.
- Existing historical HDBSCAN artifacts remain as-is.
- Readers must tolerate both historical and target schema shapes during
  transition.
- Backfill remains optional and should only be introduced if a concrete
  query path requires uniform historical shape.

## Plan-readiness checklist

- [x] v1 method/composite scope and non-goals frozen.
- [x] Selection catalog is explicitly gated for v1.
- [x] `pipeline_effective_hash` and effective `recipe_hash` relationship is decided.
- [x] V1 methods are materialized through the existing recipe mechanism.
- [x] Canonical hash serialization contract is specified.
- [x] Selection-evidence persistence shape is decided.
- [x] Minimal `input_audit_metadata` contract is specified.
- [x] V1 skip-predicate vocabulary ceiling is specified.
- [x] Normalize no-op policy is decided for v1.
- [x] Hard-constraint/validator severity matrix includes key enforcement points.
- [x] Stage vocabulary and tier precedence are documented.
- [x] HDBSCAN current-to-target mapping appendix is documented.
- [x] HDBSCAN v1 cutover policy (forward-write only) is documented.
- [x] Resolver determinism contract is normative.
- [x] Resolver golden fixtures are mandatory in CI.
- [x] Validation phasing (pre-run + post-selection) is normative.
- [x] `rule_set_hash` derivation from canonicalized parsed YAML is fixed.
- [x] Deferred decisions are converted to explicit trigger-gated decisions.
- [x] Open questions are closed for v1.

## Cross-references

- [`docs/DATA_PIPELINE.md`](../DATA_PIPELINE.md) — five-stage canonical pipeline; this design lives inside `analyze`.
- [`docs/living/METHOD_RECIPES.md`](../living/METHOD_RECIPES.md) — v1 in-scope methods materialize effective pipeline identity via recipe registration; `pipeline_effective_hash` aligns with effective `recipe_hash`.
- [`docs/living/SCHEDULING_PROVENANCE_BOUNDARY.md`](../living/SCHEDULING_PROVENANCE_BOUNDARY.md) — sweep ranges and selection rules are provenance, not orchestration.
- [`docs/living/ARCHITECTURE_CURRENT.md`](../living/ARCHITECTURE_CURRENT.md) — overall system architecture this slot fits into.
- `src/study_query_llm/pipeline/hdbscan_runner.py` — current sole clustering runner; the degenerate single-fit case in this framework.
- `src/study_query_llm/algorithms/clustering.py` — `silhouette_score_precomputed`, `pairwise_ari`, `compute_stability_metrics`; the partial selection-evidence machinery already exists.
- `src/study_query_llm/algorithms/recipes.py` — existing recipe registry; resolved pipelines under this design will register here.
