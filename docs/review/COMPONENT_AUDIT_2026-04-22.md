# Component Audit — 2026-04-22

Status: review artifact (one-shot)
Owner: documentation-maintainers
Scope: pipeline core, source-spec/parser layer, ingestion, backup verifier, tests, docs parity
Method: 9 parallel `very thorough` read-only audits + top-down synthesis

## Purpose

Forward-looking review of the recently-changed surface area (`source_specs` registry,
`twenty_newsgroups` parser, snapshot `category_filter`, HDBSCAN unknown-`k` runner,
clustering ingestion, and backup verifier) to surface bugs, contract drift, and doc
parity gaps **before** the next round of pipeline work.

This document is an audit artifact; concrete fixes belong in PRs and the
`DOC_PARITY_LEDGER.md` claim updates listed under §6.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Cross-Cutting Themes](#2-cross-cutting-themes)
3. [Per-Component Findings](#3-per-component-findings)
4. [Highest-Impact Fix Candidates](#4-highest-impact-fix-candidates)
5. [Test Coverage Gaps](#5-test-coverage-gaps)
6. [Doc Parity Actions](#6-doc-parity-actions)
7. [Open Questions for Human Judgment](#7-open-questions-for-human-judgment)
8. [Audit Methodology](#8-audit-methodology)
9. [Cross-Critique Addendum](#9-cross-critique-addendum-2026-04-22)

---

## 1. Executive Summary

**Top-down assessment:** No critical data-loss or wrong-target-write bugs were found.
The five-stage pipeline contract is largely faithful to the implementation. The
weakest surfaces are (in order):

1. **Doc/code drift in C041** — `CURRENT_STATE.md` line 25 and the `DOC_PARITY_LEDGER`
   entry say the canonical recipe has *four* component methods *including*
   `pca_svd_project`, but `recipes.py:159-189` defines a **3-stage** composite recipe
   with notes stating *"No PCA projection stage."* The doc conflates "registered
   component methods" (4) with "stages in the composite recipe" (3). Active doc
   defect.

2. **`category_filter` typed-equality has four silent-misfire paths** that span
   `types.py`, `snapshot.py`, and individual parsers. Strict typed equality is the
   intended contract, but several legitimate uses currently fail silently. See
   [§2.1](#21-category_filter-typed-equality-is-load-bearing-and-fragile).

3. **`verify_db_backup_inventory.py` always exits 0**, including on Azure failures,
   manifest mismatches, and DB count mismatches. It also has **zero test coverage**.
   A verifier that cannot signal failure in CI is operationally close to inert.
   Single-cause, high-impact fix.

4. **HDBSCAN runner has multiple latent reproducibility risks**: no random seed,
   no `core_dist_n_jobs` pin, default Euclidean metric (likely wrong for embeddings),
   silent zeroing of mismatched-length probability arrays, and `_normalize_rows`
   may mutate the caller's buffer. The runner only ships under a phase-1 BANK77
   path today, so blast radius is contained — but the contract debt grows with
   phase-2.

5. **Stage-numbering inconsistencies** in `DOC_PARITY_LEDGER.md` claims C043,
   C044, C045 — they say "stage-2 snapshot" / "stage-4 analyze" while
   `DATA_PIPELINE.md` defines parse=2, snapshot=3, embed=4, analyze=5. Pure doc
   defect; downstream readers will be confused.

6. **"Skip vs raise" policy is inconsistent across parsers** — SemEval silently
   drops short rows, AuSeM raises, BANK77 raises on empty text, Estela skips
   length-invalid prompts, twenty_newsgroups drops silently by length. There is
   no project-wide policy and no count-of-skipped diagnostic.

7. **`ingest_result_to_db` race condition**: `run_key_exists_in_db` runs in a
   separate session from the actual insert. Idempotency relies on a unique index
   added by `add_sweep_worker_safety` migration that is **not run in tests** (only
   `init_db`). In test DBs and any environment without the migration, two
   concurrent ingests can create duplicate rows.

8. **`twenty_newsgroups` text-length filter is product-shaping but undocumented**
   — `10 < len(text) <= 1000` drops a substantial fraction of real bydate posts;
   `parser_version` is `v1`; `extra` does not flag the filter. Either intended
   benchmark contract (then document) or oversight (then bump version).

The remaining ~40 findings are individually moderate to low. No single
component is broken; the audit's value is mostly in cross-cutting patterns
(§2) and the C041/HDBSCAN/verifier triple (§4).

---

## 2. Cross-Cutting Themes

### 2.1. `category_filter` typed-equality is load-bearing and fragile

`SubquerySpec.category_filter` is the new mechanism for snapshot-time slicing
(claims C049, C050). The contract is **strict typed equality** against
top-level keys in the parsed `extra_json` object. Audit found four ways this
silently misfires:

| # | Failure mode | Evidence | Severity |
|---|---|---|---|
| 1 | Missing key vs explicit `null` are indistinguishable: `parsed.get(filter_key)` returns `None` for both. A filter `{"k": [None]}` matches both rows where `k` is absent and rows where `k=null`. | `snapshot.py:108-110` | High |
| 2 | `int 1` vs `float 1.0` after `json.loads` are different types under `_typed_equals`. JSON does not preserve int/float distinction the way Python does, so legitimate-looking filters can yield empty snapshots. | `snapshot.py:84-86,105-111` + `types.py:13-44` | High |
| 3 | `SemEval` parser writes `"gold_count"` as **either `int` or `str`** depending on whether the raw value is `.isdigit()`. Same logical count never matches under typed equality. | `semeval2013_sra_5way.py:206` | High (SemEval-only today) |
| 4 | `SubquerySpec` constructor only validates/freezes `category_filter` itself; `numpy` scalars (e.g. `numpy.int64`) are not in `_ALLOWED_CATEGORY_VALUE_TYPES`, so filters built from arrays/dataframes raise at construction without an obvious workaround. | `types.py:10,30-35` | Medium |

Additionally, the docs use the notation `extra.newsgroup` / `extra.split`
(`CURRENT_STATE.md:60`, claim C050) which suggests JSONPath-style nested access,
but the implementation matches **top-level keys of the parsed `extra_json`
object**. Wording-only, but easy to misread.

**Recommended fix shape:** Document the contract precisely in `DATA_PIPELINE.md`
(missing key vs null, int vs float, allowed value types). Optionally add a
constructor-time int/float coalesce policy. Add a regression test for each of
the four failure modes.

### 2.2. "Skip vs raise" is inconsistent across parsers (silent data loss risk)

Parser layer policy on malformed input is inconsistent:

| Parser | Behavior | Evidence |
|---|---|---|
| `banking77` | **Raises** on empty text | `banking77.py:88-91` |
| `ausem` | **Raises** on missing required columns / bad `correct` value | `ausem.py:43-47,78-86` |
| `sources_uncertainty_zenodo` | **Raises** on empty fields, bad codes | `sources_uncertainty_zenodo.py:76-80,118-120` |
| `estela` | **Skips** length-invalid prompts silently | `estela.py:53-59,78-81` |
| `twenty_newsgroups` | **Skips** posts outside `10 < len(text) <= 1000` silently | `twenty_newsgroups.py:30-32,70-74` |
| `semeval2013_sra_5way` | **Skips** `len(row) < 3` answer rows and `len(row) < 6` gold rows silently | `semeval2013_sra_5way.py:99-123,145-149` |

There is no per-parser skip count or warning. None of the skip paths bump
`parser_version`. Any future change to filter constants is a silent contract
change.

**Recommended fix shape:** Decide policy (e.g. "raise unless `extra` records
the filter; track skipped count in `source_metadata` return"); apply
consistently; bump parser versions when filters change.

### 2.3. Fingerprint stability has multiple silent paths to instability

Claims C040 and C041 promise *scheduling-independent algorithmic identity* via
`fingerprint_json` / `fingerprint_hash`. Audit found these instability paths:

- `ingestion.py:280-283` — `record_method_execution` passes `metadata_json` with
  only `artifact_id` and `source`; `data_regime` and `manifest_hash` are **never
  forwarded**, even though `canonical_run_fingerprint` accepts them.
- `ingestion.py:191-217` — `recipe_hash` is only injected for algorithms in
  `COMPOSITE_RECIPES` (currently only `cosine_kllmeans_no_pca`). Any other
  algorithm with `recipe_json` registered on its `MethodDefinition` will not
  see `recipe_hash` in `config_json`. C041 is **not uniform** across all paths.
- `types.py:80-95` — `to_canonical_dict` does `int(self.sample_n)`; a float
  `2.7` becomes `2` and is hashed as if the user had asked for 2 — **without
  raising**. `sampling_seed` has the same coercion.
- `raw_call_repository.py:1410-1452` — `create_provenanced_run` UPDATEs an
  existing row's `config_json` / `fingerprint_json` / `fingerprint_hash` for
  matching `(request_group_id, run_key, run_kind)`. Re-running with a different
  config silently mutates historical fingerprint rows.
- `ingestion.py:97-98` — `primary_snapshot_id = min(sorted(set(snapshot_ids)))`,
  not "first listed by caller." Any caller assuming "primary = first" is wrong;
  the test only asserts the deterministic `min`.

**Recommended fix shape:** Document the fingerprint inputs and stability
boundary in one place (likely `SCHEDULING_PROVENANCE_BOUNDARY.md`). Forward
`data_regime` / `manifest_hash`. Reject (don't coerce) non-int `sample_n`.
Decide whether `record_method_execution` should be insert-only for new
fingerprints.

### 2.4. Idempotency has subtle gaps in three stages

| Stage | Gap | Evidence |
|---|---|---|
| `acquire` | Always re-downloads before consulting fingerprint/DB. The "no fetch on cache hit" contract is not enforced. | `pipeline/acquire.py:103-130`, `acquisition.py:91-108` |
| `snapshot` (reuse path) | `_collect_snapshot_artifact_uris` scans **all** `CallArtifact` rows, not the snapshot group's. O(artifacts in DB), grows with project age. | `snapshot.py:40-48` |
| `ingestion` | TOCTOU race between `run_key_exists_in_db` (separate session) and `create_run_group`. Test DBs lack the unique-index migration that would catch concurrent duplicates. | `ingestion.py:32-44,100-110,140-152` |

The `force=True` path in `snapshot.py:225,290-303` always creates a new group.
Multiple groups with the same `(source_dataframe_group_id, spec_hash,
resolved_index_hash)` triple can coexist; `_find_existing_snapshot_group`
returns `MAX(id)`. Document if intended.

### 2.5. Silent failure pattern (`except Exception: pass` / always-zero exit)

| Site | Behavior |
|---|---|
| `ingestion.py:287-307` | Snapshot/embedding-batch linking errors swallowed; ingest reports success |
| `verify_db_backup_inventory.py:131-140` | Azure exceptions printed and `return 0` |
| `verify_db_backup_inventory.py:121-128` | Skipped DB-count comparison if either side errors; no failure signal |
| `hdbscan_runner.py:198-208` | `mean_membership_probability` / `mean_outlier_score` become `0.0` if probability/outlier array length mismatches `n_samples` |
| `ausem.py:78-86` | Broad `except Exception` around `pd.isna` |

Combined effect: several "verifier" or "linker" code paths cannot fail loudly
in CI.

### 2.6. Per-parser variants live outside the registry

`ausem.py` exposes `parse_ausem_problem1_snapshot` ... `parse_ausem_problem4_snapshot`
(`ausem.py:202-234`). `sources_uncertainty_zenodo.py` exposes
`parse_sources_uncertainty_pm_snapshot` (lines 166-173). Both are referenced by
`DOC_PARITY_LEDGER` claims C043, C045 but **none are in `ACQUIRE_REGISTRY`** —
callers must pass `parser=`, `parser_id=`, `parser_version=` manually.

**Recommended fix shape:** Either register each variant as a first-class
`DatasetAcquireConfig` (clean, but bloats registry) or document the variant-
override pattern in `DATA_PIPELINE.md` (lighter touch).

### 2.7. Mutable interior of "frozen" dataclasses

`SnapshotRow.extra`, `StageResult.metadata`, `StageResult.artifact_uris`,
`ParserContext.artifact_uris`, `ParserContext.source_metadata` are all `dict`
fields on `frozen=True` dataclasses. Callers can still mutate the dict in
place, breaking immutability expectations downstream.

Evidence: `types.py:48-114`, `parser_protocol.py:21-30`.

### 2.8. `encoding='utf-8'` near-compliance

The repo-wide rule is well-followed in production. The single non-trivial
slip-up:

- `verify_db_backup_inventory.py:95` — `load_dotenv(REPO / ".env")` lacks
  `encoding="utf-8"`. The companion `upload_jetstream_pg_dump_to_blob.py:70`
  does pass it. Trivial fix.

`twenty_newsgroups._decode_message` (`twenty_newsgroups.py:63-67`) uses
`utf-8` first then falls back to `latin-1` with `errors="replace"` (the
`errors="replace"` is moot under `latin-1`, which decodes every byte). This
is *deterministic* but produces semantically wrong text for Windows-1252
content. See §3.4.

---

## 3. Per-Component Findings

Severity buckets are summaries; the agent reports include `file:line`
citations. Critical/High items below are the load-bearing risks.

### 3.1. `pipeline/snapshot.py` (357 lines)

**High:**
- `category_filter` missing-key vs null conflation (§2.1 #1).
- Reuse path scans all `CallArtifact` rows (§2.4).
- `force=True` can create duplicate groups with same hash triple; `MAX(id)` wins on read.
- `except Exception` on `DataFrame.query` masks unexpected pandas errors as user-input errors.
- `_collect_snapshot_artifact_uris` returns silently-incomplete dict on inconsistent DB state.

**Medium / Low:** redundant `to_canonical_dict()` recomputation, untyped `session` params, sparse docstrings on `snapshot()` public surface.

**Strengths:** unseeded sampling rejected (`:199-204`); deterministic mergesort on position (`:218`); JSON hashing uses UTF-8 explicitly (`:56-58`); strict `_typed_equals` (`:84-86`); idempotent reuse path (`:283-303`).

### 3.2. `pipeline/hdbscan_runner.py` (269 lines)

**High:**
- Default representation falls through to `"full"` if metadata omits it; out-of-band callers can bypass the rep guard (`:100-113`).
- Silent zeroing of mismatched-length probability/outlier arrays (`:198-208`).
- Soft determinism: no random seed, no `core_dist_n_jobs` pin (`:173-182`).
- Default `metric="euclidean"` (`:132`); cosine is more typical for embeddings without explicit `normalize_embeddings=True`.
- Parameter parsing errors bubble as opaque `int()`/`float()` failures (`:27-60`).
- `texts` length not asserted equal to embedding rows in this module (`:189-197`).
- `_normalize_rows` may mutate caller's buffer when `embeddings` is a writable view (`:74-76,163`).
- Zero-width feature dimension not rejected (`:117-121`).

**Medium:** Limited validation of HDBSCAN hyperparameters; no canonical label remapping; `representation` not echoed in `used_parameters`; minimal docstring.

**Strengths:** Strict non-`full` rejection (`:101-113`); UTF-8 deterministic JSON artifacts (`:79-85,248-266`); explicit noise handling (`noise_count`, `noise_label: -1`); stable sorted `cluster_ids`.

### 3.3. `pipeline/types.py` + `source_specs/parser_protocol.py`

**Critical:** Silent int-coercion of `sample_n` / `sampling_seed` in `to_canonical_dict` produces hashed payloads that don't match the user's intent (`types.py:80-97,88-92`). Float `2.7` is hashed as `2` without raising.

**High:** Missing key vs null and int vs float ambiguity in `category_filter` semantics (§2.1); `parser_id`/`parser_version` resolved by **object identity** (`parse.py:111-120`) — equivalent function references will fail; `SubquerySpec` only eagerly validates `category_filter`, deferring `label_mode`, sampling, `filter_expr` validation; mutable dicts inside frozen dataclasses (§2.7).

**Medium:** `ParserIdentity` (`parser_protocol.py:13-18`) is dead code; no `schema_version` on `to_canonical_dict`; `category_filter` AND-across-keys semantics not in docstring.

**Strengths:** Sorted, frozen, deterministic canonical form for `category_filter` (`types.py:12-45,98-102`); SHA-256 with `sort_keys=True, separators=(",", ":"), ensure_ascii=True` for `spec_hash` (`snapshot.py:56-58`); backward-compat preserved when `category_filter` is unset.

### 3.4. `source_specs/registry.py` + `twenty_newsgroups.py`

**High:**
- Aggressive 10-1000 char text filter (`twenty_newsgroups.py:30-32,70-74`) drops a substantial fraction of real bydate posts; not advertised in `extra` or `subset_profile`; `parser_version` is `v1`. Either intended benchmark contract or oversight.
- `text` includes RFC822 headers, signatures, quoted text — no body extraction (`:110-114,132-140`).
- `parse.py:99-107` raises `"no default parser registered"` for unknown `dataset_slug` instead of "unknown dataset" with valid keys.

**Medium:** `latin-1` fallback in `_decode_message` is byte-stable but semantically wrong for Windows-1252 content; categories derived from rows surviving filter (changing constants changes the entire label↔int mapping); no import-time slug uniqueness assert in `ACQUIRE_REGISTRY`; integer label gate in `parse.py:145-146` schema (no nullable type).

**Strengths:** Tar members `sorted` by name (`:97-99`); `pinning_identity` in `source_metadata` for stable `content_fingerprint` (the only existing parser doing this — see §2 pattern drift); registry-driven `parser_id`/`parser_version`.

### 3.5. Existing source-spec parsers (banking77, ausem, estela, semeval, sources_uncertainty_zenodo)

**Critical:**
- `semeval2013_sra_5way.py:99-123,145-149` — silent skip on short rows (`len < 3` for answers, `len < 6` for gold). Damaged upstream files lose join keys without diagnostic.
- `estela.py:70-71` — `pickle.load` on remote bytes. Supply-chain risk if bytes ever differ from acquisition; no integrity check beyond acquire-stage SHA.

**High:**
- `semeval2013_sra_5way.py:206` — `gold_count` is sometimes `int`, sometimes `str` (§2.1 #3); collides with `_typed_equals`.
- `ausem.py:169-173` — Optional columns indexed by `Unnamed: 6` / `Unnamed: 7` instead of header names. Fragile to upstream header changes.
- `banking77.py:64-69` + `parse.py:136-138` — `_coerce_extra` copies arbitrary extra columns as-is; future Parquet with NumPy/Arrow scalars can fail JSON serialization.
- `pinning_identity` is set on `twenty_newsgroups` only — not on the older parsers. `tests/pipeline/test_content_fingerprint.py` uses a synthetic `pinning_identity` for bank77-style; production specs don't have it.

**Medium:** Per-parser variants not registered (§2.6); skip-vs-raise inconsistency (§2.2); `subset_profile` field present in some parsers, absent in others.

**Strengths:** Pinned URLs (HF revision, git SHA, Zenodo record + DOI); deterministic ordering enforced; UTF-8 explicit on AuSeM CSV; required-column enforcement on AuSeM and sources_uncertainty.

### 3.6. `experiments/ingestion.py` (342 lines)

**Critical:**
- TOCTOU race between separate-session `run_key_exists_in_db` and ingest (§2.4).
- On `IntegrityError`, returns existing `id` and skips artifacts/k-steps/`record_method_execution`. Partial corruption from a prior crash is permanently un-repaired.
- `create_provenanced_run` UPDATEs existing row's fingerprint/config (§2.3).

**High:**
- Standalone ingests (no `request_group_id`) skip `ProvenancedRun` row entirely (`:255-285`).
- `data_regime` / `manifest_hash` never forwarded to fingerprint metadata (§2.3).
- `recipe_hash` only injected for algorithms in `COMPOSITE_RECIPES` (§2.3).
- `primary_snapshot_id = min(...)`, not "first listed" (`:97-98`).
- Linear scan over all request groups to infer `request_group_id` (`:240-250`); O(N) per ingest.

**Medium:** `print` instead of project logger; `except Exception: pass` around linking; `datetime.utcnow()` (naive) inconsistent with timezone-aware patterns elsewhere; `result: Any` type erasure.

**Strengths:** Single `session_scope` for the success path; deterministic snapshot id sort + dedupe; constructor-injected dependencies; `recipe_hash` injection genuinely absorbs recipe identity into fingerprint without changing fingerprint shape.

### 3.7. `scripts/verify_db_backup_inventory.py` (161 lines)

**High:**
- `main()` always returns `0` (`:131-140,156`); Azure failures, missing-blob mismatches, and DB count mismatches all exit success. No CI failure signal.
- Bidirectional inventory check missing — never reports orphan blobs in `db-backups` without manifests.
- Manifest with empty `blob_uri` is silently omitted from comparison (`:147-154`).
- Materializes full container listing in memory (`:86-92`); no pagination cap.

**Medium:** Uses `BlobServiceClient` directly instead of `AzureBlobStorageBackend`; `json.load` lacks try/except (single bad manifest aborts run with traceback); `load_dotenv` missing `encoding="utf-8"`; `argparse` not used despite multi-mode behavior.

**Strengths:** Manifests opened with `encoding="utf-8"`; read-only Azure usage; deterministic sorted output; no secret leakage in logs.

**Critical gap (cross-cuts §5):** Zero test coverage. Combined with always-zero exit, this script is operationally close to inert.

### 3.8. Tests audit (pipeline + datasets + experiments + scripts)

**Critical coverage gaps** (production behavior with no test):
- `verify_db_backup_inventory.py` — entire module untested.
- `twenty_newsgroups._decode_message` `latin-1` fallback never exercised.
- `ingest_result_to_db` never asserts `recipe_hash` is in `provenanced_runs.config_json`.
- `snapshot._apply_subquery` `filter_expr` branch untested.
- `snapshot()` `force=True` parameter untested.
- HDBSCAN output stability never asserted (only artifact existence).
- `embed.py` `intent_mean → label_centroid` alias untested.

**Test smells:**
- `_db(tmp_path)` helper duplicated across ~6 test files; should live in `conftest.py`.
- `test_acquire_snapshot_chain.py:97-99` lints the entire real `pipeline/` tree, not isolated; CI coupling risk.
- `test_bank77_pipeline_e2e.py:189-194` uses `>= 3` magic threshold.
- `test_runner.py:135` asserts "empty or missing" artifact dir on failure — fragile.

**Strengths:** `SubquerySpec` validation, canonical dict, sampling-seed enforcement, typed-equality "missing key → no match" all have direct tests; `embed` reuse cache verified by call-count.

### 3.9. `datasets/acquisition.py`

**High:**
- No expected-digest verification on download; only post-hoc SHA recording (`:91-108`). Truncated or wrong-mirror responses are accepted as authoritative.
- No streaming; full body in memory via `resp.read()` (`:41-50`). Large archives stress memory.
- `pipeline/acquire.py:103-130` always re-downloads before consulting fingerprint/DB (§2.4); no test enforces "no fetch on second call."
- `write_acquisition_bundle` (`:163-166`) lacks path-traversal guard on `relative_path`. Trusted today (specs are code), but the API itself is unsafe.
- No atomic write — partial bundles possible on crash.
- No retries on transient HTTP failures (`:41-54`).

**Medium:** Module docstring still says "Layer 0"; `urllib` redirect-following without allowlist (SSRF surface if URLs ever become dynamic); `extra_runner` merge can overwrite `script` / `git_commit`.

**Strengths:** `zenodo_file_download_url` rejects `..` and `/` in filenames; UTF-8 manifest writes; sorted manifest entries; `content_fingerprint` excludes `acquired_at`.

---

## 4. Highest-Impact Fix Candidates

These are the smallest changes with the largest risk-reduction. Listed in
suggested PR-grouping order.

### 4.1. C041 doc correction (1-line edit + ledger update)

`docs/living/CURRENT_STATE.md:25` — change "four component methods
(`mean_pool_tokens`, `pca_svd_project`, `kmeanspp_init`, `k_llmmeans`)" to
distinguish *registered components* (4, including `pca_svd_project` for
optional reuse) from *composite recipe stages* (3, no PCA).

`docs/review/DOC_PARITY_LEDGER.md` C041 — update claim text accordingly,
optionally add a new claim (proposed C051) for the 3-stage recipe contract.

### 4.2. `verify_db_backup_inventory.py` exit codes + minimum test

Single-PR scope:
- Track an `errors` accumulator across all check sections.
- Return `1` if any error / mismatch / Azure failure occurred.
- Add a unit test for `_same_host_port` and a smoke test for "manifest missing → exit 1."
- Add `encoding="utf-8"` to `load_dotenv`.

### 4.3. `category_filter` contract clarification + regression tests

- Add a "Filter semantics" subsection to `DATA_PIPELINE.md` that explicitly states:
  - Top-level keys of parsed `extra_json` (not nested).
  - Strict typed equality after `json.loads` (int ≠ float ≠ str).
  - Missing key → no match (current behavior); explicit `null` → matches `None` value.
- Add 4 regression tests:
  1. `int 1` filter against `float 1.0` extra value → no match (document expected).
  2. Missing key against `None` filter value → no match (document expected).
  3. SemEval `gold_count` mixed-type behavior (or fix the parser to always emit `int`).
  4. `numpy.int64` filter value error message clarity.

### 4.4. Stage numbering correction in ledger

`docs/review/DOC_PARITY_LEDGER.md` — claims C043, C044, C045 use "stage-2
snapshot" / "stage-4 analyze." Update to match `DATA_PIPELINE.md` (parse=2,
snapshot=3, embed=4, analyze=5). Status: change to `partial` until corrected,
or fix in place.

### 4.5. `ingest_result_to_db` — race + repair

- Apply `add_sweep_worker_safety` migration in test DBs (drop bare `init_db`
  for ingestion tests).
- Decide policy on partial-corruption skip (`ingestion.py:140-152`): currently
  silently returns existing id without filling missing data.
- Forward `data_regime` / `manifest_hash` from sweep metadata when present
  (`provenanced_run_service.py:191-194`).
- Add a test that asserts `ProvenancedRun.config_json["recipe_hash"]` for
  composite ingest.

### 4.6. Skip-vs-raise policy decision (cross-parser)

Document the project-wide policy in `DATA_PIPELINE.md` "Parser contract"
section. Audit each parser against the policy and bring outliers in line.
SemEval silent skips are the most concerning (data-shape rather than
data-quality).

---

## 5. Test Coverage Gaps

Production-code behaviors with **zero or weak** test coverage, in priority
order. Citations are to production code.

| # | Behavior | Production code | Severity |
|---|---|---|---|
| 1 | Entire `verify_db_backup_inventory.py` module | `scripts/verify_db_backup_inventory.py` | Critical |
| 2 | `recipe_hash` injection into `provenanced_runs.config_json` for composite ingest | `experiments/ingestion.py:262-267` | Critical |
| 3 | `twenty_newsgroups._decode_message` non-UTF-8 fallback | `datasets/source_specs/twenty_newsgroups.py:63-67` | High |
| 4 | `snapshot()` `force=True` semantics | `pipeline/snapshot.py:225,290-303` | High |
| 5 | `snapshot._apply_subquery` `filter_expr` branch | `pipeline/snapshot.py:185-194` | High |
| 6 | HDBSCAN label / score stability across runs | `pipeline/hdbscan_runner.py` | High |
| 7 | Acquisition idempotency: no fetch on second `acquire` call | `pipeline/acquire.py:103-130` | Medium |
| 8 | `_apply_category_filter` with `None` candidate values | `pipeline/snapshot.py:88-112` | Medium |
| 9 | `embed` `intent_mean → label_centroid` alias | `pipeline/embed.py:104-111` | Medium |
| 10 | `record_method_execution` retry idempotency vs UPDATE-on-conflict | `db/raw_call_repository.py:1410-1452` | Medium |
| 11 | `write_acquisition_bundle` path traversal guard (currently absent) | `datasets/acquisition.py:163-166` | Medium |
| 12 | Per-parser variants (AuSeM problem 2/4, sources_uncertainty PM) | `source_specs/ausem.py:202-234`, `sources_uncertainty_zenodo.py:166-173` | Low |

---

## 6. Doc Parity Actions

Concrete edits to `docs/review/DOC_PARITY_LEDGER.md`, `docs/living/CURRENT_STATE.md`,
`docs/DATA_PIPELINE.md`. Bulleted by severity.

### 6.1. Ledger claim status changes

| claim | current | suggested | reason |
|---|---|---|---|
| C041 | `verified` | `partial` | Recipe-stage count wrong (3, not 4); recipe_hash injection only fires for `COMPOSITE_RECIPES` not all `recipe_json` methods. |
| C043 | `verified` | `partial` | Wording "stage-2 snapshot" contradicts `DATA_PIPELINE.md` (parse = stage 2, snapshot = stage 3). Substance of PM/default parser support is correct. |
| C044 | `verified` | `partial` | Wording "stage-4 `analyze`" contradicts `DATA_PIPELINE.md` (analyze = stage 5). Substance of HDBSCAN phase-1 path is correct. |
| C045 | `verified` | `partial` | Same "stage-2 snapshot" wording issue as C043. |
| C048 | `verified` | `partial` | "Idempotent reuse asserted" implied for bank77 e2e; that test is single-pass structural. Per-stage idempotency is asserted in unit tests. Re-attribute. |

### 6.2. `CURRENT_STATE.md` edits

- L25 — Distinguish "registered components (4, including `pca_svd_project`)"
  from "composite recipe stages (3, no PCA)." See §1 #1.
- L60 — Replace `extra.newsgroup` / `extra.split` notation with "top-level
  keys `newsgroup` and `split` inside the parsed `extra_json` object."
- L60 — Add `semeval2013_sra_5way` to the dataset-spec bullet for parity with
  `registry.py:113-119`.
- L58 — Optionally re-anchor "stage-4" → "stage-5" if standardizing on
  `DATA_PIPELINE.md` numbering.
- L62 — Same "stage-2 snapshot" → "stage-2 parse" correction.

### 6.3. `DATA_PIPELINE.md` additions

- Add a "Filter semantics" subsection covering:
  - `category_filter` AND-across-keys, IN-within-key.
  - Strict typed equality after `json.loads`.
  - Missing key → no match.
  - Explicit `null` → matches `None` candidate.
  - Allowed value types (no numpy scalars).
- Add a "Parser contract" subsection covering:
  - Skip-vs-raise policy.
  - When to bump `parser_version`.
  - `extra_json` keys must be JSON-native types.
  - Determinism requirements (sorted iteration, no `os.walk` order reliance).

### 6.4. Suggested NEW claim entries

| id | source_doc | claim_text | evidence | suggested status |
|---|---|---|---|---|
| C051 | `docs/living/METHOD_RECIPES.md` | Composite `cosine_kllmeans_no_pca` lists 3 ordered stages (no `pca_svd_project`); the clustering component registry includes 4 first-class methods for reuse. | `algorithms/recipes.py:63-133,159-194` | `verified` |
| C052 | `docs/living/CURRENT_STATE.md` | `semeval2013_sra_5way` is registered in `ACQUIRE_REGISTRY` with default parser `parse_semeval2013_sra_5way_snapshot` and is exercised by `tests/pipeline/test_semeval_snapshot.py`. | `source_specs/registry.py:113-119`, `source_specs/semeval2013_sra_5way.py`, `tests/pipeline/test_semeval_snapshot.py` | `verified` |
| C053 | `docs/living/CURRENT_STATE.md` | HDBSCAN runner enforces `representation == "full"` (`hdbscan_runner.py:101-113`) and `run_bank77_pipeline.py` enforces `full` when `--analysis-strategy hdbscan`. | `pipeline/hdbscan_runner.py:100-113`, `scripts/run_bank77_pipeline.py:206-208` | `verified` |

### 6.5. "Open" sections in ledger

Lines 77-88 of `DOC_PARITY_LEDGER.md` say "Open High/Medium/Low: None." If
those sections track *known doc defects in `verified` rows*, they should now
list the C041/C043/C044/C045 wording defects above. If they track *only
newly-filed work items*, the section name should be clarified.

---

## 7. Open Questions for Human Judgment

These need product/architectural decision rather than code fix.

1. **`twenty_newsgroups` text-length filter** — Is `10 < len(text) <= 1000`
   the intentional benchmark contract? If yes, document it loudly. If no, fix
   and bump `parser_version`.

2. **Estela pickle trust model** — Is `pickle.load` on
   `raw.githubusercontent` bytes the documented security model for
   researchers? If yes, document. If no, replace with a parsed format
   (Parquet, JSON Lines) and bump `parser_version`.

3. **`force=True` on `snapshot()`** — Is "always create new group, even with
   identical hash triple" the intended semantics? If yes, document and
   downstream readers should pick a single `group_id`.

4. **`record_method_execution` UPDATE behavior** — Is silent overwrite of
   `fingerprint_json` for matching `(request_group_id, run_key, run_kind)`
   intended for repair, or should it be insert-only?

5. **Verifier exit codes in CI** — Should `verify_db_backup_inventory.py`
   skipping (no `AZURE_STORAGE_CONNECTION_STRING`) be exit 0 (current) or
   exit non-zero in environments that expect the check?

6. **HDBSCAN default metric** — Should default be `cosine` (typical for
   embeddings) or `euclidean` (current)? Current default is a footgun without
   `normalize_embeddings=True`.

7. **`sample_n` / `sampling_seed` non-int coercion** — Should `to_canonical_dict`
   reject non-integers, or continue silently truncating? Current behavior
   means `2.7` and `2.0` and `2` all hash identically.

8. **Per-parser variants in registry** — First-class `DatasetAcquireConfig`
   for each profile, or document the variant-override pattern? Affects C043
   and C045.

---

## 8. Audit Methodology

- 9 parallel `very thorough` read-only `explore` subagents, one per component
  area, launched in a single batch on 2026-04-22.
- Each agent received a self-contained prompt with: project context, the
  five-stage pipeline contract, recently verified claims (C037, C041,
  C043-C050), per-component focus files, allowed cross-references, and a
  fixed audit-dimension checklist.
- Agent outputs were aggregated and cross-referenced for themes only visible
  across components (§2).
- Two of the highest-impact findings (C041 recipe-stage count, SemEval
  `gold_count` type duality) were independently verified by direct file read
  before inclusion.
- No code or doc was modified during the audit.
- This document is one-shot; it is not intended to be living. Follow-up
  changes belong in PRs and `DOC_PARITY_LEDGER.md` updates.

### Components covered

| # | Component | Focus files | Lines |
|---|---|---|---|
| 1 | Pipeline snapshot | `pipeline/snapshot.py` | 357 |
| 2 | HDBSCAN runner | `pipeline/hdbscan_runner.py` | 269 |
| 3 | Foundational contracts | `pipeline/types.py`, `source_specs/parser_protocol.py` | 114 + ~50 |
| 4 | New source-spec pattern | `source_specs/registry.py`, `source_specs/twenty_newsgroups.py` | 130 + 151 |
| 5 | Existing parsers (parity) | `source_specs/{banking77,ausem,estela,semeval2013_sra_5way,sources_uncertainty_zenodo}.py` | ~800 combined |
| 6 | Sweep ingestion | `experiments/ingestion.py` | 342 |
| 7 | Backup verifier | `scripts/verify_db_backup_inventory.py` | 161 |
| 8 | Tests | `tests/pipeline/*.py`, `tests/test_datasets/*.py`, `tests/test_experiments/test_ingestion.py`, `tests/test_scripts/test_upload_jetstream_pg_dump_to_blob.py` | ~3500 combined |
| 9 | Docs parity | `docs/DATA_PIPELINE.md`, `docs/living/CURRENT_STATE.md`, `docs/review/DOC_PARITY_LEDGER.md` | 152 + 90 + 104 |
| 10 | Acquisition layer | `datasets/acquisition.py` | ~210 |

---

## 9. Cross-Critique Addendum (2026-04-22)

After the initial audit, this document was cross-critiqued by a second model
pass and then re-checked against code. This section captures the corrections
to §1-§8, findings missed on the first pass, and the merged action plan.

**Where this section conflicts with §1-§8, this section wins.**

### 9.1. Corrections to original audit findings

| location | original claim | correction | reason |
|---|---|---|---|
| §3.2, §2.5 | `_normalize_rows` may mutate caller's buffer when `embeddings` is a writable view | **Strike.** Returns a new array. | `matrix / np.maximum(norms, 1e-12)` produces a fresh ndarray; numpy doesn't mutate `matrix` in place. The other §3.2 items (no seed, default metric, no `core_dist_n_jobs` pin, mismatched-length zeroing) stand. |
| §3.3 Critical | `sample_n` / `sampling_seed` int coercion in `to_canonical_dict` | Re-bucket as **High**. | Requires non-int caller input; no current call site does this. Still worth fixing because it fails silently when it does happen. |
| §3.5 Critical | `estela.py` `pickle.load` supply-chain risk | Re-bucket as **Medium today, High the moment a new pickle source is added**. | Bytes pinned to project's own GitHub repo at a recorded commit SHA; threat surface narrow until the source list expands. |
| §6.1 C048 | proposed `verified` -> `partial` | **Keep `verified`.** | Per-stage idempotency *is* asserted in `tests/pipeline/test_parse.py`, `test_snapshot.py`, `test_embed.py`, `test_analyze.py`. The e2e file is single-pass, but the contract claim "idempotent reuse asserted across parse/snapshot/embed/analyze" holds via the union of tests. |
| §1 #1, §6.1 C041 | doc says "four component methods" | Doc is **even more off** than stated: `algorithms/recipes.py:134-155` registers a 5th component (`umap_project`); the composite recipe still has **3** stages. The right framing is *5 registered components, 3 recipe stages*. | Confirmed by direct re-read of `algorithms/recipes.py`. |
| §2.4, §3.6 | "ingestion TOCTOU race" framed inside `ingest_result_to_db` | **Reframe.** The race is in the public `run_key_exists_in_db` API surface, called *before* a long sweep from 8+ sites. The insert-time `IntegrityError` handler in `ingest_result_to_db` is a backstop, not the contract; it doesn't recover wasted sweep compute. Severity stays **High**. | See call-site evidence below. |

`run_key_exists_in_db` external callers (each runs a long sweep between the
existence check and the eventual ingest):

- `scripts/run_pca_kllmeans_sweep.py:258`
- `scripts/run_pca_kllmeans_sweep_full.py:465`
- `scripts/history/experiments/run_no_pca_50runs_sweep.py:204`
- `scripts/history/experiments/run_no_pca_multi_embedding_sweep.py:360`
- `scripts/history/experiments/run_custom_full_categories_sweep.py:233`
- `scripts/history/experiments/run_experimental_sweep.py:676`
- `src/study_query_llm/experiments/runtime_sweeps.py:451`
- `src/study_query_llm/experiments/sweep_worker_main.py:1070`

### 9.2. Findings missed on the first pass

- **Repeated O(N) artifact / group scans in 6+ places.** Same anti-pattern across:
  - `pipeline/snapshot.py:40-49` (`_collect_snapshot_artifact_uris`)
  - `pipeline/snapshot.py:140-155` (`_find_existing_snapshot_group`)
  - `pipeline/parse.py:46-59` (`_collect_acquisition_artifact_uris`)
  - `pipeline/parse.py:62-73` (`_collect_dataframe_artifact_uris`)
  - `pipeline/parse.py:177-194` (`_find_existing_dataframe_group`)
  - `pipeline/acquire.py:43-60` (`_collect_acquisition_artifact_uris`)
  - `pipeline/acquire.py:67-78` (`_find_dataset_group_by_fingerprint`)

  All of these load the full `CallArtifact` / `Group` table, then filter in
  Python by `metadata_json->>'group_id'` / `content_fingerprint`. No
  correctness bug today; pure latency tax that grows with project age. A
  JSONB expression index plus a thin helper would convert each to a point
  lookup.

### 9.3. Merged action list (canonical fix order)

This list supersedes the §4 ordering.

1. **`verify_db_backup_inventory.py` hardening.** Failure accumulator,
   non-zero exit on any mismatch / Azure failure / missing blob,
   `encoding="utf-8"` on `load_dotenv`, minimum tests for mismatch + error
   paths. **High.** *(Only finding with current CI consequence — placed first.)*
2. **Docs parity sweep.** C041 (5 registered components / 3 recipe stages),
   C043 / C044 / C045 stage-numbering wording, `METHOD_RECIPES.md:88`
   component list, `CURRENT_STATE.md:25` and `:60` wording, `extra.<key>` ->
   "top-level key in parsed `extra_json`" notation. **Medium-High.**
3. **`category_filter` contract.** Document semantics in `DATA_PIPELINE.md`
   ("Filter semantics" subsection): missing-key vs explicit-null, strict
   typed equality after `json.loads`, allowed value types, AND-across-keys /
   IN-within-key. Add 4 regression tests covering each silent-misfire path.
   **Medium-High.**
4. **SemEval `gold_count` normalization.** Emit one stable type (preferred:
   always `int`; fallback: split into `gold_count_int` + `gold_count_raw`);
   bump `parser_version`. **Medium.**
5. **JSONB metadata `group_id` index + helper.** Replace the ~6 O(N) scans
   listed in §9.2 with point lookups. **Medium.**
6. **Fingerprint input forwarding.** Pass `manifest_hash` / `data_regime`
   from sweep metadata into `record_method_execution`; make `recipe_hash`
   injection uniform across all `recipe_json`-bearing methods, not just
   members of `COMPOSITE_RECIPES`. **Medium.**
7. **Ingestion idempotency reshape.** Deprecate standalone
   `run_key_exists_in_db` in favor of an atomic claim-or-skip call (single
   transaction with the unique-index-backed insert), or document the TOCTOU
   contract loudly and audit the 8 existing callers. **Medium.**
8. **HDBSCAN deterministic-defaults policy.** Decide cosine vs euclidean
   default, seed pin, `core_dist_n_jobs` policy. **Low-Medium — product
   call needed.**

---

*Audit artifact. Not a living document. Action items belong in PRs and ledger updates.*
