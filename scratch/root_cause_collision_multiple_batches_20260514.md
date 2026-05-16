# Root Cause Report: `collision_multiple_batches` on Per-Pair Backfill

Date: 2026-05-14  
Scope: investigate why the per-pair backfill dry-run marks 63/120 pairs as `collision_multiple_batches` for snapshots `6/9/10`.

## Data Sources and Method

- Dry-run artifacts:
  - `experimental_results/backfill_per_pair/20260514T085641Z/infeasible_pairs.json`
  - `experimental_results/backfill_per_pair/20260514T085641Z/pair_classification.json`
- Prior coverage baseline (showing `92/95/89` cells):
  - `scratch/exports/full_coverage_20260512T235751Z/coverage_summary.md`
- Read-only DB probe output (this investigation):
  - `scratch/exports/collision_probe_20260514T191040Z/collision_probe.json`
- SQL scope:
  1. Pull snapshot lineage (`source_dataframe_group_id`, `source_dataframe_row_count`, `snapshot_row_count`).
  2. Pull all `groups` rows where `group_type='embedding_batch'` and metadata matches backfill lineage filter:
     - `source_dataframe_group_id`
     - `entry_max` (`source_dataframe_row_count`)
     - `embedding_engine`
     - `provider` (pair provider)
  3. Pull prior non-backfill `provenanced_runs` analysis rows for each sampled pair (legacy methods: `hdbscan`, `kmeans+silhouette+kneedle`, `gmm+bic+argmin`) and resolve which candidate batch ID those runs actually used.

## A/B) Sampled Collision Pairs (3 per snapshot) with Full Batch Evidence

Selection rule: deterministic fixed engines across each research snapshot (`baai/bge-base-en-v1.5`, `openai/text-embedding-3-large`, `perplexity/pplx-embed-v1-0.6b`).

---

### Snapshot 6 (`sources_uncertainty_qc`)

#### Pair: `snap6__baai_bge-base-en-v1.5`

- Snapshot lineage:
  - `source_dataframe_group_id=5`
  - `source_dataframe_row_count=2086`
  - `snapshot_row_count=2086`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 125 | 2026-04-25T15:39:44.627353+00:00 | 2086 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:5:full | null | raw_v1 |
| 345 | 2026-04-25T20:09:55.779151+00:00 | 2086 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:5:full | null | raw_v1 |

Prior `92/95/89` coverage run usage (legacy analysis methods):

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 155 | 2026-04-26T09:14:45.227384+00:00 | hdbscan | 345 |
| 255 | 2026-04-26T09:37:10.556523+00:00 | kmeans+silhouette+kneedle | 345 |
| 355 | 2026-04-26T10:04:25.130153+00:00 | gmm+bic+argmin | 345 |

---

#### Pair: `snap6__openai_text-embedding-3-large`

- Snapshot lineage:
  - `source_dataframe_group_id=5`
  - `source_dataframe_row_count=2086`
  - `snapshot_row_count=2086`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 103 | 2026-04-25T15:22:07.348464+00:00 | 2086 | openai/text-embedding-3-large | openrouter | full | dataframe:5:full | null | raw_v1 |
| 353 | 2026-04-25T20:10:48.922969+00:00 | 2086 | openai/text-embedding-3-large | openrouter | full | dataframe:5:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 162 | 2026-04-26T09:16:20.699530+00:00 | hdbscan | 353 |
| 262 | 2026-04-26T09:39:00.545038+00:00 | kmeans+silhouette+kneedle | 353 |
| 362 | 2026-04-26T10:09:50.055789+00:00 | gmm+bic+argmin | 353 |

---

#### Pair: `snap6__perplexity_pplx-embed-v1-0.6b`

- Snapshot lineage:
  - `source_dataframe_group_id=5`
  - `source_dataframe_row_count=2086`
  - `snapshot_row_count=2086`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 109 | 2026-04-25T15:31:56.005763+00:00 | 2086 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:5:full | null | raw_v1 |
| 337 | 2026-04-25T20:09:08.800428+00:00 | 2086 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:5:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 165 | 2026-04-26T09:17:08.600177+00:00 | hdbscan | 337 |
| 265 | 2026-04-26T09:39:48.279670+00:00 | kmeans+silhouette+kneedle | 337 |
| 365 | 2026-04-26T10:12:06.926844+00:00 | gmm+bic+argmin | 337 |

---

### Snapshot 9 (`estela`)

#### Pair: `snap9__baai_bge-base-en-v1.5`

- Snapshot lineage:
  - `source_dataframe_group_id=8`
  - `source_dataframe_row_count=324`
  - `snapshot_row_count=324`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 58 | 2026-04-25T07:54:29.105325+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 88 | 2026-04-25T14:29:39.284380+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 244 | 2026-04-25T18:58:30.367692+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 303 | 2026-04-25T20:05:44.004533+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 324 | 2026-04-25T20:07:40.704914+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 175 | 2026-04-26T09:19:24.894287+00:00 | hdbscan | 324 |
| 275 | 2026-04-26T09:42:20.081317+00:00 | kmeans+silhouette+kneedle | 324 |
| 375 | 2026-04-26T10:18:55.318679+00:00 | gmm+bic+argmin | 324 |

---

#### Pair: `snap9__openai_text-embedding-3-large`

- Snapshot lineage:
  - `source_dataframe_group_id=8`
  - `source_dataframe_row_count=324`
  - `snapshot_row_count=324`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 2026-04-25T07:14:55.337937+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 96 | 2026-04-25T14:29:57.980580+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 253 | 2026-04-25T18:58:44.168189+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 311 | 2026-04-25T20:06:06.507753+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 332 | 2026-04-25T20:08:00.207602+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 182 | 2026-04-26T09:20:30.649586+00:00 | hdbscan | 332 |
| 282 | 2026-04-26T09:43:44.802283+00:00 | kmeans+silhouette+kneedle | 332 |
| 382 | 2026-04-26T10:20:48.523800+00:00 | gmm+bic+argmin | 332 |

---

#### Pair: `snap9__perplexity_pplx-embed-v1-0.6b`

- Snapshot lineage:
  - `source_dataframe_group_id=8`
  - `source_dataframe_row_count=324`
  - `snapshot_row_count=324`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | 2026-04-25T07:39:10.190761+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 79 | 2026-04-25T14:29:24.969116+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 237 | 2026-04-25T18:58:17.227655+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 294 | 2026-04-25T20:05:27.465726+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 316 | 2026-04-25T20:07:25.322157+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 185 | 2026-04-26T09:20:59.032119+00:00 | hdbscan | 316 |
| 285 | 2026-04-26T09:44:20.866726+00:00 | kmeans+silhouette+kneedle | 316 |
| 385 | 2026-04-26T10:21:36.541909+00:00 | gmm+bic+argmin | 316 |

---

### Snapshot 10 (`estela` research subset)

#### Pair: `snap10__baai_bge-base-en-v1.5`

- Snapshot lineage:
  - `source_dataframe_group_id=8`
  - `source_dataframe_row_count=324`
  - `snapshot_row_count=286`

Matching `embedding_batch` rows (same candidates as snapshot 9):

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 58 | 2026-04-25T07:54:29.105325+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 88 | 2026-04-25T14:29:39.284380+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 244 | 2026-04-25T18:58:30.367692+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 303 | 2026-04-25T20:05:44.004533+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |
| 324 | 2026-04-25T20:07:40.704914+00:00 | 324 | baai/bge-base-en-v1.5 | openrouter | full | dataframe:8:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 195 | 2026-04-26T09:22:32.956166+00:00 | hdbscan | 324 |
| 295 | 2026-04-26T09:46:21.766406+00:00 | kmeans+silhouette+kneedle | 324 |
| 395 | 2026-04-26T10:24:17.174983+00:00 | gmm+bic+argmin | 324 |

---

#### Pair: `snap10__openai_text-embedding-3-large`

- Snapshot lineage:
  - `source_dataframe_group_id=8`
  - `source_dataframe_row_count=324`
  - `snapshot_row_count=286`

Matching `embedding_batch` rows (same candidates as snapshot 9):

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 2026-04-25T07:14:55.337937+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 96 | 2026-04-25T14:29:57.980580+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 253 | 2026-04-25T18:58:44.168189+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 311 | 2026-04-25T20:06:06.507753+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |
| 332 | 2026-04-25T20:08:00.207602+00:00 | 324 | openai/text-embedding-3-large | openrouter | full | dataframe:8:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 202 | 2026-04-26T09:23:38.616529+00:00 | hdbscan | 332 |
| 302 | 2026-04-26T09:48:10.284220+00:00 | kmeans+silhouette+kneedle | 332 |
| 402 | 2026-04-26T10:26:09.873999+00:00 | gmm+bic+argmin | 332 |

---

#### Pair: `snap10__perplexity_pplx-embed-v1-0.6b`

- Snapshot lineage:
  - `source_dataframe_group_id=8`
  - `source_dataframe_row_count=324`
  - `snapshot_row_count=286`

Matching `embedding_batch` rows (same candidates as snapshot 9):

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | 2026-04-25T07:39:10.190761+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 79 | 2026-04-25T14:29:24.969116+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 237 | 2026-04-25T18:58:17.227655+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 294 | 2026-04-25T20:05:27.465726+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |
| 316 | 2026-04-25T20:07:25.322157+00:00 | 324 | perplexity/pplx-embed-v1-0.6b | openrouter | full | dataframe:8:full | null | raw_v1 |

Prior `92/95/89` coverage run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 205 | 2026-04-26T09:24:06.792692+00:00 | hdbscan | 316 |
| 305 | 2026-04-26T09:48:54.646861+00:00 | kmeans+silhouette+kneedle | 316 |
| 405 | 2026-04-26T10:26:57.513349+00:00 | gmm+bic+argmin | 316 |

## C) Comparison Case (Non-Collision): Snapshot 17

Pair: `snap17__openai_text-embedding-3-large` (`classification=clustering-ready`, `manifest_status=ok`)

- Snapshot lineage:
  - `source_dataframe_group_id=16`
  - `source_dataframe_row_count=13069`
  - `snapshot_row_count=1160`

Matching `embedding_batch` rows:

| id | created_at | entry_max | embedding_engine | provider | representation | dataset_key | dimension | key_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 760 | 2026-04-26T06:28:07.083324+00:00 | 13069 | openai/text-embedding-3-large | openrouter | full | dataframe:16:full | null | raw_v1 |

Prior run usage:

| provenanced_run_id | created_at | method_name | used_batch_id |
| --- | --- | --- | --- |
| 222 | 2026-04-26T09:27:52.708526+00:00 | hdbscan | 760 |
| 322 | 2026-04-26T09:53:32.978917+00:00 | kmeans+silhouette+kneedle | 760 |
| 422 | 2026-04-26T10:32:50.582386+00:00 | gmm+bic+argmin | 760 |

Structural difference vs research snapshots:

- Research snapshots (`6/9/10`) map to source dataframes that have duplicate same-key embedding batches (2x on dataframe 5; 5x on dataframe 8), so lineage + engine resolves to multiple candidates.
- Banking snapshot (`17`) has exactly one candidate batch for this engine on its source dataframe (`16`), so no collision.

## D) Root-Cause Bucketing

Bucket definitions from request:

- (a) re-embed with same key
- (b) provider drift
- (c) representation drift
- (d) dimension drift
- (e) stale lineage numeric collision
- (f) other

Observed bucket assignment:

- **All sampled collisions** are **(a) re-embed with same key**:
  - same `source_dataframe_group_id`
  - same `entry_max`
  - same `embedding_engine`
  - same `provider`
  - same `representation`
  - same `dataset_key`
  - same `key_version`
  - no dimension split observed (`dimension=null` throughout sampled collisions)

No evidence of:

- provider drift (`b`)
- representation drift (`c`)
- dimension drift (`d`)
- stale lineage numeric collision (`e`)

## E) Tally Across All 63 Infeasible Pairs

Method:

- Classify every `reason_code=collision_multiple_batches` pair from `infeasible_pairs.json` using DB batch metadata under the same lineage filter.
- Cross-check prior non-backfill legacy analysis runs to see which candidate batch had historically been used.

Results:

- Total collisions: `63`
- By snapshot:
  - snapshot `6`: `21`
  - snapshot `9`: `21`
  - snapshot `10`: `21`
- Bucket split:
  - `(a) re-embed with same key`: `63/63` (**100%**)
  - `(b/c/d/e/f)`: `0/63`

Legacy-run usage profile over all 63 collision pairs:

- `60/63` pairs had prior legacy analysis runs using exactly one candidate batch, and that used batch was the **latest candidate** (`max(batch_id)`).
- `3/63` pairs had **no prior legacy runs** (all three are `qwen/qwen3-embedding-4b` on snapshots `6/9/10`).

## Root Cause (Final)

The new per-pair orchestrator is exposing pre-existing **same-key duplicate `embedding_batch` rows** on research dataframes (`5` and `8`) for openrouter engines. The collision detector is behaving correctly: lineage + engine no longer identifies a unique batch for these pairs. Prior `92/95/89` runs succeeded because those executions happened to pick one concrete batch (typically the latest inserted one), but the current orchestrator intentionally refuses to guess when multiple equally-matching batches exist.
