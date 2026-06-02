# Scheduling vs Provenance Boundary

> **Living document** — update when execution models or provenance rules change.

## Purpose

This document defines when a provenance stage candidate in a method or analysis pipeline should
be a **schedulable unit** (orchestration job) versus an **in-job provenance
event** (artifact, structured result, `provenanced_runs` row).

The distinction matters because:

- Orchestration jobs carry lease, retry, and dependency overhead.
- Provenance events carry audit/lineage value but not scheduling cost.
- The two should be decoupled so that changing job granularity does not change
  what the system records about *what happened to the data*.

## Execution Vocabulary

Use these terms consistently in prose:

- **`provenance_stage`**: lineage node within run/request provenance.
- **`algorithm_iteration`**: one inner-loop update cycle in an iterative method.
- **`restart_try`**: one seeded restart/try for fixed run configuration.
- **`orchestration_job`**: schedulable/leased job-table unit.
- **`planning_step`**: roadmap milestone such as `STEP-*`.

Keep literal identifiers unchanged when quoting code/schema names (for example
`step_name`, `step_type`, `clustering_step`).

## Schedulable Unit (Orchestration Job)

Use a separate orchestration job when the candidate work unit needs:

- **Its own lease**: long-running work that may time out and need re-claim.
- **A retry boundary**: failure of this orchestration job should not abort sibling jobs.
- **Multi-worker isolation**: the work unit may run on a different machine or process.
- **An explicit DAG dependency edge**: downstream jobs must wait for this job.

**Examples**: embedding a full dataset, running a clustering sweep across K
values, calling an LLM API with rate limits, executing an MCQ probe.

## In-Job Provenance Event

Use an in-job provenance event (artifact, Group provenance stage, `provenanced_runs` row,
`analysis_results` entry) when:

- The work item is computationally trivial relative to the job it belongs to.
- It does not need its own lease or retry boundary.
- Failure can be handled within the enclosing job's error path.

**Examples**: computing a metric from already-present data, validating a
manifest, recording a config hash, writing an analysis result, PCA projection
of a small matrix.

## Fingerprint Independence Rule

The **canonical run fingerprint** (`fingerprint_json` / `fingerprint_hash` on
`provenanced_runs`) must be identical regardless of whether provenance stages
are separate orchestration jobs or in-job provenance events.

If changing granularity changes the fingerprint, either:

1. The fingerprint includes scheduling-only fields (fix the fingerprint), or
2. The granularity boundary splits or merges *semantically different* work (fix
   the boundary).

## Owner of Granularity Decisions

The **planner / enqueue site** (e.g. `SweepRequestService`, request config)
decides how to partition work into jobs. Workers execute whatever shape they
receive.

Current control-plane seam details:

- Sweep-type adapters emit deterministic orchestration graph specs (job nodes + dependency edges).
- Job execution dispatch is registry-based by `job_type`.
- Reducer/finalizer execution is routed through a typed reducer plugin seam.
- Clustering analysis jobs (when enabled) follow a producer/consumer contract: planner emits per-run `analysis_run` nodes keyed by request+run+analysis; dependency gating is either per-run `finalize_run` (default) or a single request-level `finalize_request` node when `SQ_USE_REQUEST_FINALIZER_JOB=1`. Workers late-bind analyze inputs from request-delivered `clustering_run` lineage metadata (`dataset_snapshot_ids`, `embedding_batch_group_id`) by `(request_id, run_key)`.
- In `SQ_USE_REQUEST_FINALIZER_JOB=1` mode, per-run `finalize_run` jobs materialize run facts + request delivery links only; one request-level `finalize_request` job is the single writer for request fulfillment/sweep linkage decisions.

Neither the DB schema nor the fingerprint encode or depend on the job graph
shape (fan-out, batch size, number of jobs).

## Overhead Diagnosis

When orchestration claim/complete overhead dominates a job's wall-clock time,
the remedy is to adjust granularity at the planner level (coarser jobs, larger
batches) — not to remove provenance. Timing instrumentation on claim and
complete paths (see `raw_call_repository.py` and `sweep_worker_main.py`) makes
this visible in logs.

## Reducer Aggregates vs Leaf Try Payloads

`reduce_k` is intentionally **aggregating**: it selects best objective labels/metadata across sibling leaf shards while still emitting audit-grade summaries (`objectives`, `labels_all`). After this update it also preserves **every** leaf shard’s structured try row under `by_k[*].tries` (including profiling markers such as `try_idx`, `seed_value`, and the full `k_payload` blob). Leaf shards are therefore expected to carry exactly **one** `by_k` bucket each; multi-`k` leaf payloads violate reducer assumptions and raise `RuntimeError`.

## Request-Group Identity Uniqueness

A one-off analysis execution is grouped under an `analysis_request` Group keyed
by the identity `(method_name, input_id, run_key)` carried in
`groups.metadata_json`. This identity is a **DB-enforced uniqueness invariant**:
concurrent `analyze()` callers that resolve the same identity converge on a
single `analysis_request` group rather than each creating a duplicate.

- The guard is a partial UNIQUE *functional* index
  (`uq_groups_analysis_request_identity`) over the JSON-extracted identity
  fields, scoped `WHERE group_type = 'analysis_request'`. JSON extraction is
  dialect-specific, so the index DDL is emitted per dialect from `after_create`
  events in `db/models_v2.py`; `init_db()`/`create_all()` therefore builds it on
  both SQLite (tests) and Postgres (fresh installs). The already-provisioned
  canonical Postgres database receives it via
  `db/migrations/add_analysis_request_unique_index.py`.
- `ProvenanceService.create_analysis_request_group` delegates to
  `RawCallRepository.insert_or_get_analysis_request_group`, which performs a
  conflict-safe insert/reselect (fast-path lookup, then a SAVEPOINT-guarded
  insert that recovers from a unique-index collision by re-selecting the
  winner). Get-or-create is thus race-free at the database layer, not merely
  inside the per-run in-process lock — that lock is acquired *after*
  request-group resolution in `pipeline/analyze.py` and so cannot serialize it.

Identity uniqueness is a *provenance* invariant, independent of scheduling: it
holds whether the analysis runs as a standalone orchestration job or an in-job
provenance event, and does not depend on job-graph shape. Remediating
pre-existing duplicates (keep the lowest group id; repoint every reference onto
it) is a one-off canonical write performed by
`scripts/remediate_analysis_request_duplicates.py` before the index is added.

## See Also

- [STANDING_ORDERS.md](../STANDING_ORDERS.md) — Method Definitions and Provenance conventions
- [ARCHITECTURE_CURRENT.md](ARCHITECTURE_CURRENT.md) — Orchestration and Provenance Notes
- `provenanced_run_service.py` — `canonical_run_fingerprint`, `fingerprints_match`
