# Scheduling vs Provenance Boundary

> **Living document** — update when execution models or provenance rules change.

## Purpose

This document defines when a sub-step in a method or analysis pipeline should
be a **schedulable unit** (orchestration job) versus an **in-job provenance
event** (artifact, structured result, `provenanced_runs` row).

The distinction matters because:

- Orchestration jobs carry lease, retry, and dependency overhead.
- Provenance events carry audit/lineage value but not scheduling cost.
- The two should be decoupled so that changing job granularity does not change
  what the system records about *what happened to the data*.

## Schedulable Unit (Orchestration Job)

Use a separate orchestration job when the sub-step needs:

- **Its own lease**: long-running work that may time out and need re-claim.
- **A retry boundary**: failure of this step should not abort sibling steps.
- **Multi-worker isolation**: the step may run on a different machine or process.
- **An explicit DAG dependency edge**: downstream steps must wait for this step.

**Examples**: embedding a full dataset, running a clustering sweep across K
values, calling an LLM API with rate limits, executing an MCQ probe.

## In-Job Provenance Event

Use an in-job provenance event (artifact, Group step, `provenanced_runs` row,
`analysis_results` entry) when:

- The step is computationally trivial relative to the job it belongs to.
- It does not need its own lease or retry boundary.
- Failure can be handled within the enclosing job's error path.

**Examples**: computing a metric from already-present data, validating a
manifest, recording a config hash, writing an analysis result, PCA projection
of a small matrix.

## Fingerprint Independence Rule

The **canonical run fingerprint** (`fingerprint_json` / `fingerprint_hash` on
`provenanced_runs`) must be identical regardless of whether sub-steps are
separate orchestration jobs or in-job provenance events.

If changing granularity changes the fingerprint, either:

1. The fingerprint includes scheduling-only fields (fix the fingerprint), or
2. The granularity boundary splits or merges *semantically different* work (fix
   the boundary).

## Owner of Granularity Decisions

The **planner / enqueue site** (e.g. `SweepRequestService`, request config)
decides how to partition work into jobs. Workers execute whatever shape they
receive.

Neither the DB schema nor the fingerprint encode or depend on the job graph
shape (fan-out, batch size, number of jobs).

## Overhead Diagnosis

When orchestration claim/complete overhead dominates a job's wall-clock time,
the remedy is to adjust granularity at the planner level (coarser jobs, larger
batches) — not to remove provenance. Timing instrumentation on claim and
complete paths (see `raw_call_repository.py` and `sweep_worker_main.py`) makes
this visible in logs.

## See Also

- [STANDING_ORDERS.md](../STANDING_ORDERS.md) — Method Definitions and Provenance conventions
- [ARCHITECTURE_CURRENT.md](ARCHITECTURE_CURRENT.md) — Orchestration and Provenance Notes
- `provenanced_run_service.py` — `canonical_run_fingerprint`, `fingerprints_match`
