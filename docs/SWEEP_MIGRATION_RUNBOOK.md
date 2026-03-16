# Sweep Migration Runbook

This runbook performs targeted migration and hardening for request-driven sweep execution.

## Phase 2 Preflight Checklist (Blob Ops Hardening)

Complete this checklist before any Phase 2 behavior changes:

- Confirm [docs/PHASE1_5_VERIFICATION.md](docs/PHASE1_5_VERIFICATION.md) is green.
- Confirm lane and auth policy:
  - `ARTIFACT_RUNTIME_ENV` is explicitly set (`dev`, `stage`, or `prod`).
  - `ARTIFACT_AUTH_MODE` matches lane policy (`connection_string` for local dev, `managed_identity` for hosted lanes).
  - Container routing resolves correctly for the lane (`AZURE_STORAGE_CONTAINER_<LANE>` preferred).
- Run storage health check:
  - `python scripts/check_azure_blob_storage.py`
- Baseline test gates before edits:
  - `pytest tests/test_services/test_artifact_service.py -v`
  - `pytest tests/test_storage/test_azure_blob.py -v`
  - `pytest tests/test_scripts/test_ingest_sweep_to_db.py -v`
- Record current invariants:
  - Artifact URIs in blob mode are HTTPS blob URLs.
  - `run_key` idempotency remains intact (no duplicate `clustering_run`).
  - URI/backend compatibility checks pass in ingestion flows.

## 1) Audit (read-only)

```bash
python scripts/audit_last_partial_sweep.py
python scripts/check_run_groups.py --summary-only
python scripts/check_sweep_requests.py --select-last-partial --expected-min 100 --completed-min 1 --completed-max 25
```

Confirm:
- no duplicate `clustering_run` `run_key`
- no duplicate `group_links` triplets
- one clear target partial request candidate

## 2) Additive DB migrations

```bash
python -m study_query_llm.db.migrations.add_sweep_request_indexes
python -m study_query_llm.db.migrations.add_sweep_worker_safety
```

`add_sweep_worker_safety` fails fast when duplicate data would violate uniqueness.

## 3) Reconcile single partial request

Dry-run:

```bash
python scripts/reconcile_last_partial_sweep.py --dry-run --expected-min 100 --completed-min 1 --completed-max 25
```

Apply:

```bash
python scripts/reconcile_last_partial_sweep.py --expected-min 100 --completed-min 1 --completed-max 25
```

If some runs exist only as PKLs:

```bash
python scripts/reconcile_last_partial_sweep.py --ingest-artifacts --artifacts-dir experimental_results --expected-min 100 --completed-min 1 --completed-max 25
```

## 4) Verify request status

```bash
python scripts/check_sweep_requests.py --request-id <REQUEST_ID>
```

Success condition:
- `missing=0`
- request marked fulfilled
- linked `clustering_sweep` created

## 5) Controlled worker rollout

Single worker first:

```bash
python scripts/run_300_bigrun_sweep.py --request-id <REQUEST_ID> --worker-id worker-1 --claim-lease-seconds 3600
```

Scale out only after stability:

```bash
python scripts/run_300_bigrun_sweep.py --request-id <REQUEST_ID> --worker-id worker-2 --claim-lease-seconds 3600
```

Start with one worker and monitor:
- duplicate run_key errors (should be none)
- duplicate link errors (should be none)
- stable decrease in `missing_count`

Scale worker count only after one-worker execution is stable.

## 6) Local 300 (one container per engine)

Use this mode for `local_300_2datasets` when running local TEI models on desktop GPU.

**Mode selection:** The `--job-mode` flag (`standalone` or `sharded`) selects a strategy/factory-based orchestrator. Both modes preserve the same job semantics; `standalone` uses run-key claims via `sweep_run_claims`, while `sharded` uses the `orchestration_jobs` path with reducer/finalize flow. No behavior change from prior branching—only internal delegation to explicit orchestrator objects.

Stage A (1 worker):

```bash
python scripts/run_local_300_2datasets_engine_supervisor.py --request-id <REQUEST_ID> --workers 1
```

Stage B (3 workers):

```bash
python scripts/run_local_300_2datasets_engine_supervisor.py --request-id <REQUEST_ID> --workers 3
```

Stage C (up to 10 workers):

```bash
python scripts/run_local_300_2datasets_engine_supervisor.py --request-id <REQUEST_ID> --workers 10
```

Safety knobs:
- `--engine-allowlist` to run a subset of embedding engines
- `--max-worker-restarts` and `--max-tei-restarts` to bound retry loops
- `--idle-exit-seconds` to let idle workers exit cleanly
- `--progress-poll-seconds` to reduce DB polling frequency on busy machines

## 7) Sharded job-table mode (K/try sharding)

Use sharded mode when you need finer-grained parallel work units.

- Request metadata:
  - `execution_mode = "sharded"`
  - `shard_config = {"k_ranges": [[2, 5], [6, 10], [11, 20]], "tries_per_k": 3}`
- Job types:
  - `run_k_try` (leaf)
  - `reduce_k` (per-K reducer)
  - `finalize_run` (canonical run reducer)

Run supervisor in sharded mode:

```bash
python scripts/run_local_300_2datasets_engine_supervisor.py --request-id <REQUEST_ID> --workers 3 --job-mode sharded
```

Inspect jobs:

```bash
python scripts/check_orchestration_jobs.py --request-id <REQUEST_ID>
```

## 8) Job runner architecture and langgraph_run

**Architecture boundary:** The DB `orchestration_jobs` table is the outer control plane (claim/lease/complete/fail). Job runners (`JobRunnerFactory` by `job_type`) execute the work inside a claimed job. LangGraph is an in-job workflow runtime—one DB job = one LangGraph run. LangGraph handles internal branching/parallelism; the DB handles scheduling and durability.

- **Job types:** `run_k_try`, `reduce_k`, `finalize_run` (sweep), `langgraph_run` (agentic).
- **Workers:** Sweep workers use `run_local_300_2datasets_worker.py`; LangGraph jobs use `run_langgraph_job_worker.py`.

Run LangGraph worker:

```bash
python scripts/run_langgraph_job_worker.py --request-id <REQUEST_ID> --worker-id lg-worker-1 --idle-exit-seconds 60
```

Enqueue `langgraph_run` jobs via `RawCallRepository.enqueue_orchestration_job` with `job_type="langgraph_run"` and `payload_json={"prompt": "..."}`.

## Troubleshooting (Phase 2)

### Auth failures

Symptoms:
- `AuthenticationFailed`, `AuthorizationPermissionMismatch`, or 401/403

Actions:
- Verify `ARTIFACT_AUTH_MODE` aligns with lane policy.
- For `connection_string`, confirm `AZURE_STORAGE_CONNECTION_STRING` is valid.
- For `managed_identity`, confirm `AZURE_STORAGE_ACCOUNT_URL` and role assignment.
- In hosted lanes, do not use local fallback; treat as blocking configuration issue.

### Missing blob / URI mismatch

Symptoms:
- `Artifact not found`
- URI/backend mismatch errors

Actions:
- Verify `ARTIFACT_STORAGE_BACKEND` and lane-specific container settings.
- Confirm URI container matches resolved container and prefix policy.
- Re-run `python scripts/check_azure_blob_storage.py`.

### Checksum/size mismatch

Symptoms:
- Artifact byte size mismatch
- Artifact checksum mismatch

Actions:
- Re-read artifact once; if mismatch persists, quarantine artifact and re-run producer.
- Verify no manual blob mutation happened post-write.
- Record incident with run ID, artifact ID, URI, expected vs actual values.

### Throttling / transient network failures

Symptoms:
- intermittent blob I/O failures, timeout/reset errors

Actions:
- Confirm retry knobs:
  - `AZURE_STORAGE_MAX_RETRIES`
  - `AZURE_STORAGE_RETRY_BACKOFF_SECONDS`
- Reduce parallel load and re-run.
- Escalate if repeated failures continue after bounded retries.
