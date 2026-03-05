# Sweep Migration Runbook

This runbook performs targeted migration and hardening for request-driven sweep execution.

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
