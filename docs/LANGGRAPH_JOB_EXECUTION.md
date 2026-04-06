# LangGraph Job Execution

## Architecture boundary

- **DB `orchestration_jobs` table:** Outer control plane. Handles claim, lease, complete, fail. One row = one unit of work.
- **LangGraph:** In-job workflow runtime. One claimed DB job = one LangGraph graph invocation. LangGraph manages internal branching, cycles, and parallelism; the DB does not.

## Flow

1. Enqueue `langgraph_run` jobs via `RawCallRepository.enqueue_orchestration_job`.
2. Worker claims jobs, runs them via `LangGraphJobRunner`, completes or fails. Implementation: `study_query_llm.services.jobs.runtime_workers` (CLI: `python -m study_query_llm.cli jobs langgraph-worker`; compatibility script: `scripts/run_langgraph_job_worker.py`).
3. Result artifact path is stored in `result_ref`.
4. Method-level provenance is recorded in `analysis_results` for both success and failure (see [Method provenance](#method-provenance)).

## Payload

`payload_json` for `langgraph_run`:

- `prompt` (str): Input to the minimal echo graph (extensible for custom graphs).
- `config` (dict): Optional config (extensible). Supports:
  - `method_name`, `method_version`: Override default method identity for provenance.
  - `checkpoint`: When truthy, enables optional checkpoint reference metadata capture (see [Checkpoint references](#checkpoint-references)).

## Method provenance

The worker records standardized `analysis_results` rows for every `langgraph_run` job outcome (success or failure). This is **best-effort**: provenance write failures log a warning and do not affect job completion or failure status transitions.

### Method identity

- **Default:** `langgraph_run.<task_name_or_job_key_prefix>` (fallback `langgraph_run.default`), version `"1"`.
- **Override:** Set `config.method_name` and optionally `config.method_version` in the payload.

### Result envelope

Each recorded row uses a standard `result_json` envelope:

| Field | Description |
|-------|-------------|
| `status` | `"completed"` or `"failed"` |
| `job_id` | Orchestration job ID |
| `job_key` | Job key |
| `result_ref` | Artifact path (success) or null |
| `error` | Error message (failure only) |
| `parameters` | Redacted payload parameters (sensitive keys masked) |
| `method` | `{ "name": "...", "version": "..." }` |
| `recorded_at` | ISO timestamp |
| `checkpoint_refs` | Optional; present when checkpoint capture is enabled |

### Parameter redaction

Before persistence, sensitive parameter keys are redacted (replaced with `"[REDACTED]"`): `api_key`, `token`, `secret`, `password`, `authorization`, `bearer`.

### Source group

`source_group_id` in `analysis_results` is set from `request_group_id` when available (from the job payload or claim context).

## Checkpoint references

When `payload.config.checkpoint` is truthy, the runner can capture lightweight checkpoint metadata without storing full history:

- **`thread_id`:** From `config.thread_id` or derived as `job_{job_id}`.
- **`checkpoint_id`:** Latest checkpoint ID from the run (when using `InMemorySaver`).

These appear in `result_json.checkpoint_refs` and in the output artifact metadata. Checkpoint capture is opt-in; when disabled, no checkpointer is used.

## Running the worker

```bash
python -m study_query_llm.cli jobs langgraph-worker --request-id <REQUEST_ID> --worker-id lg-worker-1 --idle-exit-seconds 60
```

Equivalent (thin wrapper):

```bash
python scripts/run_langgraph_job_worker.py --request-id <REQUEST_ID> --worker-id lg-worker-1 --idle-exit-seconds 60
```

## Extending

To add custom LangGraph graphs, extend `LangGraphJobRunner` or add new runner types in `jobs/job_runner_factory.py`. Keep the boundary: DB orchestrates jobs; LangGraph orchestrates steps inside a job.
