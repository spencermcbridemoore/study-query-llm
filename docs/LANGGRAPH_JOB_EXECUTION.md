# LangGraph Job Execution

## Architecture boundary

- **DB `orchestration_jobs` table:** Outer control plane. Handles claim, lease, complete, fail. One row = one unit of work.
- **LangGraph:** In-job workflow runtime. One claimed DB job = one LangGraph graph invocation. LangGraph manages internal branching, cycles, and parallelism; the DB does not.

## Flow

1. Enqueue `langgraph_run` jobs via `RawCallRepository.enqueue_orchestration_job`.
2. Worker (`run_langgraph_job_worker.py`) claims jobs, runs them via `LangGraphJobRunner`, completes or fails.
3. Result artifact path is stored in `result_ref`.

## Payload

`payload_json` for `langgraph_run`:

- `prompt` (str): Input to the minimal echo graph (extensible for custom graphs).
- `config` (dict): Optional config (extensible).

## Running the worker

```bash
python scripts/run_langgraph_job_worker.py --request-id <REQUEST_ID> --worker-id lg-worker-1 --idle-exit-seconds 60
```

## Extending

To add custom LangGraph graphs, extend `LangGraphJobRunner` or add new runner types in `job_runner_factory.py`. Keep the boundary: DB orchestrates jobs; LangGraph orchestrates steps inside a job.
