"""LangGraph job runner for langgraph_run orchestration jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

from .job_payload_models import parse_job_snapshot, parse_langgraph_run_payload
from .job_runners import JobRunContext, JobRunOutcome


def _run_minimal_graph(prompt: str) -> Dict[str, Any]:
    """Run a minimal LangGraph echo graph. Returns state dict."""
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class GraphState(TypedDict):
        input: str
        output: str

    def echo_node(state: GraphState) -> Dict[str, str]:
        return {"output": state.get("input", "") or "(empty)"}

    workflow = StateGraph(GraphState)
    workflow.add_node("echo", echo_node)
    workflow.add_edge(START, "echo")
    workflow.add_edge("echo", END)
    compiled = workflow.compile()
    result = compiled.invoke({"input": prompt, "output": ""})
    return dict(result)


class LangGraphJobRunner:
    """Runner for langgraph_run jobs. Executes a minimal graph and saves result."""

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        try:
            parse_job_snapshot(job_snapshot)
            payload = parse_langgraph_run_payload(job_snapshot.get("payload_json") or {})
        except ValidationError as e:
            job_id = int(job_snapshot.get("id", 0))
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=f"payload_validation_error: {e}",
                db_updated_by_runner=False,
            )

        job_id = int(job_snapshot["id"])
        job_key = str(job_snapshot.get("job_key", "langgraph"))
        try:
            result_state = _run_minimal_graph(payload.prompt)
            out_dir = Path(context.repo_root) / "experimental_results" / "langgraph_outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_key = job_key.replace("/", "_").replace("-", "_")
            out_path = out_dir / f"{safe_key}_{job_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"state": result_state, "job_id": job_id, "job_key": job_key},
                    f,
                    indent=2,
                )
            return JobRunOutcome(
                job_id=job_id,
                result_ref=str(out_path),
                error=None,
                db_updated_by_runner=False,
            )
        except Exception as e:
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=str(e)[:1000],
                db_updated_by_runner=False,
            )
