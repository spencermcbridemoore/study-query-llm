"""LangGraph job runner for langgraph_run orchestration jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import ValidationError

from .job_payload_models import parse_job_snapshot, parse_langgraph_run_payload
from .job_runners import JobRunContext, JobRunOutcome


def _run_minimal_graph(
    prompt: str,
    *,
    use_checkpoint: bool = False,
    thread_id: Optional[str] = None,
    job_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Run a minimal LangGraph echo graph. Returns (state dict, optional checkpoint_refs)."""
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

    checkpoint_refs: Optional[Dict[str, Any]] = None
    if use_checkpoint:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        compiled = workflow.compile(checkpointer=checkpointer)
        tid = thread_id or (f"job_{job_id}" if job_id is not None else "job_0")
        config: Dict[str, Any] = {"configurable": {"thread_id": tid}}
        result = compiled.invoke({"input": prompt, "output": ""}, config)
        snapshot = compiled.get_state(config)
        cid = None
        if snapshot and snapshot.config:
            cid = (snapshot.config.get("configurable") or {}).get("checkpoint_id")
        checkpoint_refs = {"thread_id": tid, "checkpoint_id": cid}
    else:
        compiled = workflow.compile()
        result = compiled.invoke({"input": prompt, "output": ""})

    return dict(result), checkpoint_refs


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
        config = payload.config or {}
        use_checkpoint = bool(config.get("checkpoint"))
        thread_id = config.get("thread_id")

        try:
            result_state, checkpoint_refs = _run_minimal_graph(
                payload.prompt,
                use_checkpoint=use_checkpoint,
                thread_id=thread_id,
                job_id=job_id,
            )
            out_dir = Path(context.repo_root) / "experimental_results" / "langgraph_outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_key = job_key.replace("/", "_").replace("-", "_")
            out_path = out_dir / f"{safe_key}_{job_id}.json"
            out_data: Dict[str, Any] = {
                "state": result_state,
                "job_id": job_id,
                "job_key": job_key,
            }
            if checkpoint_refs:
                out_data["checkpoint_refs"] = checkpoint_refs
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            return JobRunOutcome(
                job_id=job_id,
                result_ref=str(out_path),
                error=None,
                db_updated_by_runner=False,
                metadata={"checkpoint_refs": checkpoint_refs} if checkpoint_refs else None,
            )
        except Exception as e:
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=str(e)[:1000],
                db_updated_by_runner=False,
            )
