"""Orchestration job runners, payloads, factory, and reducer."""

from .job_payload_models import (
    AnalysisRunPayload,
    FinalizeRunPayload,
    JobSnapshot,
    LangGraphRunPayload,
    McqRunPayload,
    ReduceKPayload,
    RunKTryPayload,
    parse_analysis_run_payload,
    parse_finalize_run_payload,
    parse_job_snapshot,
    parse_mcq_run_payload,
    parse_langgraph_run_payload,
    parse_reduce_k_payload,
    parse_run_k_try_payload,
)
from .job_reducer_service import JobReducerService
from .job_runner_factory import create_job_runner, get_supported_job_types, register_job_runner
from .job_runners import (
    AnalysisRunRunner,
    FinalizeRunRunner,
    JobRunContext,
    JobRunOutcome,
    JobRunner,
    McqRunRunner,
    ReduceKRunner,
    RunKTryRunner,
)
from .langgraph_job_runner import LangGraphJobRunner
from .p0_baseline import build_p0_baseline_snapshot, normalize_result_ref, normalize_result_refs
from .reducer_plugin import (
    ClusteringReducerPlugin,
    ReducerInput,
    ReducerOutput,
    ReducerPlugin,
)

__all__ = [
    "FinalizeRunRunner",
    "AnalysisRunRunner",
    "AnalysisRunPayload",
    "FinalizeRunPayload",
    "JobReducerService",
    "JobRunContext",
    "JobRunOutcome",
    "JobRunner",
    "JobSnapshot",
    "LangGraphJobRunner",
    "LangGraphRunPayload",
    "McqRunPayload",
    "ReduceKPayload",
    "McqRunRunner",
    "ReduceKRunner",
    "RunKTryPayload",
    "RunKTryRunner",
    "create_job_runner",
    "get_supported_job_types",
    "register_job_runner",
    "parse_job_snapshot",
    "parse_analysis_run_payload",
    "parse_mcq_run_payload",
    "parse_langgraph_run_payload",
    "parse_reduce_k_payload",
    "parse_finalize_run_payload",
    "parse_run_k_try_payload",
    "build_p0_baseline_snapshot",
    "normalize_result_ref",
    "normalize_result_refs",
    "ReducerInput",
    "ReducerOutput",
    "ReducerPlugin",
    "ClusteringReducerPlugin",
]
