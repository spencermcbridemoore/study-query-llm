"""
Deterministic request/target schemas and adapters for sweep-request lifecycles.

This module provides:
- Backward-compatible clustering helpers (run-key and axis expansion).
- A registry-backed sweep-type adapter contract for typed request expansion.
- MCQ sweep target expansion/keys and analysis contract metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


# Schema version for request metadata (bump when breaking changes)
REQUEST_SCHEMA_VERSION = 2

# Valid request_status values
REQUEST_STATUS_REQUESTED = "requested"
REQUEST_STATUS_RUNNING = "running"
REQUEST_STATUS_FULFILLED = "fulfilled"
REQUEST_STATUS_CANCELLED = "cancelled"

# Analysis lifecycle values for request metadata.
ANALYSIS_STATUS_NOT_REQUIRED = "not_required"
ANALYSIS_STATUS_NOT_STARTED = "not_started"
ANALYSIS_STATUS_RUNNING = "running"
ANALYSIS_STATUS_COMPLETE = "complete"
ANALYSIS_STATUS_FAILED = "failed"

# Sweep types used by typed request dispatch.
SWEEP_TYPE_CLUSTERING = "clustering"
SWEEP_TYPE_MCQ = "mcq"
MCQ_ORCHESTRATION_GRAPH_KIND = "single_job_per_run"
MCQ_HISTORICAL_BACKFILL_POLICY = "compatibility_only"
MCQ_DETERMINISM_CLASS = "non_deterministic"


@dataclass(frozen=True)
class RunTarget:
    """Single clustering run target from expanded parameter axes."""

    dataset: str
    embedding_engine: str
    summarizer: str
    entry_max: int
    n_restarts_suffix: str  # e.g. "50runs"


@dataclass(frozen=True)
class SweepTargetSpec:
    """Generic typed target payload paired with deterministic run_key."""

    run_key: str
    target: Dict[str, Any]


@dataclass(frozen=True)
class SweepAnalysisDefinition:
    """Formal analysis contract entry for a sweep type."""

    analysis_key: str
    method_name: str
    method_version: str
    scope: str  # "run" | "sweep"
    required: bool = True
    blocking: bool = False
    result_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OrchestrationJobSpec:
    """Single schedulable orchestration-job node emitted by adapters."""

    job_type: str
    job_key: str
    base_run_key: Optional[str]
    payload_json: Dict[str, Any]
    max_attempts: int = 3
    seed_value: Optional[int] = None
    depends_on_job_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OrchestrationGraphSpec:
    """Adapter-defined orchestration graph specification for one request."""

    graph_kind: str
    jobs: Tuple[OrchestrationJobSpec, ...]
    metadata_updates: Dict[str, Any] = field(default_factory=dict)


class SweepTypeAdapter(Protocol):
    """Adapter contract for typed sweep request expansion/finalization."""

    sweep_type: str
    request_group_type: str
    run_group_type: str
    sweep_group_type: str

    def default_algorithm(self) -> str:
        ...

    def build_targets(
        self,
        parameter_axes: Dict[str, Any],
        fixed_config: Dict[str, Any],
        entry_max: Optional[int],
        n_restarts_suffix: str,
    ) -> List[SweepTargetSpec]:
        ...

    def analysis_definitions(self) -> List[SweepAnalysisDefinition]:
        ...

    def build_orchestration_graph(
        self,
        *,
        request_group_id: int,
        run_key_to_target: Dict[str, Dict[str, Any]],
        fixed_config: Dict[str, Any],
        execution_mode: str,
        shard_config: Dict[str, Any],
        analysis_catalog: List[Dict[str, Any]],
        enable_analysis_jobs: bool,
    ) -> OrchestrationGraphSpec:
        ...


def _safe_name(s: str) -> str:
    """Normalize string for use in run_key."""
    return str(s).replace("-", "_").replace("/", "_").replace(" ", "_")


def _coerce_axis_values(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return [raw]


def _axis_values(parameter_axes: Dict[str, Any], names: Sequence[str]) -> List[Any]:
    for name in names:
        if name in parameter_axes:
            return _coerce_axis_values(parameter_axes.get(name))
    return []


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_k_ranges(
    *,
    fixed_config: Dict[str, Any],
    shard_config: Dict[str, Any],
) -> List[Tuple[int, int]]:
    raw_ranges = list(shard_config.get("k_ranges") or [])
    ranges: List[Tuple[int, int]] = []
    for item in raw_ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        lo = _safe_int(item[0], 2)
        hi = _safe_int(item[1], lo)
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    if ranges:
        return ranges
    k_min = _safe_int(fixed_config.get("k_min"), 2)
    k_max = _safe_int(fixed_config.get("k_max"), max(k_min, 20))
    if k_max < k_min:
        k_min, k_max = k_max, k_min
    return [(k_min, k_max)]


def normalize_summarizer(value: Any) -> str:
    """Convert summarizer value to canonical string for run_key.

    None -> "None", otherwise str(value).
    """
    if value is None:
        return "None"
    return str(value)


def build_run_key(
    dataset: str,
    embedding_engine: str,
    summarizer: str,
    entry_max: int,
    n_restarts_suffix: str = "50runs",
) -> str:
    """Build deterministic clustering run_key used by current ingestion scripts."""
    engine_safe = _safe_name(embedding_engine)
    sum_safe = _safe_name(normalize_summarizer(summarizer))
    return f"{dataset}_{engine_safe}_{sum_safe}_{entry_max}_{n_restarts_suffix}"


def expand_parameter_axes(
    parameter_axes: Dict[str, List[Any]],
    entry_max: int,
    n_restarts_suffix: str = "50runs",
) -> List[RunTarget]:
    """Expand clustering parameter axes into deterministic list of RunTargets."""
    datasets = parameter_axes.get("datasets", [])
    engines = parameter_axes.get("embedding_engines", [])
    summarizers = parameter_axes.get("summarizers", [])

    targets: List[RunTarget] = []
    for dataset in datasets:
        for engine in engines:
            for summ in summarizers:
                targets.append(
                    RunTarget(
                        dataset=str(dataset),
                        embedding_engine=str(engine),
                        summarizer=normalize_summarizer(summ),
                        entry_max=entry_max,
                        n_restarts_suffix=n_restarts_suffix,
                    )
                )
    return targets


def targets_to_run_keys(targets: List[RunTarget]) -> List[str]:
    """Convert clustering RunTargets to run_key strings."""
    return [
        build_run_key(
            t.dataset,
            t.embedding_engine,
            t.summarizer,
            t.entry_max,
            t.n_restarts_suffix,
        )
        for t in targets
    ]


def build_mcq_run_key(
    deployment: str,
    level: str,
    subject: str,
    options_per_question: int,
    questions_per_test: int,
    label_style: str,
    spread_correct_answer_uniformly: bool,
    samples_per_combo: int,
    template_version: str = "v1",
) -> str:
    """Build deterministic run_key for an MCQ tuple and sample budget."""
    spread = "spread" if bool(spread_correct_answer_uniformly) else "no_spread"
    return (
        "mcq_"
        f"{_safe_name(deployment)}_"
        f"{_safe_name(level)}_"
        f"{_safe_name(subject)}_"
        f"{int(options_per_question)}opt_"
        f"{int(questions_per_test)}q_"
        f"{_safe_name(label_style)}_"
        f"{spread}_"
        f"{int(samples_per_combo)}samples_"
        f"{_safe_name(template_version)}"
    )


class ClusteringSweepAdapter:
    """Typed adapter for existing clustering sweep request behavior."""

    sweep_type = SWEEP_TYPE_CLUSTERING
    request_group_type = "clustering_sweep_request"
    run_group_type = "clustering_run"
    sweep_group_type = "clustering_sweep"

    def default_algorithm(self) -> str:
        return "cosine_kllmeans_no_pca"

    def build_targets(
        self,
        parameter_axes: Dict[str, Any],
        fixed_config: Dict[str, Any],
        entry_max: Optional[int],
        n_restarts_suffix: str,
    ) -> List[SweepTargetSpec]:
        if entry_max is None:
            raise ValueError("entry_max is required for clustering sweep requests")
        targets = expand_parameter_axes(
            parameter_axes,
            entry_max=int(entry_max),
            n_restarts_suffix=n_restarts_suffix,
        )
        return [
            SweepTargetSpec(
                run_key=build_run_key(
                    t.dataset,
                    t.embedding_engine,
                    t.summarizer,
                    t.entry_max,
                    t.n_restarts_suffix,
                ),
                target={
                    "dataset": t.dataset,
                    "embedding_engine": t.embedding_engine,
                    "summarizer": t.summarizer,
                },
            )
            for t in targets
        ]

    def analysis_definitions(self) -> List[SweepAnalysisDefinition]:
        return []

    def build_orchestration_graph(
        self,
        *,
        request_group_id: int,
        run_key_to_target: Dict[str, Dict[str, Any]],
        fixed_config: Dict[str, Any],
        execution_mode: str,
        shard_config: Dict[str, Any],
        analysis_catalog: List[Dict[str, Any]],
        enable_analysis_jobs: bool,
    ) -> OrchestrationGraphSpec:
        del request_group_id, analysis_catalog, enable_analysis_jobs
        resolved_shard_config = dict(shard_config or {})
        if str(execution_mode or "standalone").lower() != "sharded":
            k_min = _safe_int(fixed_config.get("k_min"), 2)
            k_max = _safe_int(fixed_config.get("k_max"), max(k_min, 20))
            if k_max < k_min:
                k_min, k_max = k_max, k_min
            resolved_shard_config = {
                "k_ranges": [[k, k] for k in range(k_min, k_max + 1)],
                "tries_per_k": _safe_int(fixed_config.get("n_restarts"), 50),
                "standalone_special_case": True,
            }

        k_ranges = _normalize_k_ranges(
            fixed_config=fixed_config,
            shard_config=resolved_shard_config,
        )
        tries_per_k = max(
            1,
            _safe_int(
                resolved_shard_config.get("tries_per_k", fixed_config.get("n_restarts", 1)),
                1,
            ),
        )
        max_attempts = max(1, _safe_int(fixed_config.get("max_attempts"), 3))
        base_seed = _safe_int(fixed_config.get("base_seed"), 0)

        jobs: List[OrchestrationJobSpec] = []
        for run_key, target in run_key_to_target.items():
            reduce_job_keys: List[str] = []
            for k_min, k_max in k_ranges:
                leaf_job_keys: List[str] = []
                for try_idx in range(tries_per_k):
                    leaf_job_key = f"{run_key}__k{k_min}_{k_max}__try{try_idx}"
                    leaf_job_keys.append(leaf_job_key)
                    jobs.append(
                        OrchestrationJobSpec(
                            job_type="run_k_try",
                            job_key=leaf_job_key,
                            base_run_key=run_key,
                            payload_json={
                                "run_key": run_key,
                                "dataset": target.get("dataset"),
                                "embedding_engine": target.get("embedding_engine"),
                                "summarizer": target.get("summarizer", "None"),
                                "k_min": int(k_min),
                                "k_max": int(k_max),
                                "try_idx": int(try_idx),
                                "tries_per_k": int(tries_per_k),
                            },
                            max_attempts=max_attempts,
                            seed_value=base_seed + try_idx,
                        )
                    )

                reduce_job_key = f"{run_key}__reduce_k{k_min}_{k_max}"
                reduce_job_keys.append(reduce_job_key)
                jobs.append(
                    OrchestrationJobSpec(
                        job_type="reduce_k",
                        job_key=reduce_job_key,
                        base_run_key=run_key,
                        payload_json={
                            "run_key": run_key,
                            "dataset": target.get("dataset"),
                            "embedding_engine": target.get("embedding_engine"),
                            "summarizer": target.get("summarizer", "None"),
                            "k_min": int(k_min),
                            "k_max": int(k_max),
                            "tries_per_k": int(tries_per_k),
                        },
                        max_attempts=max_attempts,
                        depends_on_job_keys=tuple(leaf_job_keys),
                    )
                )

            finalize_job_key = f"{run_key}__finalize_run"
            jobs.append(
                OrchestrationJobSpec(
                    job_type="finalize_run",
                    job_key=finalize_job_key,
                    base_run_key=run_key,
                    payload_json={
                        "run_key": run_key,
                        "dataset": target.get("dataset"),
                        "embedding_engine": target.get("embedding_engine"),
                        "summarizer": target.get("summarizer", "None"),
                        "k_ranges": [[int(lo), int(hi)] for (lo, hi) in k_ranges],
                        "tries_per_k": int(tries_per_k),
                    },
                    max_attempts=max_attempts,
                    depends_on_job_keys=tuple(reduce_job_keys),
                )
            )

        return OrchestrationGraphSpec(
            graph_kind="k_try_reduce_finalize",
            jobs=tuple(jobs),
            metadata_updates={},
        )


class McqSweepAdapter:
    """Typed adapter for MCQ probe sweeps with formal analysis contract."""

    sweep_type = SWEEP_TYPE_MCQ
    request_group_type = "mcq_sweep_request"
    run_group_type = "mcq_run"
    sweep_group_type = "mcq_sweep"

    def default_algorithm(self) -> str:
        return "mcq_answer_position_probe"

    def build_targets(
        self,
        parameter_axes: Dict[str, Any],
        fixed_config: Dict[str, Any],
        entry_max: Optional[int],
        n_restarts_suffix: str,
    ) -> List[SweepTargetSpec]:
        del n_restarts_suffix  # not used for MCQ keys

        levels = _axis_values(parameter_axes, ("levels", "level"))
        subjects = _axis_values(parameter_axes, ("subjects", "subject"))
        options_list = _axis_values(parameter_axes, ("options_per_question",))
        questions_list = _axis_values(parameter_axes, ("questions_per_test",))
        labels = _axis_values(parameter_axes, ("label_styles", "label_style"))
        if not labels:
            labels = ["upper"]
        spreads = _axis_values(parameter_axes, ("spread_correct_answer_uniformly",))
        deployments = _axis_values(parameter_axes, ("deployments", "llms", "models"))

        samples_per_combo = int(fixed_config.get("samples_per_combo", entry_max or 1))
        template_version = str(fixed_config.get("template_version", "v1"))
        concurrency = int(fixed_config.get("concurrency", 8))
        temperature = float(fixed_config.get("temperature", 0.7))
        max_tokens = int(fixed_config.get("max_tokens", 900))
        progress_every = int(fixed_config.get("progress_every", 0))

        targets: List[SweepTargetSpec] = []
        for level in levels:
            for subject in subjects:
                for deployment in deployments:
                    for num_options in options_list:
                        for num_questions in questions_list:
                            for label_style in labels:
                                for spread in spreads:
                                    run_key = build_mcq_run_key(
                                        deployment=str(deployment),
                                        level=str(level),
                                        subject=str(subject),
                                        options_per_question=int(num_options),
                                        questions_per_test=int(num_questions),
                                        label_style=str(label_style),
                                        spread_correct_answer_uniformly=bool(spread),
                                        samples_per_combo=samples_per_combo,
                                        template_version=template_version,
                                    )
                                    targets.append(
                                        SweepTargetSpec(
                                            run_key=run_key,
                                            target={
                                                "deployment": str(deployment),
                                                "level": str(level),
                                                "subject": str(subject),
                                                "options_per_question": int(num_options),
                                                "questions_per_test": int(num_questions),
                                                "label_style": str(label_style),
                                                "spread_correct_answer_uniformly": bool(spread),
                                                "samples_per_combo": samples_per_combo,
                                                "template_version": template_version,
                                                "concurrency": concurrency,
                                                "temperature": temperature,
                                                "max_tokens": max_tokens,
                                                "progress_every": progress_every,
                                            },
                                        )
                                    )
        return targets

    def analysis_definitions(self) -> List[SweepAnalysisDefinition]:
        return [
            SweepAnalysisDefinition(
                analysis_key="mcq_compliance",
                method_name="mcq_compliance_metrics",
                method_version="1.0",
                scope="run",
                required=True,
                blocking=False,
                result_keys=(
                    "format_compliance_rate",
                    "question_count_compliance_rate",
                    "answer_key_parse_rate",
                ),
            ),
            SweepAnalysisDefinition(
                analysis_key="mcq_answer_position_distribution",
                method_name="mcq_answer_position_distribution",
                method_version="1.0",
                scope="sweep",
                required=True,
                blocking=False,
                result_keys=(
                    "position_distribution",
                    "position_mean",
                    "position_stdev",
                ),
            ),
            SweepAnalysisDefinition(
                analysis_key="mcq_answer_position_chi_square",
                method_name="mcq_answer_position_chi_square",
                method_version="1.0",
                scope="sweep",
                required=False,
                blocking=False,
                result_keys=("chi_square", "p_value"),
            ),
        ]

    def build_orchestration_graph(
        self,
        *,
        request_group_id: int,
        run_key_to_target: Dict[str, Dict[str, Any]],
        fixed_config: Dict[str, Any],
        execution_mode: str,
        shard_config: Dict[str, Any],
        analysis_catalog: List[Dict[str, Any]],
        enable_analysis_jobs: bool,
    ) -> OrchestrationGraphSpec:
        del execution_mode, shard_config
        max_attempts = max(1, _safe_int(fixed_config.get("max_attempts"), 3))

        jobs: List[OrchestrationJobSpec] = []
        mcq_job_keys: List[str] = []
        for run_key, target in run_key_to_target.items():
            payload = dict(target or {})
            payload["run_key"] = run_key
            payload["determinism_class"] = MCQ_DETERMINISM_CLASS
            job_key = f"{run_key}__mcq_run"
            mcq_job_keys.append(job_key)
            jobs.append(
                OrchestrationJobSpec(
                    job_type="mcq_run",
                    job_key=job_key,
                    base_run_key=run_key,
                    payload_json=payload,
                    max_attempts=max_attempts,
                )
            )

        analysis_job_count = 0
        if enable_analysis_jobs:
            for entry in analysis_catalog:
                if not isinstance(entry, dict):
                    continue
                analysis_key = str(entry.get("analysis_key") or "").strip()
                if not analysis_key:
                    continue
                payload = {
                    "request_id": int(request_group_id),
                    "sweep_type": SWEEP_TYPE_MCQ,
                    "analysis_key": analysis_key,
                    "scope": str(entry.get("scope") or "run"),
                    "method_name": str(entry.get("method_name") or analysis_key),
                    "method_version": str(entry.get("method_version") or "1.0"),
                    "required": bool(entry.get("required")),
                    "blocking": bool(entry.get("blocking")),
                    "result_keys": list(entry.get("result_keys") or []),
                }
                job_key = f"req{int(request_group_id)}__analysis__{analysis_key}"
                jobs.append(
                    OrchestrationJobSpec(
                        job_type="analysis_run",
                        job_key=job_key,
                        base_run_key=analysis_key,
                        payload_json=payload,
                        max_attempts=max_attempts,
                        depends_on_job_keys=tuple(mcq_job_keys),
                    )
                )
                analysis_job_count += 1

        return OrchestrationGraphSpec(
            graph_kind=MCQ_ORCHESTRATION_GRAPH_KIND,
            jobs=tuple(jobs),
            metadata_updates={
                "orchestration_graph_kind": MCQ_ORCHESTRATION_GRAPH_KIND,
                "historical_backfill_policy": MCQ_HISTORICAL_BACKFILL_POLICY,
                "determinism_class": MCQ_DETERMINISM_CLASS,
                "analysis_execution_mode": (
                    "orchestration_jobs" if enable_analysis_jobs else "compatibility_only"
                ),
                "analysis_job_count": int(analysis_job_count),
            },
        )


_SWEEP_TYPE_REGISTRY: Dict[str, SweepTypeAdapter] = {
    SWEEP_TYPE_CLUSTERING: ClusteringSweepAdapter(),
    SWEEP_TYPE_MCQ: McqSweepAdapter(),
}


def list_registered_sweep_types() -> List[str]:
    return sorted(_SWEEP_TYPE_REGISTRY.keys())


def list_request_group_types() -> List[str]:
    return sorted({adapter.request_group_type for adapter in _SWEEP_TYPE_REGISTRY.values()})


def get_sweep_type_adapter(sweep_type: Optional[str]) -> SweepTypeAdapter:
    """Return registered adapter for sweep_type (defaults to clustering)."""
    key = (sweep_type or SWEEP_TYPE_CLUSTERING).strip().lower()
    adapter = _SWEEP_TYPE_REGISTRY.get(key)
    if adapter is None:
        supported = ", ".join(list_registered_sweep_types())
        raise ValueError(f"Unsupported sweep_type={sweep_type!r}; supported: {supported}")
    return adapter
