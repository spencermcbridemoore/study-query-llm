"""Execute one MCQ answer-position probe target (for sweep worker)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.experiments.mcq_answer_position_probe import run_probe
from study_query_llm.experiments.mcq_run_persistence import persist_mcq_probe_result
from study_query_llm.utils.mcq_template_loader import labels_for_mcq_options


def _run_async_sync(coro_factory):
    from concurrent.futures import ThreadPoolExecutor

    def _runner():
        return asyncio.run(coro_factory())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_runner).result()


def execute_mcq_standalone_run(
    *,
    db: DatabaseConnectionV2,
    request_id: int,
    run_key: str,
    target: Dict[str, Any],
    worker_label: str = "",
) -> Tuple[Optional[int], Optional[str]]:
    """
    Run MCQ probe for one target; persist mcq_run and record delivery.

    Returns (run_group_id, None) on success, (None, error_message) on failure.
    Caller should skip before claim when mcq_run_key_exists_in_db and not force.
    """
    deployment = str(target.get("deployment") or "")
    subject = str(target.get("subject") or "physics")
    level_raw = target.get("level")
    level = str(level_raw).strip() if level_raw is not None else ""
    spread_correct = bool(target.get("spread_correct_answer_uniformly", False))
    num_options = int(target.get("options_per_question") or 4)
    question_count = int(target.get("questions_per_test") or 10)
    label_style = str(target.get("label_style") or "upper")
    samples = int(target.get("samples_per_combo") or 1)
    concurrency = int(target.get("concurrency") or 8)
    temperature = float(target.get("temperature") or 0.7)
    max_tokens = int(target.get("max_tokens") or 900)
    progress_every = int(target.get("progress_every") or 0)

    try:
        labels = labels_for_mcq_options(num_options, label_style)
    except ValueError as exc:
        return None, str(exc)

    prefix = f"[{worker_label}] " if worker_label else ""

    def _progress(
        completed: int,
        samples: int,
        valid_runs: int,
        call_errors: int,
        parse_failures: int,
    ) -> None:
        if progress_every <= 0:
            return
        print(
            f"{prefix}[progress] completed={completed}/{samples} "
            f"valid={valid_runs} call_errors={call_errors} parse_failures={parse_failures}"
        )

    async def _run():
        return await run_probe(
            deployment=deployment,
            subject=subject,
            question_count=question_count,
            labels=labels,
            samples=samples,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=max_tokens,
            progress_every=progress_every,
            progress_callback=_progress if progress_every > 0 else None,
            level=level or None,
            spread_correct_answer_uniformly=spread_correct,
        )

    try:
        probe_details = _run_async_sync(_run)
    except Exception as exc:
        return None, str(exc)[:1000]

    run_id = persist_mcq_probe_result(
        db, request_id, run_key, target, probe_details
    )
    return run_id, None
