"""Sweep result saving and paraphraser factory shared across sweep scripts."""

import asyncio
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.summarization_service import (
    SummarizationService,
    SummarizationRequest,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "experimental_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Paraphraser factory
# ---------------------------------------------------------------------------


def create_paraphraser_for_llm(
    llm_deployment: Optional[str],
    db: DatabaseConnectionV2,
    *,
    combine_texts: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 256,
    timeout: float = 300.0,
) -> Optional[Callable[[List[str]], str]]:
    """Create a synchronous paraphraser callable for use inside ``run_sweep``.

    Args:
        llm_deployment: Azure deployment name (``None`` → return ``None``).
        db: Database connection for the summarization service.
        combine_texts: When ``True`` (default), concatenate all input texts
            into a single prompt and return one summary string.  When
            ``False``, pass texts directly to ``SummarizationService`` and
            return ``result.summaries`` (a list).
        temperature: Sampling temperature for the LLM call.
        max_tokens: Maximum tokens for the response.
        timeout: Per-call wall-clock timeout in seconds.
    """
    if llm_deployment is None:
        return None

    async def _paraphrase_async(texts: List[str]):
        async def _summarize():
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                service = SummarizationService(repository=repo)

                if combine_texts:
                    combined = (
                        "Summarize the following texts into a single "
                        "coherent summary:\n\n"
                    )
                    for i, text in enumerate(texts, 1):
                        combined += f"Text {i}:\n{text}\n\n"
                    req_texts = [combined]
                else:
                    req_texts = texts

                request = SummarizationRequest(
                    texts=req_texts,
                    llm_deployment=llm_deployment,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                try:
                    result = await service.summarize_batch(request)
                    if combine_texts:
                        return (
                            result.summaries[0]
                            if result.summaries
                            else (texts[0] if texts else "")
                        )
                    return result.summaries
                except Exception as exc:
                    msg = str(exc)
                    if "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg:
                        print("          [WARN] Content filtered, using original text")
                    else:
                        print(f"          [WARN] Summarization error: {msg[:100]}")
                    return texts[0] if texts else ""

        try:
            return await asyncio.wait_for(_summarize(), timeout=timeout)
        except asyncio.TimeoutError:
            print("      [WARN] Summarization timed out, using first text")
            return texts[0] if texts else ""

    def paraphrase_sync(texts: List[str]):
        """Synchronous wrapper suitable for ``run_sweep``."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
            asyncio.set_event_loop(None)
        except RuntimeError:
            pass
        return asyncio.run(_paraphrase_async(texts))

    return paraphrase_sync


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def serialize_sweep_result(result: Any) -> Dict[str, Any]:
    """Convert a single ``SweepResult`` to a JSON-safe dictionary."""
    data: Dict[str, Any] = {"pca": result.pca, "by_k": {}}

    if result.Z is not None:
        data["Z"] = result.Z.tolist()
    if result.Z_norm is not None:
        data["Z_norm"] = result.Z_norm.tolist()
    if result.dist is not None:
        data["dist"] = result.dist.tolist()

    for k, k_data in result.by_k.items():
        labels_raw = k_data.get("labels", [])
        labels_all_raw = k_data.get("labels_all")
        data["by_k"][k] = {
            "representatives": k_data.get("representatives", []),
            "labels": (
                labels_raw.tolist()
                if hasattr(labels_raw, "tolist")
                else labels_raw
            ),
            "labels_all": (
                [
                    l.tolist() if hasattr(l, "tolist") else l
                    for l in labels_all_raw
                ]
                if labels_all_raw is not None
                else None
            ),
            "objective": k_data.get("objective", {}),
            "objectives": k_data.get("objectives", []),
            "stability": k_data.get("stability"),
        }

    return data


def save_single_sweep_result(
    result: Any,
    output_file: str,
    ground_truth_labels: Optional[np.ndarray] = None,
    dataset_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a single ``SweepResult`` with metadata to a pickle file."""
    final = {
        "result": serialize_sweep_result(result),
        "ground_truth_labels": (
            ground_truth_labels.tolist()
            if ground_truth_labels is not None
            else None
        ),
        "dataset_name": dataset_name,
        "metadata": metadata or {},
    }
    with open(output_file, "wb") as f:
        pickle.dump(final, f)
    return output_file


def save_batch_sweep_results(
    all_results: Dict[str, Any],
    output_file: Optional[str] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    dataset_name: str = "unknown",
) -> str:
    """Save multiple ``SweepResult`` objects keyed by summarizer name."""
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(OUTPUT_DIR / f"pca_kllmeans_sweep_results_{ts}.pkl")

    serialized = {
        name: serialize_sweep_result(res)
        for name, res in all_results.items()
    }

    final = {
        "summarizers": serialized,
        "ground_truth_labels": (
            ground_truth_labels.tolist()
            if ground_truth_labels is not None
            else None
        ),
        "dataset_name": dataset_name,
    }
    with open(output_file, "wb") as f:
        pickle.dump(final, f)
    return output_file
