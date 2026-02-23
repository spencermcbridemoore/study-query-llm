"""Sweep result saving and paraphraser factory shared across sweep scripts."""

import asyncio
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sqlalchemy import text as sa_text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import ProvenanceService
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
                    combined = ""
                    for i, text in enumerate(texts, 1):
                        combined += f"Text {i}:\n{text}\n\n"
                    req_texts = [combined.strip()]
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


# ---------------------------------------------------------------------------
# NeonDB in-memory ingestion (digest during calculation)
# ---------------------------------------------------------------------------

_METRICS = [
    "objective",
    "dispersion",
    "silhouette",
    "ari",
    "cosine_sim",
    "cosine_sim_norm",
]


def _dist_from_result(result_dict: Dict[str, Any]) -> Optional[np.ndarray]:
    """Compute or retrieve a cosine-distance matrix from a serialized result dict."""
    dist = result_dict.get("dist")
    if dist is not None:
        return np.asarray(dist)
    Z = result_dict.get("Z")
    if Z is None:
        return None
    Z = np.asarray(Z)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = Z / np.maximum(norms, 1e-12)
    return np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, 2.0)


def _try_ari_safe(gt: Optional[np.ndarray], labels: Optional[np.ndarray]) -> Optional[float]:
    if gt is None or labels is None or len(labels) != len(gt):
        return None
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(gt, labels))
    except Exception:
        return None


def _try_silhouette_safe(
    dist: Optional[np.ndarray],
    labels: Optional[np.ndarray],
) -> Optional[float]:
    if dist is None or labels is None:
        return None
    try:
        from sklearn.metrics import silhouette_score
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        return None


def _extract_by_k_metrics(
    result_dict: Dict[str, Any],
    ground_truth_labels: Optional[np.ndarray],
) -> Dict[int, Dict[str, List]]:
    """Extract per-k metric arrays from an in-memory serialized sweep result.

    Returns a dict of ``{k: {metric_name: [values…]}}`` matching the layout
    written by ``ingest_sweep_to_db.py``.
    """
    by_k_raw = result_dict.get("by_k") or {}
    gt = ground_truth_labels
    if gt is not None:
        gt = np.asarray(gt)
    n_samples = len(gt) if gt is not None else 0

    dist = _dist_from_result(result_dict)

    by_k: Dict[int, Dict[str, List]] = defaultdict(lambda: {m: [] for m in _METRICS})

    for k_str, entry in by_k_raw.items():
        try:
            k = int(k_str)
        except (ValueError, TypeError):
            continue
        objectives = entry.get("objectives") or []
        labels_all = entry.get("labels_all") or []
        n_restarts = max(len(objectives), len(labels_all))
        if n_restarts == 0:
            continue

        for i in range(n_restarts):
            ob = objectives[i] if i < len(objectives) else None
            lab = np.asarray(labels_all[i]) if (labels_all and i < len(labels_all)) else None

            n = n_samples or (len(lab) if lab is not None else 0)
            dispersion = (float(ob) / n) if (ob is not None and n > 0) else None
            cosine_sim = (1.0 - dispersion) if dispersion is not None else None
            cosine_sim_norm = ((cosine_sim + 1.0) / 2.0) if cosine_sim is not None else None
            sil = _try_silhouette_safe(dist, lab)
            ari = _try_ari_safe(gt, lab)

            bucket = by_k[k]
            bucket["objective"].append(float(ob) if ob is not None else None)
            bucket["dispersion"].append(dispersion)
            bucket["silhouette"].append(sil)
            bucket["ari"].append(ari)
            bucket["cosine_sim"].append(cosine_sim)
            bucket["cosine_sim_norm"].append(cosine_sim_norm)

    return dict(by_k)


def ingest_result_to_db(
    result: Any,
    metadata: Dict[str, Any],
    ground_truth_labels: Optional[np.ndarray],
    db: DatabaseConnectionV2,
    run_key: str,
) -> Optional[int]:
    """Save an in-memory sweep result directly to NeonDB as Group/GroupLink entries.

    This mirrors the logic in ``scripts/ingest_sweep_to_db.py`` but operates on
    the live result object rather than a pkl file, so it can be called immediately
    after each run completes (no intermediate file required).

    Args:
        result: The ``SweepResult`` object returned by ``run_sweep()``.
        metadata: The metadata dict you would normally pass to
            ``save_single_sweep_result()`` (must include ``benchmark_source``,
            ``embedding_engine``, ``summarizer``, ``n_restarts``).
        ground_truth_labels: Ground-truth label array (``None`` for datasets
            without ground truth such as estela).
        db: Active ``DatabaseConnectionV2`` instance.
        run_key: Unique idempotency key, e.g.
            ``"dbpedia_embed_v_4_0_gpt_5_chat_300_50runs"``.

    Returns:
        The run-group ID written to the DB, or ``None`` if the key already
        existed (skipped) or the write failed.
    """
    result_dict = serialize_sweep_result(result)
    dataset = metadata.get("benchmark_source", "unknown")
    engine = metadata.get("embedding_engine", "?")
    summarizer = str(metadata.get("summarizer", "None"))
    n_restarts = metadata.get("n_restarts", 50)
    n_samples = metadata.get("actual_entry_count", 0)
    data_type = "50runs"

    try:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            provenance = ProvenanceService(repo)

            # Idempotency: skip if already ingested under this run_key
            existing = session.query(Group).filter(
                Group.group_type == "run",
                sa_text("metadata_json->>'run_key' = :rk"),
            ).params(rk=run_key).first()
            if existing:
                print(f"      [SKIP] Already in DB: run_key={run_key} (group {existing.id})")
                return None

            run_metadata = {
                "algorithm": "cosine_kllmeans_no_pca",
                "run_key": run_key,
                "dataset": dataset,
                "embedding_engine": engine,
                "summarizer": summarizer,
                "n_restarts": n_restarts,
                "n_samples": n_samples,
                "data_type": data_type,
                "k_range": [
                    metadata.get("sweep_config", {}).get("k_min", 2),
                    metadata.get("sweep_config", {}).get("k_max", 20),
                ],
                "source": "digested_during_calculation",
                "ingested_at": datetime.utcnow().isoformat(),
                **{k: v for k, v in metadata.items() if k not in ("sweep_config",)},
            }

            run_id = provenance.create_run_group(
                algorithm="cosine_kllmeans_no_pca",
                config=run_metadata,
                name=f"sweep_{dataset}_{engine}_{data_type}",
                description=f"{dataset}/{engine}/{summarizer} ({n_restarts} restarts)",
            )

            # Overwrite metadata_json to flatten config (match ingest_sweep_to_db layout)
            run_group = repo.get_group_by_id(run_id)
            run_group.metadata_json = run_metadata
            session.flush()

            # Extract metric arrays and create one step group per k
            by_k = _extract_by_k_metrics(result_dict, ground_truth_labels)
            for k in sorted(by_k.keys()):
                metrics_for_k = by_k[k]
                step_metadata: Dict[str, Any] = {
                    "k": k,
                    "n_samples": n_samples,
                }
                for m in _METRICS:
                    step_metadata[f"{m}s"] = [
                        v for v in metrics_for_k.get(m, []) if v is not None
                    ]

                step_id = provenance.create_step_group(
                    parent_run_id=run_id,
                    step_name=f"k={k}",
                    step_type="clustering",
                    metadata=step_metadata,
                )
                repo.create_group_link(
                    parent_group_id=run_id,
                    child_group_id=step_id,
                    link_type="step",
                    position=k,
                )

            print(f"      [DB] Saved run_key={run_key} -> group {run_id} ({len(by_k)} k-steps)")
            return run_id

    except Exception as exc:
        print(f"      [ERROR] DB ingestion failed for run_key={run_key}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
