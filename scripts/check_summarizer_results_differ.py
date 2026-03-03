"""
Check that summarizer runs (LLM) do NOT produce identical results to None summarizer.

Queries the DB for clustering_run metrics, then for each (dataset, engine) that has
both "None" and at least one LLM summarizer, compares mean objective and (when
available) full objective lists per k. If an LLM summarizer matches None (same mean
or identical list), the script reports FAIL; otherwise PASS.

Usage:
  python scripts/check_summarizer_results_differ.py

Requires DATABASE_URL and existing clustering_run data from sweep ingestion.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from study_query_llm.config import config
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.sweep_query_service import SweepQueryService


# Tolerance for float comparison (mean objective)
MEAN_TOLERANCE = 1e-9


def main() -> int:
    db = DatabaseConnectionV2(config.database.connection_string)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepQueryService(repo)
        df = svc.get_sweep_metrics_df(exclude_pre_fix=True)

    if df.empty:
        print("No clustering_run metrics in DB. Nothing to compare.")
        return 0

    # Group by (dataset, engine) and collect summarizers
    keys = df.groupby(["dataset", "engine"])["summarizer"].apply(set).to_dict()
    summarizers_by_key = {k: sorted(v) for k, v in keys.items()}

    none_label = "None"
    failures: list[str] = []
    checks = 0

    for (dataset, engine), summarizers in summarizers_by_key.items():
        if none_label not in summarizers:
            continue
        llm_summarizers = [s for s in summarizers if s != none_label]
        if not llm_summarizers:
            continue

        sub = df[(df["dataset"] == dataset) & (df["engine"] == engine)]
        k_values = sub["k"].unique()

        for k in k_values:
            k_sub = sub[sub["k"] == k]
            none_rows = k_sub[k_sub["summarizer"] == none_label]
            none_objectives = none_rows["objective"].dropna().tolist()
            if not none_objectives:
                continue
            none_mean = sum(none_objectives) / len(none_objectives)

            for summ in llm_summarizers:
                summ_rows = k_sub[k_sub["summarizer"] == summ]
                summ_objectives = summ_rows["objective"].dropna().tolist()
                if not summ_objectives:
                    continue
                summ_mean = sum(summ_objectives) / len(summ_objectives)
                checks += 1

                # Same number of restarts and identical sorted lists? (strongest signal)
                identical_list = (
                    len(summ_objectives) == len(none_objectives)
                    and sorted(summ_objectives) == sorted(none_objectives)
                )
                if identical_list:
                    failures.append(
                        f"  Identical objective list: dataset={dataset!r}, engine={engine!r}, "
                        f"k={k}, summarizer={summ!r} vs None"
                    )
                elif abs(summ_mean - none_mean) < MEAN_TOLERANCE:
                    # Same mean but different list (e.g. permutation)
                    failures.append(
                        f"  Same mean objective: dataset={dataset!r}, engine={engine!r}, "
                        f"k={k}, summarizer={summ!r} vs None (mean={none_mean})"
                    )

    print("Summarizer vs None check (DB clustering_run metrics)")
    print("-" * 60)
    print(f"Pairs (dataset, engine) with both None and LLM summarizers: "
          f"{sum(1 for s in summarizers_by_key.values() if none_label in s and any(x != none_label for x in s))}")
    print(f"Comparisons performed: {checks}")

    if failures:
        print("\nFAIL: Some LLM summarizer runs match None.")
        print(f"Total comparisons where LLM matched None: {len(failures)}")
        max_show = 30
        for f in failures[:max_show]:
            print(f)
        if len(failures) > max_show:
            print(f"  ... and {len(failures) - max_show} more.")
        return 1

    print("\nPASS: No LLM summarizer produced identical results to None.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
