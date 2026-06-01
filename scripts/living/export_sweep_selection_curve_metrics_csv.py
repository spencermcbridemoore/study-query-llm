#!/usr/bin/env python3
"""Emit CSV rows for each candidate k in sweep-and-select clustering curves.

Loads ``*_selection_curve.json`` artifacts referenced from ``analysis_results``
(``result_key=artifacts``) for registry methods with ``fit_mode=sweep_select``
(default: kmeans+normalize+pca+sweep, gmm+normalize+pca+sweep).

Examples::

    python scripts/living/export_sweep_selection_curve_metrics_csv.py --out sweep_curves.csv

    python scripts/living/export_sweep_selection_curve_metrics_csv.py --out kmeans_only.csv ^
        --methods kmeans+normalize+pca+sweep --artifact-dir artifacts

Requires ``DATABASE_URL`` or ``CANONICAL_DATABASE_URL`` unless ``--database-url`` is set.
Artifact blobs must be readable via the configured storage backend (local dir or Azure).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.selection_curve_export import (
    default_sweep_method_names,
    iter_selection_curve_metric_rows,
    write_metrics_csv,
)
from study_query_llm.services.artifact_service import ArtifactService


def _parse_analysis_group_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL") or os.environ.get("CANONICAL_DATABASE_URL"),
        help="Postgres URL (default: DATABASE_URL or CANONICAL_DATABASE_URL)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=os.environ.get("ARTIFACT_DIR", "artifacts"),
        help="Local artifact root when using local storage backend",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Clustering method names to include "
            f"(default: {', '.join(default_sweep_method_names())})"
        ),
    )
    parser.add_argument(
        "--analysis-group-ids",
        default=None,
        help="Optional space/comma-separated analysis_run group ids to filter",
    )
    args = parser.parse_args(argv)

    if not args.database_url:
        print(
            "ERROR: No database URL; set DATABASE_URL or pass --database-url",
            file=sys.stderr,
        )
        return 2

    db = DatabaseConnectionV2(
        args.database_url,
        quiet=True,
        write_intent=default_write_intent_for_connection(args.database_url),
    )
    filt = _parse_analysis_group_ids(args.analysis_group_ids)

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        artifact_service = ArtifactService(repository=repo, artifact_dir=args.artifact_dir)
        rows = iter_selection_curve_metric_rows(
            session,
            artifact_service,
            method_names=args.methods,
            analysis_group_ids=filt,
        )
        n = write_metrics_csv(rows, args.out)

    logging.info("Wrote %s rows to %s", n, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
