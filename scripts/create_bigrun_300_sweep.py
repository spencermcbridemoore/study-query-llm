"""
One-time retroactive script: register the 36 bigrun_300 runs under a
clustering_sweep group called "bigrun_300_feb2026".

Finds all clustering_run groups whose metadata run_key ends with '_300_50runs',
creates a single Group(type='clustering_sweep'), then links each run to it via
GroupLink(type='contains').

Idempotent: if a clustering_sweep named 'bigrun_300_feb2026' already exists
its run children are updated (missing links are added, nothing is duplicated).
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text as sa_text
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import ProvenanceService


SWEEP_NAME = "bigrun_300_feb2026"
ALGORITHM = "cosine_kllmeans_no_pca"
FIXED_CONFIG = {
    "n_samples": 300,
    "n_restarts": 50,
    "k_min": 2,
    "k_max": 20,
    "skip_pca": True,
    "distance_metric": "cosine",
    "normalize_vectors": True,
    "llm_interval": 20,
}
PARAMETER_AXES = {
    "datasets": ["dbpedia", "yahoo_answers", "estela"],
    "embedding_engines": ["embed-v-4-0", "text-embedding-3-large", "text-embedding-3-small"],
    "summarizers": ["None", "gpt-4o-mini", "gpt-4o", "gpt-5-chat", "claude-opus-4-6"],
}


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)

    with db.session_scope() as session:
        from study_query_llm.db.models_v2 import Group, GroupLink

        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        # ------------------------------------------------------------------ #
        # Find all bigrun_300 clustering_run groups                          #
        # ------------------------------------------------------------------ #
        runs = (
            session.query(Group)
            .filter(
                Group.group_type == "clustering_run",
                sa_text("metadata_json->>'run_key' LIKE :pat"),
            )
            .params(pat="%_300_50runs")
            .order_by(Group.id)
            .all()
        )

        print(f"Found {len(runs)} clustering_run groups matching '_300_50runs':")
        for r in runs:
            meta = r.metadata_json or {}
            print(
                f"  [{r.id}] {meta.get('dataset','?')} / "
                f"{meta.get('embedding_engine','?')} / "
                f"{meta.get('summarizer','?')}  (run_key={meta.get('run_key','?')})"
            )

        if not runs:
            print("Nothing to link — aborting.")
            return

        # ------------------------------------------------------------------ #
        # Find or create the clustering_sweep group                          #
        # ------------------------------------------------------------------ #
        existing_sweep = (
            session.query(Group)
            .filter(
                Group.group_type == "clustering_sweep",
                Group.name == SWEEP_NAME,
            )
            .first()
        )

        if existing_sweep:
            sweep_id = existing_sweep.id
            print(f"\nUsing existing clustering_sweep '{SWEEP_NAME}' (id={sweep_id})")
        else:
            sweep_id = provenance.create_clustering_sweep_group(
                sweep_name=SWEEP_NAME,
                algorithm=ALGORITHM,
                fixed_config=FIXED_CONFIG,
                parameter_axes=PARAMETER_AXES,
                description=(
                    "300-sample bigrun: 3 datasets x 3 embeddings x 5 summarizers, "
                    "50 restarts, cosine distance, no PCA, Feb 2026."
                ),
            )
            print(f"\nCreated clustering_sweep '{SWEEP_NAME}' (id={sweep_id})")

        # ------------------------------------------------------------------ #
        # Link runs to sweep (idempotent: skips existing links)              #
        # ------------------------------------------------------------------ #
        linked = 0
        skipped = 0
        for pos, run in enumerate(runs):
            existing_link = (
                session.query(GroupLink)
                .filter_by(
                    parent_group_id=sweep_id,
                    child_group_id=run.id,
                    link_type="contains",
                )
                .first()
            )
            if existing_link:
                skipped += 1
                continue

            provenance.link_run_to_clustering_sweep(sweep_id, run.id, position=pos)
            linked += 1

        print(f"\nLinked {linked} runs to sweep (skipped {skipped} already linked).")

        # ------------------------------------------------------------------ #
        # Update n_runs count in sweep metadata                              #
        # ------------------------------------------------------------------ #
        sweep_group = repo.get_group_by_id(sweep_id)
        if sweep_group and sweep_group.metadata_json:
            sweep_group.metadata_json = {
                **sweep_group.metadata_json,
                "n_runs": len(runs),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            session.flush()

        print(f"\n[OK] clustering_sweep '{SWEEP_NAME}' (id={sweep_id}) now covers {len(runs)} runs.")


if __name__ == "__main__":
    main()
