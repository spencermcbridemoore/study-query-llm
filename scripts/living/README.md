# Scripts Living Lane

This lane is reserved for actively maintained script entrypoints.

- `backfill_all_variant_clustering_analysis.py` — plan/execute registry-wide `pipeline.analyze` runs for a single `embedding_engine` (see `study_query_llm.experiments.clustering_analysis_backfill`).

During migration, many active scripts remain at the root `scripts/` path for command
stability. A later pass can relocate them here after wrapper coverage and doc parity
are validated.
