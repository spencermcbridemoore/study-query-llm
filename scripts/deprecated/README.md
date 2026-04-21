# Scripts Deprecated Lane

Use this lane for entrypoints that are fully retired and retained only for migration
context. Prefer replacement paths in `scripts/README.md` and runbooks.

At present, **thin** compatibility wrappers remain in the root `scripts/` directory
while consumers migrate to lane-based paths. Canonical implementations for move set
**v1.1** live here; root scripts forward via `runpy` to `scripts.deprecated.<module>`.

## Move set v1.1 (canonical under `scripts/deprecated/`)

- `analyze_dataset_lengths.py` — forwards to `scripts/history/analysis/`
- `analyze_dbpedia_character_length_grid.py`
- `analyze_estela_lengths.py`
- `plot_no_pca_50runs.py`
- `plot_no_pca_multi_embedding.py`
- `run_custom_full_categories_sweep.py`
- `run_experimental_sweep.py`
- `run_no_pca_50runs_sweep.py`
- `run_no_pca_multi_embedding_sweep.py`
- `test_no_pca_sweep.py`
- `migrate_v1_to_v2.py` — historical stub (exits non-zero)
- `pca_kllmeans_sweep.py` — forwards to `scripts/run_pca_kllmeans_sweep.py`

Prefer calling `scripts/history/...` directly for experiment reproduction; use
`scripts/deprecated/...` when you want the relocated file without the extra root hop.
