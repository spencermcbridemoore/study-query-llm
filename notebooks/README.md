# Notebooks Index

Status: living  
Owner: analytics-maintainers  
Last reviewed: 2026-06-01

Active working notebooks. Sweep *execution* now lives in tested `scripts/` and
the Panel app (`panel_app/`); the notebooks here are for embedding backfill and
result inspection only.

- `colab_embeddings.ipynb` — Embed-if-missing backfill of Estela prompts into the v2 DB from Colab.
- `pca_kllmeans_analysis.ipynb` — Analysis of PCA k-LLMeans sweep result pickles.
- `mcq_recent_1000_big_run_visualizer.ipynb` — QC visualizer for the v2 MCQ big-run groups.

`estela_prompt_data.pkl` is load-bearing input data (consumed by
`src/study_query_llm/experiments/` and the sweep scripts); do not remove it.
