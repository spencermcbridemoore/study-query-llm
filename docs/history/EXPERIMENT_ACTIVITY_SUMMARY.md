# Experiment Activity Summary (Informational Only)

> **Important scope note:** this document is a historical, evidence-based summary only.  
> It is **not** a source of truth for architecture, implementation, release decisions, or codebase management.

## Purpose

This file records what experiments appear to have been run (or planned/discussed) based on:
- repository artifacts (notebooks/scripts/config patterns)
- prior agent-transcript evidence
- user-provided context in chat

It is intended as a human-readable recap, not an operational runbook.

## Additional Evidence Pass (Artifact Inventory)

An additional artifact-only pass was completed across:
- `experimental_results`
- `logs`
- `notebooks`
- `data`

High-level inventory from that pass:
- `experimental_results/job_shards`: `10054` JSON
- `experimental_results/mcq_answer_position_probe`: `1873` JSON
- `experimental_results/_defective`: `453` PKL
- `experimental_results` top-level: `516` PKL, `23` CSV, `3` JSON
- `experimental_results/benchmark_logs`: `310` LOG
- `logs/supervisor`: `1` LOG (`stage_a_20260305_005957.log`)

## Paper Context You Referenced

You indicated this work was modeled on:
- *Summaries as Centroids for Interpretable and Scalable Text Clustering* (Jairo Diaz-Rodriguez)

The relevant method pattern (as reflected in your repo direction) is summary-as-centroid clustering (`k-LLMmeans` style), with periodic summary/re-embed updates in the clustering loop.

## What You Experimented With (Evidence-Based Reconstruction)

## 1) No-PCA 50-restart clustering sweeps

- Pattern: `k=2..20`, `n_restarts=50`, no PCA, cosine/normalized settings
- Strong evidence in: `scripts/run_no_pca_50runs_sweep.py`
- Confirmed retained artifacts: `6` top-level PKLs in `experimental_results/no_pca_50runs_*.pkl`
- Confirmed dataset in retained PKLs: `dbpedia`
- Confirmed summarizers in retained PKLs: `None`, `gpt-5-chat`
- Related analysis output families present in `experimental_results/plots/no_pca_50runs/*`

## 2) Local GPU multi-engine sweep with ~300 samples and 3 datasets

- Pattern: broad local-embedding sweep using many engines and multiple summarizers
- Evidence in notebook lineage:
  - `notebooks/archive/local_gpu_300_sweep.ipynb`
  - `notebooks/local_300_3datasets_14engines_4summarizers.ipynb` (historical variant naming/content drift appears in outputs)
- Dataset pattern in this family: `dbpedia`, `yahoo_answers`, `estela`
- Sweep parameters commonly shown: `k=2..20`, 50 restarts
- Concrete retained run family also visible as `bigrun_300_*.pkl` (`36` PKLs: 3 datasets x 3 embedding deployments x 4 summarizers)

## 3) Local GPU multi-sample sweep (100/500/1000) for 2 datasets

- Pattern: run only `dbpedia` + `yahoo_answers` across sample sizes `100`, `500`, `1000`
- Evidence in:
  - `notebooks/archive/local_gpu_multi_sweep.ipynb`
  - transcript evidence indicating target of 420 combinations (2 datasets x 3 sample sizes x 14 engines x 5 summarizers)
- This appears to be an intentional scaling experiment on sample-size sensitivity.
- Artifact evidence includes:
  - retained top-level `local_gpu_multi_*.pkl`: `140`
  - archived/quarantined `_defective/local_gpu_multi_*.pkl`: `289`

## 4) Ingestion and sweep-group materialization

- Pattern: PKL output ingestion into DB and linking runs to sweep groups
- Evidence in:
  - `scripts/ingest_sweep_to_db.py`
  - transcript evidence of sweep-group names such as `local_gpu_300_feb2026` and `local_gpu_multi_mar2026`
- Additional run-worker evidence in `experimental_results/job_shards`:
  - `9203` shard JSON (`rq*`)
  - `777` reduce jobs (`reduce_k_job_*`)
  - `74` finalize jobs (`finalize_run_job_*`)
  - observed saved/reduced timestamps span roughly `2026-03-05` to `2026-03-07`

## 5) Validation and defect-discovery experiments

- You explicitly checked whether summarizer runs changed clustering outcomes vs `None`.
- Evidence indicates two major discovered issues:
  1. LLM update path blocked for `skip_pca=True` due to guard logic.
  2. Local TEI single-model behavior caused "different engine" runs to return effectively identical embeddings when not managed per engine container.
- Resulting corrective activity (per transcript evidence) included:
  - algorithm guard fix path discussions
  - defective run segregation/cleanup
  - DB cleanup/re-ingest cycles
  - notebook refactor toward per-engine TEI management
- Artifact corroboration:
  - `_defective` archive contains `453` quarantined PKLs
  - `experimental_results/plots/invalid_presummarization/README.md` documents archived invalid plots from pre-fix conditions

## 6) Request/fulfillment workflow experiments

- You also iterated on "sweep as order/request" lifecycle concepts and implementation.
- Evidence in:
  - `src/study_query_llm/services/sweep_request_service.py`
  - `src/study_query_llm/experiments/sweep_worker_main.py`
  - related scripts and tests under `tests/test_services/`
- This appears to be a reliability/operations layer on top of experiment execution.

## 7) Broader PCA/label-max sweep family (pre/local mixed)

- Top-level `experimental_sweep_*.pkl` retained artifacts: `314`
- Datasets visible in retained files:
  - `20newsgroups_10cat`
  - `20newsgroups_6cat`
  - `dbpedia`
  - `news_category`
  - `yahoo_answers`
- Typical metadata pattern includes `entry_max`, `label_max`, `sweep_config`, embedding deployment, and summarizer.

## 8) MCQ answer-position probe campaign

- Strong artifact evidence in `experimental_results/mcq_answer_position_probe/*.json` (`1873` files)
- Observed date windows by filename: `2026-03-16` through `2026-03-30`
- Aggregated from file summaries:
  - total requested samples: `96853`
  - successful calls: `63563`
  - valid-answer samples: `62223`
  - total parsed answers: `1078973`
- `summary.provider` explicitly set to `openrouter` in a large subset (`1079` files); older files often infer provider only from deployment/model naming.

## 9) MCQ sweep export/aggregation runs

- Evidence in:
  - `experimental_results/mcq_db_option_counts/*.json` and `.csv`
  - top-level `experimental_results/mcq_sweep_*.csv`
- Exported sweep ids observed: `144`, `165`, `306`, `447`, `507`
- Includes SQL-derived fields (`sweep_id`, `run_group_id`, `run_key`) indicating DB-backed source materialization.

## 10) Benchmark and throughput probes

- `experimental_results/benchmark_logs`: `310` logs (`azbench`, `bench20leaf`, `bench`)
- Top-level benchmark CSV summaries (`23` total CSVs) include worker-scaling and throughput comparisons.
- Additional utility probes:
  - `openrouter_concurrency_probe_20260330.json`
  - `openrouter_concurrency_probe_200x60_50-100tok_20260330.json`
  - `openrouter_model_verify_mcq_candidates.json`
  - `langgraph_outputs/lg_smoke_1_1.json` and `lg_redact_1_1.json`

## Dataset-Specific Summary (What Was Actually Run vs Discussed)

- **DBpedia:** strong evidence of repeated execution.
- **Estela:** strong evidence of repeated execution (especially in 3-dataset and later 2-dataset paths that retained estela).
- **Yahoo Answers:** strong evidence of repeated execution in several sweep variants.
- **20 Newsgroups / News Category:** strong evidence in `experimental_sweep` retained PKLs.
- **Bank77:** discussed as candidate/benchmark reference; no clear evidence here of completed run pipelines in this repo history.
- **GoEmotion:** discussed in context of the paper/dataset landscape; no clear evidence here of completed run pipelines in this repo history.

## Storage Target Findings (Artifact-Evidenced)

- **NeonDB (`neondb`)**: strongly evidenced in notebook/log output strings (including supervisor and benchmark logs).
- **SQLite**: explicitly configured in `notebooks/colab_setup.ipynb` (`sqlite:///study_query_llm.db`) for Colab setup flow.
- **Local PostgreSQL**: no direct URI evidence found in the searched artifact folders from this pass.

## Confidence and Limits

- **High confidence:** artifact counts/date windows and major families above (derived from concrete files).
- **High confidence:** storage-target evidence for `neondb` and `sqlite` in searched artifacts.
- **Medium confidence:** exact run completeness for each intended factorial grid (partial runs/retries/cleanup can shift totals).
- **Not claimed:** this is not a mathematically complete ledger of every artifact ever produced.

## Primary Evidence Anchors

- `scripts/run_no_pca_50runs_sweep.py`
- `scripts/ingest_sweep_to_db.py`
- `notebooks/archive/local_gpu_300_sweep.ipynb`
- `notebooks/archive/local_gpu_multi_sweep.ipynb`
- `notebooks/local_300_2datasets_14engines_4summarizers.ipynb`
- `src/study_query_llm/services/sweep_request_service.py`
- `src/study_query_llm/experiments/sweep_worker_main.py`
- `experimental_results/job_shards/*`
- `experimental_results/mcq_answer_position_probe/*`
- `experimental_results/mcq_db_option_counts/*`
- `experimental_results/benchmark_logs/*`
- `logs/supervisor/stage_a_20260305_005957.log`
- `docs/history/EXPERIMENT_ATTEMPT_TABLE.md`

