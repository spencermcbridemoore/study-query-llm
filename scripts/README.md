# Scripts Directory

This directory contains standalone utility and sweep runner scripts for the Study Query LLM project.

**Note:** The `scripts/common/` subdirectory has been removed. All shared library code
previously in `scripts/common/` now lives in canonical locations under `src/study_query_llm/`:

| Former `scripts/common/` module | New canonical location |
|---|---|
| Model managers (protocol, ACI, Docker, Ollama) | `study_query_llm.providers.managers` |
| `data_utils.py`, `estela_loader.py` | `study_query_llm.utils` |
| `embedding_utils.py` | `study_query_llm.services.embeddings` (`EmbeddingService`, `fetch_embeddings_async`, file cache helpers) |
| `sweep_utils.py` (serialization/save) | `study_query_llm.experiments.sweep_io` |
| `sweep_utils.py` (paraphraser factory) | `study_query_llm.services.paraphraser_factory` |
| `sweep_utils.py` (DB ingestion) | `study_query_llm.experiments.ingestion` |
| `sweep_utils.py` (metric helpers) | `study_query_llm.experiments.result_metrics` |

## Script Status

### ✅ Active Scripts
Scripts that are currently maintained, tested, and actively used:

- **`docker_smoke.py`** - Docker deployment smoke testing
  - Tests Docker stack locally
  - Referenced in `docs/DEPLOYMENT.md`
  - Status: Active

- **`run_pca_kllmeans_sweep.py`** - PCA/KLLMeans sweep analysis runner
  - Runs analysis using algorithms library and service layer
  - Can be run directly or in Jupyter
  - Status: Active

- **`pca_kllmeans_sweep.py`** - Legacy wrapper for PCA/KLLMeans sweep
  - Now uses algorithm core library from `src/study_query_llm/algorithms`
  - Status: Active (legacy wrapper, but functional)

- **`check_all_data.py`** - Database data inspection utility
  - Checks all data in v2 database
  - Status: Active utility

- **`check_db_empty.py`** - Quick database empty check
  - Verifies if database is empty
  - Status: Active utility

- **`backup_mcq_db_to_json.py`** - Export MCQ-related v2 rows to `scratch/mcq_db_backups/`
  - Uses `LOCAL_DATABASE_URL` or `DATABASE_URL`; writes `*_full.json` + `*_summary.json` with embedded backup metadata
  - Output path is gitignored; may contain prompts — do not commit
  - Status: Active utility

- **`check_embedding_calls.py`** - Embedding calls inspection
  - Lists embedding calls in database
  - Status: Active utility

- **`check_run_groups.py`** - Run groups inspection
  - Lists run groups in database and summarizes run_key quality (missing/duplicates)
  - Status: Active utility

- **`audit_last_partial_sweep.py`** - Last partial sweep migration audit
  - Read-only audit: duplicate `run_key`, missing `run_key`, duplicate group links
  - Selects newest partial request candidate for targeted migration
  - Status: Active utility

- **`reconcile_last_partial_sweep.py`** - Single-request reconciliation/finalize
  - Targets one partial `clustering_sweep_request`, backfills request->run links
  - Optionally ingests PKL artifacts, then finalizes if request is fulfilled
  - Status: Active utility

- **`create_dataset_snapshots_286.py`** - Create frozen 286-entry dataset snapshots
  - Creates/reuses `dataset_snapshot` groups for dbpedia (labeled) and estela (unlabeled)
  - Stores exact sample manifests as `dataset_snapshot_manifest` artifacts
  - Status: Active utility

- **`validate_and_backfill_run_snapshots.py`** - Snapshot linkage validator/backfill
  - Validates `clustering_run` snapshot linkage and optionally backfills missing links
  - Writes `dataset_snapshot_ids` metadata and `depends_on` links with `--apply`
  - Status: Active utility

- **`run_local_300_2datasets_worker.py`** - Request worker for local 2-dataset sweep
  - Supports legacy self-managed TEI mode and shared-endpoint mode
  - Shared mode flags: `--embedding-engine`, `--tei-endpoint`, `--idle-exit-seconds`
  - Job mode flags: `--job-mode standalone|sharded` for run-key claims vs job-table claims
  - Status: Active runner

- **`run_local_300_2datasets_engine_supervisor.py`** - One-container-per-engine supervisor
  - Starts one TEI container per embedding engine and launches worker pool
  - Advances to next engine only when current engine `missing_run_keys` reaches zero
  - Includes guardrails: TEI health polling, bounded worker/container restarts, backoff
  - Supports `--job-mode sharded` to track per-engine progress from orchestration jobs
  - Status: Active runner

- **`run_cached_job_supervisor.py`** - Cached-job supervisor (single DB client, queue workers)
  - For one (request_id, engine): fetches batches of run_k_try jobs, distributes via multiprocessing queues to N workers, batch-completes (repository performs promote), runs reduce_k/finalize_run in-process.
  - **When to use:** High worker count (e.g. 32–64), single engine per run. Reduces DB contention vs sharded workers that each claim/complete.
  - **Required env vars:** `DATABASE_URL` (and provider-specific vars if using `--embedding-provider`).
  - **Example:** `python scripts/run_cached_job_supervisor.py --request-id N --worker-count 32 --engine "Qwen/Qwen3-Embedding-0.6B" --provider-label local_docker_tei_shared --tei-endpoint http://localhost:8080/v1`
  - Workers use DB only for read-only L3/L2 embedding cache; claim/complete/promote are done by the supervisor.
  - Status: Active runner

- **`check_orchestration_jobs.py`** - Job-table inspection utility
  - Summarizes orchestration jobs by type/status for a request
  - Useful for sharded execution diagnostics (`run_k_try`, `reduce_k`, `finalize_run`)
  - Status: Active utility

- **`azure_embeddings_smoke.py`** - Azure OpenAI embedding deployment smoke test
  - Tests embedding deployments from .env configuration
  - Status: Active

- **`mark_kllmeans_runs_defective.py`** - Data maintenance script
  - Marks PCA KLLMeans sweep runs as defective
  - Status: Active (maintenance tool)

- **`label_pre_fix_runs.py`** - Label pre-centroid-fix clustering runs
  - Sets `metadata_json["centroid_fix_era"] = "pre_fix"` on all existing clustering_run groups
  - Allows downstream tools (check scripts, Panel UI) to exclude pre-fix data
  - Supports `--dry-run`
  - Prereqs: `DATABASE_URL`
  - Status: Active (maintenance tool)

- **`archive_pre_fix_runs.py`** - Archive pre-fix clustering runs to local DB
  - Copies clustering_run + clustering_step groups and GroupLinks to local backup DB, then deletes from Neon
  - Run `label_pre_fix_runs.py` first to mark runs
  - Supports `--dry-run`
  - Prereqs: `DATABASE_URL`, `LOCAL_DATABASE_URL`, local DB initialized via `init_local_db.py`
  - Status: Active (maintenance tool)

- **`check_summarizer_results_differ.py`** - Summarizer vs None validation
  - Checks that LLM summarizer runs produce different results from None runs
  - Excludes pre-fix runs (those with `centroid_fix_era = "pre_fix"`)
  - Status: Active (validation tool)

### ⚠️ Needs Verification
Scripts that may still work but need testing:

- **`embedding_cluster_stability.py`** - Embedding cluster stability analyzer
  - Large analysis script (~900 lines)
  - May have been superseded by algorithms library
  - Status: Unknown - needs verification if still used

- **`test_validation_logic.py`** - Validation logic testing
  - Tests SummarizationService validation logic
  - Status: Unknown - may be obsolete or for debugging only

### 📦 Archive (Deprecated/One-time)
Scripts that were used for one-time tasks or are no longer maintained:

- **`migrate_v1_to_v2.py`** - V1 to V2 database migration
  - One-time migration script (completed)
  - Kept for reference
  - Status: Archived

- **`verify_migration.py`** - Migration verification
  - Verified v1 → v2 migration
  - One-time use (completed)
  - Status: Archived

- **`test_deployment_completion.py`** - Legacy deployment testing
  - Tests text completion with specific deployments
  - May be obsolete (superseded by other testing)
  - Status: Archived

## Usage

### Running Scripts

Most scripts require environment variables to be set. Load from `.env` file:

```bash
# From project root
python scripts/script_name.py
```

Or set environment variables directly:

```bash
DATABASE_URL=postgresql://... python scripts/check_all_data.py
```

### Script Categories

1. **Deployment/Testing**: `docker_smoke.py`, `azure_embeddings_smoke.py`
2. **Analysis**: `run_pca_kllmeans_sweep.py`, `pca_kllmeans_sweep.py`
3. **Database Utilities**: `check_*.py` scripts
4. **Data Maintenance**: `mark_kllmeans_runs_defective.py`, `label_pre_fix_runs.py`, `archive_pre_fix_runs.py`
5. **Validation**: `check_summarizer_results_differ.py`
6. **Archive**: See `archive/` directory for deprecated scripts

## Adding New Scripts

When adding new scripts:

1. Add a clear docstring at the top explaining purpose
2. Include usage examples in docstring
3. Update this README with script status
4. Use `encoding='utf-8'` for file operations (Windows compatibility)
5. Follow project coding conventions (see `.cursorrules`)

## Deprecation Policy

- Scripts that are no longer needed should be moved to `archive/` directory
- Add deprecation warnings to scripts that may not work with current codebase
- Document deprecation status in this README
- Keep archived scripts in git history for reference
