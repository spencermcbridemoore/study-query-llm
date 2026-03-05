# Scripts Directory

This directory contains standalone utility and sweep runner scripts for the Study Query LLM project.

**Note:** The `scripts/common/` subdirectory has been removed. All shared library code
previously in `scripts/common/` now lives in canonical locations under `src/study_query_llm/`:

| Former `scripts/common/` module | New canonical location |
|---|---|
| Model managers (protocol, ACI, Docker, Ollama) | `study_query_llm.providers.managers` |
| `data_utils.py`, `estela_loader.py` | `study_query_llm.utils` |
| `embedding_utils.py` | `study_query_llm.services.embedding_helpers` |
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
