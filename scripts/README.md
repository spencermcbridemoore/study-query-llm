# Scripts Directory

This directory contains standalone utility scripts for the Study Query LLM project.

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
  - Lists run groups in database
  - Status: Active utility

- **`azure_embeddings_smoke.py`** - Azure OpenAI embedding deployment smoke test
  - Tests embedding deployments from .env configuration
  - Status: Active

- **`mark_kllmeans_runs_defective.py`** - Data maintenance script
  - Marks PCA KLLMeans sweep runs as defective
  - Status: Active (maintenance tool)

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
4. **Data Maintenance**: `mark_kllmeans_runs_defective.py`
5. **Archive**: See `archive/` directory for deprecated scripts

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
