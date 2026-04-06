# User Guide (V1 Legacy Setup)

Status: historical  
Owner: documentation-maintainers  
Last reviewed: 2026-04-06

This file preserves legacy v1 setup notes that were previously embedded in `docs/USER_GUIDE.md`.
Use only when you intentionally need v1 (`InferenceRepository` / `inference_runs`) behavior.

## Legacy v1 Database Initialization

```bash
python -c "from study_query_llm.db.connection import DatabaseConnection; from study_query_llm.config import config; db = DatabaseConnection(config.database.connection_string); db.init_db()"
```

## Legacy v1 Migration Context

Older docs and scripts that assume v1 may reference:

- `study_query_llm.db.connection.DatabaseConnection`
- `study_query_llm.db.inference_repository.InferenceRepository`
- table `inference_runs`

Current development defaults are v2 (`RawCallRepository` and related v2 models).
