# Subagent 2 — DB Write-Path Inventory (raw)

Date: 2026-04-24
Scope: every code path that writes to a database (insert/update/delete/DDL/migration), classified by intended lane.
Method: readonly explore subagent.

**Lane labels:** `CANONICAL` = intended Jetstream/production truth (not enforced in code); `READ_MIRROR` = explicit local clone / `LOCAL_DATABASE_URL` target; `SANDBOX` = tests / explicit `:memory:` / scratch; `AMBIGUOUS` = target is whatever URL is passed or `DATABASE_URL` / `config.database.connection_string` without verifying Jetstream vs local.

---

## A. Inventory table (write sites)

### Core connection / DDL

| `file:line` | tables / effect | lane | session source | preflight? | classification |
|-------------|-----------------|------|----------------|------------|----------------|
| `src/study_query_llm/db/_base_connection.py:127-131` | `CREATE TABLE` via `MetaData.create_all` | AMBIGUOUS | `DatabaseConnectionV2` / `BaseDatabaseConnection` engine from constructor URL | none | `init_db` DDL |
| `src/study_query_llm/db/_base_connection.py:183-196` | commits ORM transaction | AMBIGUOUS | `session_scope` → `get_session()` → `SessionLocal()` | none | auto-commit wrapper |
| `src/study_query_llm/db/_base_connection.py:174-177` | `DROP` via `drop_all` | SANDBOX / guarded | same | `_assert_destructive_operation_allowed` at `133-172` | destructive DDL guard (`SQLLM_ALLOW_DESTRUCTIVE_DDL`, Jetstream match block) |
| `src/study_query_llm/db/connection_v2.py:43-60` | `CREATE EXTENSION IF NOT EXISTS vector`; then `create_all` | AMBIGUOUS | engine from `DatabaseConnectionV2(connection_string)` | pgvector try/except only | `init_db` DDL |

### `RawCallRepository` (central write implementation)

| `file:line` | tables | lane | session | preflight? | classification |
|-------------|--------|------|---------|------------|----------------|
| `raw_call_repository.py:109-118` | `raw_calls` INSERT | AMBIGUOUS | `self.session` from ctor | none | `insert_raw_call` |
| `raw_call_repository.py:146-149` | `raw_calls` bulk INSERT | AMBIGUOUS | same | none | `batch_insert_raw_calls` |
| `raw_call_repository.py:586-592` | `groups` INSERT | AMBIGUOUS | same | none | `create_group` |
| `raw_call_repository.py:630-632` | `group_members` INSERT | AMBIGUOUS | same | dedup query `614-621` | `add_call_to_group` |
| `raw_call_repository.py:876-902` | `embedding_cache_entries` UPSERT/INSERT | AMBIGUOUS | same | lookup `870-874` | `upsert_embedding_cache_entry` |
| `raw_call_repository.py:905-918` | `embedding_cache_entries` UPDATE | AMBIGUOUS | same | none | `touch_embedding_cache_hit` |
| `raw_call_repository.py:947-948,951-955` | `embedding_cache_leases` INSERT/UPDATE | AMBIGUOUS | same | lease steal logic | `try_acquire_embedding_cache_lease` |
| `raw_call_repository.py:971-972` | `embedding_cache_leases` DELETE | AMBIGUOUS | same | owner check | `release_embedding_cache_lease` |
| `raw_call_repository.py:1031-1032` | `group_links` INSERT | AMBIGUOUS | same | existing link check `1006-1017` | `create_group_link` |
| `raw_call_repository.py:1184-1198` | `orchestration_jobs`, `orchestration_job_dependencies` INSERT | AMBIGUOUS | same | idempotent lookup `1161-1163` | `enqueue_orchestration_job` |
| `raw_call_repository.py:1243-1250` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | dependency gate | `claim_next_orchestration_job` |
| `raw_call_repository.py:1306-1313` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | same | `claim_orchestration_job_batch` |
| `raw_call_repository.py:1335-1338` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | status check | `heartbeat_orchestration_job` |
| `raw_call_repository.py:1348-1353` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | none | `complete_orchestration_job` |
| `raw_call_repository.py:1372-1380` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | none | `complete_orchestration_jobs_batch` |
| `raw_call_repository.py:1394-1401` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | attempt cap | `fail_orchestration_job` |
| `raw_call_repository.py:1409-1413` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | none | `release_orchestration_job` |
| `raw_call_repository.py:1425-1427` | `orchestration_job_dependencies` INSERT | AMBIGUOUS | same | dedup `1417-1423` | `add_orchestration_job_dependency` |
| `raw_call_repository.py:1451-1455` | `orchestration_jobs` UPDATE | AMBIGUOUS | same | readiness | `promote_ready_orchestration_jobs` |
| `raw_call_repository.py:1529-1596` | `provenanced_runs` UPDATE or INSERT | AMBIGUOUS | same | upsert key `1518-1528` | `create_provenanced_run` |
| `raw_call_repository.py:1689-1700` | `provenanced_runs` UPDATE | AMBIGUOUS | same | none | `update_provenanced_run` |

### Services (delegate to repository / session)

| `file:line` | tables | lane | session | preflight? | classification |
|-------------|--------|------|---------|------------|----------------|
| `method_service.py:123-140` | `method_definitions` UPDATE+INSERT | AMBIGUOUS | `self.repository.session` | deactivate prior `115-124` | `register_method` |
| `method_service.py:179-180` | `method_definitions` UPDATE | AMBIGUOUS | same | row existence | `update_recipe` |
| `method_service.py:263-265` | `analysis_results` INSERT | AMBIGUOUS | same | none | `record_result` |
| `provenance_service.py:139-144,177-182,237,281-286,469-474,529-534,607-612,646-651,685-690` | `groups` INSERT | AMBIGUOUS | `repository.session` | none | `create_*_group` variants |
| `provenance_service.py:314-319` | `group_members` INSERT | AMBIGUOUS | same | via repo dedup | `link_raw_calls_to_group` |
| `provenance_service.py:367-368` | `call_artifacts` INSERT | AMBIGUOUS | same | none | `link_artifacts_to_group` |
| `provenance_service.py:543-548,561-568,712-717,729-733` | `group_links` INSERT | AMBIGUOUS | same | various | link helpers |
| `artifact_service.py:434-463` | `raw_calls` (optional) + `call_artifacts` | AMBIGUOUS | `repository.session` | optional placeholder call | `_link_artifact_to_group` |
| `sweep_request_service.py:634-636,721-738,913-924,972-989,1051-1062,1073-1085,1107-1115,1142-1187` | `groups` + `group_links` + metadata | AMBIGUOUS | `repository.session` | request type checks | request lifecycle |
| `provenanced_run_service.py:196-213,248-264` | `provenanced_runs` via repo | AMBIGUOUS | `repository.session` | fingerprint | `record_*_execution` |
| `data_quality_service.py:67-71` | `groups` INSERT | AMBIGUOUS | `repository.session` | lookup `58-64` | `get_or_create_defective_group` |
| `inference_service.py:203-219,510-515,576-593` | `raw_calls`, `groups`, `group_members` | AMBIGUOUS | `self.repository` session | none | inference write paths |
| `summarization_service.py:208-224,242-258` | `raw_calls` INSERT | AMBIGUOUS | `self.repository` | none | `_log_summarization_result` |
| `embeddings/persistence.py:46-57,106-116,187-197,120-133,199-212` | `raw_calls`, `embedding_cache_entries` | AMBIGUOUS | `repository` | none | embedding persistence |
| `langgraph_provenance.py:169-198` | `method_definitions`, `analysis_results`, `provenanced_runs` | AMBIGUOUS | via service repos | method resolve | `record_langgraph_job_outcome` |

### Pipeline (`src/study_query_llm/pipeline/`)

| `file:line` | tables / effect | lane | session | preflight? | classification |
|-------------|-----------------|------|---------|------------|----------------|
| `pipeline/runner.py:98-136` | `groups`, `group_links`, `provenanced_runs` | AMBIGUOUS | `db.session_scope()` | none | `run_stage` claim |
| `pipeline/runner.py:163-177` | `provenanced_runs` UPDATE | AMBIGUOUS | `db.session_scope()` | none | success/failure finalize |
| `pipeline/acquire.py:109-154` | `groups` + blob-backed `call_artifacts` | AMBIGUOUS | `db_conn.session_scope()` | stage validation | acquire stage |
| `pipeline/parse.py:232-323` | `groups` + `call_artifacts` | AMBIGUOUS | same | stage validation | parse stage |
| `pipeline/embed.py:133-217` | `groups` + `call_artifacts` | AMBIGUOUS | same | stage validation | embed stage |
| `pipeline/snapshot.py:228-303` | `groups` + `call_artifacts` | AMBIGUOUS | same | stage validation | snapshot stage |
| `pipeline/analyze.py:651-836` | `groups` + `call_artifacts` + `analysis_results` | AMBIGUOUS | same | stage validation | analyze stage |

### Experiments / workers

| `file:line` | tables | lane | session | preflight? | classification |
|-------------|--------|------|---------|------------|----------------|
| `experiments/ingestion.py:102-348` | `groups`, `call_artifacts`, `group_links`, `method_definitions`, `analysis_results`, `provenanced_runs` | AMBIGUOUS | `db.session_scope()` | run_key existence `106-112,141-154` | `ingest_result_to_db` |
| `experiments/sweep_worker_main.py:217-249,265-293` | `sweep_run_claims` INSERT/UPDATE | AMBIGUOUS | `db.session_scope()` | concurrency | `_claim_run_target` / complete |
| `experiments/runtime_sweeps.py:116-151,172-215` | `sweep_run_claims` | AMBIGUOUS | `db.session_scope()` | same pattern | claim helpers |
| `experiments/mcq_run_persistence.py:68-124` | `groups`, `method_definitions`, `provenanced_runs`, `group_links` | AMBIGUOUS | `db.session_scope()` | none | `persist_mcq_probe_result` |
| `services/jobs/runtime_workers.py:27-34,115-176` | `orchestration_jobs` + provenance tables | AMBIGUOUS | `db.session_scope()` | claim | `claim_next_langgraph_job` / worker |
| `services/jobs/job_reducer_service.py:61-111,115-168` | `orchestration_jobs` + full sweep ingestion | AMBIGUOUS | `self.db.session_scope()` | none | `reduce_k_job` / `finalize_run_job` |
| `services/jobs/runtime_supervisors.py:72-114,199-218,360-368` | `orchestration_jobs` etc. | AMBIGUOUS | `DatabaseConnectionV2(os.environ["DATABASE_URL"])` | none | supervisors |

### Panel app

| `file:line` | tables | lane | session | preflight? | classification |
|-------------|--------|------|---------|------------|----------------|
| `panel_app/views/inference.py:22-26` | `raw_calls` UPDATE | AMBIGUOUS | passed `session` | none | `_source_tag` |
| `panel_app/views/inference.py:37-41` | `groups` INSERT | AMBIGUOUS | same | lookup `31-35` | `_find_or_create_run` |
| `panel_app/views/inference.py:226-244` | `raw_calls` + `group_members` + `groups` | AMBIGUOUS | `get_db_connection().session_scope()` | UI | standalone inference |
| `panel_app/views/inference.py:417-434` | `groups`, `group_members`, `group_links` | AMBIGUOUS | same | UI | batch inference |

### Scripts — ingestion / maintenance / sync

| `file:line` | tables | lane | session | preflight? | classification |
|-------------|--------|------|---------|------------|----------------|
| `scripts/ingest_sweep_to_db.py:266-303,420-449` | `groups`, `group_links` | AMBIGUOUS | `repository.session` inside `db.session_scope()` | idempotency `240-246,415-418` | PKL/artifact ingest |
| `scripts/ingest_sweep_to_db.py:487-488` | DDL `init_db` | AMBIGUOUS | `DatabaseConnectionV2(config.database.connection_string)` | none | uses `config` |
| `scripts/ingest_mcq_probe_json_to_sweep_db.py:206-244` | `groups`, `group_links` | AMBIGUOUS | `db.session_scope()` | run_key pre-scan `155-193` | MCQ ingest |
| `scripts/register_clustering_methods.py:71-120` | `method_definitions` | AMBIGUOUS | `db.session_scope()` | `--dry-run` rollback `97-98` | method registration |
| `scripts/register_text_classification_methods.py:73` | DDL `init_db` | AMBIGUOUS | `DATABASE_URL` | uncertain | full body not re-read; may call `MethodService` below |
| `scripts/backfill_run_fingerprints.py:92-100` | `provenanced_runs` UPDATE | AMBIGUOUS | `db.session_scope()` `50-100` | `--dry-run` | fingerprint backfill |
| `scripts/validate_and_backfill_run_snapshots.py:169-174` | `groups` + `group_links` | AMBIGUOUS | `db.session_scope()` `114-174` | eligibility filters | `--apply` backfill |
| `scripts/create_bank77_contrast_snapshots.py:173-174` | `groups.name` UPDATE | AMBIGUOUS | `db.session_scope()` `166-174` | hash checks `140-156` | rename snapshot |
| `scripts/history/sweep_recovery/label_pre_fix_runs.py:107-114` | `groups.metadata_json` | AMBIGUOUS | `DatabaseConnectionV2(config.database.connection_string)` `47-48` | `--dry-run` | bulk metadata |
| `scripts/purge_dataset_acquisition.py:217-220` | `call_artifacts`, `raw_calls`, `groups` DELETE | AMBIGUOUS (URL chosen by operator) | ad-hoc `sessionmaker` `161-163` | `--execute` requires explicit `--database-url` `138-144`; loopback/remote `145-150`; name confirm `204-211` | destructive |
| `scripts/sync_from_online.py:172-218` | `groups`, `raw_calls`, `group_members`, `call_artifacts` | **READ_MIRROR** (local target) | `sessionmaker(local_engine)` `155-156` | source≠target `269-274`; loopback target `276-281` | mirror sync |
| `scripts/archive_defective_data.py:71-155,217-243` | local INSERT; online DELETE `raw_calls` | CANONICAL delete / READ_MIRROR copy **if** env interpreted as Neon+local | dual `session_scope` `210-217,233-243` | `--dry-run` | destructive on online session |
| `scripts/history/sweep_recovery/archive_pre_fix_runs.py:67-146,271-290` | `groups`, `group_links` | dual-URL pattern as archive | `online_db` / `local_db` | `--dry-run` | destructive on online |
| `scripts/history/one_offs/migrate_group_types_to_clustering.py:62-104` | `groups`, `group_links` UPDATE | AMBIGUOUS | `db.engine.connect()` `34` | pre-count | one-off SQL |
| `scripts/history/one_offs/create_bigrun_300_sweep.py:102-148` | `groups`, `group_links`, metadata | AMBIGUOUS | script session | link idempotency `120-131` | sweep linker |

### Migrations (`src/study_query_llm/db/migrations/`)

| `file:line` | effect | lane | URL | preflight? | classification |
|-------------|--------|------|-----|------------|----------------|
| `add_provenanced_runs_table.py:26-36` | `CREATE TABLE provenanced_runs` | AMBIGUOUS | `DATABASE_URL` | none | DDL |
| `add_group_links.py:28-42` | `CREATE TABLE group_links` | AMBIGUOUS | `DATABASE_URL` | none | DDL |
| `add_method_analysis_tables.py:29-46` | `method_definitions`, `analysis_results` | AMBIGUOUS | `DATABASE_URL` | none | DDL |
| `add_sweep_request_indexes.py:33-65` | `CREATE INDEX IF NOT EXISTS` | AMBIGUOUS | `DATABASE_URL` | try/except per index | DDL |
| `add_sweep_worker_safety.py:79-120` | duplicate checks; `CREATE UNIQUE INDEX`; `sweep_run_claims` table | AMBIGUOUS | `DATABASE_URL` | `_require_no_duplicates` `31-76` | DDL + safety |
| `add_fingerprint_columns.py:48+` | `ALTER TABLE provenanced_runs` (`text("ALTER TABLE...")` `64-74`) | AMBIGUOUS | `DATABASE_URL` | none | DDL |
| `add_recipe_json_column.py:82-114` | `init_db` + `ALTER TABLE method_definitions` | AMBIGUOUS | `DATABASE_URL` | comments on table existence | DDL |
| `normalize_provenanced_run_kind_execution.py:46-78` | ORM UPDATE + `ALTER TABLE ... DROP/ADD CONSTRAINT` | AMBIGUOUS | `DATABASE_URL` | filters rows | **destructive constraint change** |
| `drop_embedding_vectors.py:43-67` | `DROP TABLE embedding_vectors` | AMBIGUOUS | `DATABASE_URL` | row count log | **destructive** |

### Tests (`tests/`)

| `file:line` | tables | lane | session | preflight? | classification |
|-------------|--------|------|---------|------------|----------------|
| `tests/test_db/test_repository.py` etc. | v1 `inference_runs` | SANDBOX | `sqlite:///:memory:` fixture `tests/test_db/test_connection.py:11-13` | n/a | unit |
| `tests/test_db/test_repository_v2.py:20-22` | all v2 tables | SANDBOX | fixture + `drop_all_tables` | guard tests in `test_destructive_guard.py` | integration |
| `tests/pipeline/*.py`, `tests/test_services/*.py` | v2 tables | SANDBOX | per-file `DatabaseConnectionV2("sqlite:///:memory:")` e.g. `tests/test_integration/test_combined_sweep_integration.py:112-113` | n/a | tests |
| `tests/test_scripts/test_ingest_sweep_to_db.py:37,56,133,214` | subprocess env `DATABASE_URL` → temp sqlite | SANDBOX | child process | n/a | CLI tests |

### Deprecated / legacy

| `file:line` | tables | lane | classification |
|-------------|--------|------|----------------|
| `db/inference_repository.py:92-94,125-126` | `inference_runs` | AMBIGUOUS if session target not sqlite | V1 deprecated |

---

## B. Central writers (many callers)

- **`RawCallRepository`** (`raw_call_repository.py:65-1700`): single hub for `raw_calls`, `groups`, `group_members`, `group_links`, `embedding_cache_*`, `orchestration_*`, `provenanced_runs` mutations; session injected at `60-61`.
- **`MethodService`** (`method_service.py:73-265`): `method_definitions`, `analysis_results`.
- **`ProvenanceService`** (`provenance_service.py:109-740`): composes `create_group`, `create_group_link`, `add_call_to_group`, direct `CallArtifact` inserts in `link_artifacts_to_group:354-368`.
- **`ArtifactService`** (`artifact_service.py:400-471` + `store_*` at `473+`): blob I/O + `_link_artifact_to_group` → `raw_calls` + `call_artifacts`.
- **`SweepRequestService`** (`sweep_request_service.py:620-1187`): request lifecycle, metadata, links, analysis recording.
- **`ProvenancedRunService`** (`provenanced_run_service.py:165-264`): unified execution rows via `create_provenanced_run`.
- **`DatabaseConnectionV2.session_scope`** (`db/_base_connection.py:183-196`): **commits** any work done on the yielded session.

---

## G. FINDINGS

**AMBIGUOUS lane (env-driven; same code paths usable for canonical or sandbox)**

- **Global default DB URL:** `config.py:118-121` — empty `DATABASE_URL` falls back to `sqlite:///study_query_llm.db` **or** uses env; operators can point either at Jetstream or local without code distinction.
- **Pipeline stages:** `pipeline/parse.py:38-40`, `pipeline/acquire.py:35-37`, `pipeline/analyze.py:85-87` — require `database_url` or `DATABASE_URL`.
- **Workers / supervisors:** `services/jobs/runtime_workers.py:61-69`, `services/jobs/runtime_supervisors.py:199-201,360-362` — `DATABASE_URL` only.
- **Ingestion:** `experiments/ingestion.py:54-68` docstring references "NeonDB" but code uses passed `DatabaseConnectionV2` only — **no Jetstream verification** at write sites `102-348`.
- **`scripts/ingest_sweep_to_db.py:487-488`** uses `config.database.connection_string` (`config.py:118-121`) — **AMBIGUOUS**.
- **Migrations** uniformly `os.environ.get("DATABASE_URL")` e.g. `db/migrations/drop_embedding_vectors.py:43-46`, `add_sweep_worker_safety.py:80-83`.

**Sites that use `DATABASE_URL` (or config derived from it) without verifying "which DB" / lane**

- Essentially **all** entries marked AMBIGUOUS in section A, including **`session_scope` auto-commit** `db/_base_connection.py:183-196`.
- **Panel** uses `get_db_connection()` (helpers) — same ambiguity as app config; writes at `panel_app/views/inference.py:226-244,397-434`.

**Destructive operations with weak or operator-only protection**

- **`scripts/purge_dataset_acquisition.py:217-220`** — deletes rows; guarded by explicit `--database-url` with `--execute` `138-144`, loopback rule `145-150`, confirmation `204-211` **but** still allows remote if `--allow-remote-target` `145-150`.
- **`scripts/archive_defective_data.py:231-243`** — deletes online `raw_calls`; no automated "is this Jetstream?" check beyond operator-supplied URLs `169-170`.
- **`scripts/history/sweep_recovery/archive_pre_fix_runs.py:271-290`** — deletes online `groups`; same pattern as archive script.
- **`db/migrations/drop_embedding_vectors.py:63-67`** — unconditional drop once table exists; URL from env `43-46`.
- **`drop_all_tables` / `recreate_db`** guarded for Postgres targets `db/_base_connection.py:133-172`; **tests** call drop in `tests/test_db/test_repository_v2.py:22`.

**Could operator set `DATABASE_URL` to Jetstream and run "local" scripts?**

- **Yes** for any script/service that uses only `DATABASE_URL` / `config` — e.g. `scripts/register_clustering_methods.py:61-72`, `scripts/backfill_run_fingerprints.py:38-44`, `label_pre_fix_runs:47-48`, workers `services/jobs/runtime_workers.py:61-63`. **No code-level distinction** between Jetstream and local in those paths.

**READ_MIRROR-specific (narrow)**

- **`scripts/sync_from_online.py:250-281`** — writes only to `local_url` (`LOCAL_DATABASE_URL` or `--local-url`) with explicit guardrails; DML in `write_batch:172-218`.
