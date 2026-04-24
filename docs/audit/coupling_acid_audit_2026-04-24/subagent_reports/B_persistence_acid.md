# Subagent B — Persistence & ACID Seams

Status: raw subagent finding (verbatim, edited only for fenced code blocks)
Authored by: explore subagent B, 2026-04-24
Scope: DB touchpoints, transactions, dual-writes, idempotency, connection lifecycle, v1/v2 hazards.

---

## 1. Database Touchpoint Inventory

**Repository (legitimate)** — files under `db/` that own engines, sessions, migrations, or repository-pattern access:

- `src/study_query_llm/db/_base_connection.py` — `create_engine`, `sessionmaker`, `session_scope` (commit/rollback at lines 183-196)
- `src/study_query_llm/db/connection_v2.py` — `engine.connect()`, raw `text("CREATE EXTENSION...")`, `conn.commit()` at lines 47-51
- `src/study_query_llm/db/connection.py` — v1 wrapper (deprecated)
- `src/study_query_llm/db/raw_call_repository.py` — primary v2 repository; `session.execute(text(...))` for stats at lines 231-257; extensive `flush` / `begin_nested`
- `src/study_query_llm/db/inference_repository.py` — v1 compat; `session_scope` in classmethod at lines 29-30
- `src/study_query_llm/db/migrations/*.py` — `engine.begin()`, `conn.execute(text(...))`, etc.

**Service-via-repository (correct)** — open `DatabaseConnectionV2.session_scope()`, construct `RawCallRepository` / `MethodService` / `ProvenancedRunService`, and perform work primarily through those APIs:

- `services/method_service.py`, `provenance_service.py`, `provenanced_run_service.py`, `sweep_request_service.py`, `summarization_service.py`, `study_service.py`, `data_quality_service.py`, `paraphraser_factory.py`, `supervisor_mode.py`
- `services/embeddings/service.py`, `services/embeddings/helpers.py` (and `services/embeddings/persistence.py` — only calls repository methods)
- `services/jobs/runtime_workers.py`, `runtime_supervisors.py`, `job_reducer_service.py`
- `pipeline/runner.py`, `acquire.py`, `parse.py`, `snapshot.py`, `embed.py`, `analyze.py`
- `analysis/mcq_analyze_request.py`
- `experiments/ingestion.py`, `runtime_sweeps.py`, `sweep_worker_main.py`, `mcq_run_persistence.py`
- `services/inference_service.py` — takes a repository from caller (no `session_scope` inside file; still service-layer)

**Direct DB access from non-db layer (LEAK)** — SQLAlchemy `session.query` / `session.add` / model construction **outside** `RawCallRepository` method bodies (bypasses the "all DB through repository" comment in `raw_call_repository.py`):

| Area | Examples |
|------|-----------|
| Pipeline helpers | `analyze.py`, `embed.py`, `parse.py`, `snapshot.py` — direct `CallArtifact` / `Group` queries (e.g. `pipeline/analyze.py` lines 152-180) |
| Query / analytics services | `sweep_query_service.py` — many `session.query(Group)`, `GroupLink` (e.g. lines 249-269) |
| Provenance / unified views | `provenance_service.py` (`session.query(Group)`, `CallArtifact`); `provenanced_run_service.py` (`list_unified_execution_view` ORM at lines 329-405); `sweep_request_service.py` occasional `repository.session.query(Group)` |
| Artifacts | `artifact_service.py` — builds `CallArtifact` and `session.add` in `_link_artifact_to_group` at lines 452-463 |
| Method compatibility table | `method_service.py` — direct `MethodDefinition` / `AnalysisResult` ORM (e.g. lines 253-265) |
| Experiments | `runtime_sweeps.py`, `sweep_worker_main.py` — `SweepRunClaim` ORM; `ingestion.py` — `Group` / `CallArtifact` queries |

**Not DB:** `utils/session_utils.py` — Cursor/gitignore "session" utilities only.

---

## 2. Transaction & Atomicity Boundaries

**`run_stage` — three separate committed transactions** (`pipeline/runner.py` lines 98-178):

```python
with db.session_scope() as session:
    repo = RawCallRepository(session)
    ...
    run_id = repo.create_provenanced_run(... RUN_STATUS_RUNNING ...)
try:
    if write_artifacts is not None:
        with db.session_scope() as session:
            ...
            artifact_uris = _normalize_artifact_uris(
                write_artifacts(artifact_service, identity)
            )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        if finalize_db is not None:
            result_metadata = dict(finalize_db(repo, identity, artifact_uris) or {})
        if identity.run_id is not None:
            repo.update_provenanced_run(
                identity.run_id,
                run_status=RUN_STATUS_COMPLETED,
            )
except Exception as exc:
    if identity is not None and identity.run_id is not None:
        with db.session_scope() as session:
            ...
            repo.update_provenanced_run(
                identity.run_id,
                run_status=RUN_STATUS_FAILED,
                ...
            )
    raise
```

- **Atomicity:** Group + links + "running" `provenanced_runs` row commit **before** blob/CallArtifact writes; those commit in a **second** transaction; finalize + "completed" in a **third**.
- **Partial failure:** If artifact TX succeeds and finalize fails, you can have **persisted blobs/CallArtifact rows** tied to a run left **running** until the failure path opens a **fourth** session and marks `failed` (good for status, bad for garbage blobs).
- **Rollback path:** `_base_connection.session_scope` rolls back on exception inside a single `with` block (lines 184-194); there is **no cross-transaction compensating delete** for blobs.

**Analyze stage `finalize_db` (dual-write) — single transaction for DB writes**: Inside the third `run_stage` session, `_finalize_analysis` calls `MethodService.record_result` many times then `ProvenancedRunService.record_analysis_execution` (`pipeline/analyze.py` 853-1008) — all share one session until `session_scope` commits.

**LangGraph worker — job completion + dual-write in one transaction** (`services/jobs/runtime_workers.py` 134-155):

```python
with db.session_scope() as session:
    repo = RawCallRepository(session)
    repo.complete_orchestration_job(
        outcome.job_id, result_ref=outcome.result_ref
    )
    method_svc = MethodService(repo)
    provenanced_run_svc = ProvenancedRunService(repo)
    record_langgraph_job_outcome(
        ...
    )
```

**Embedding persistence — RawCall + cache row in one session**: `persist_embedding` inserts `RawCall` then `upsert_embedding_cache_entry` (`services/embeddings/persistence.py` 106-133) — atomic **per** `session_scope` wrapping the embedding batch.

**Chunked embedding helper — intentional multi-transaction**:

```python
# For chunked mode, run one DB transaction per chunk so progress is
# committed incrementally and restart/resume can reuse persisted cache rows.
```

(`services/embeddings/helpers.py` 104-105)

**Orchestration job enqueue — nested savepoint on insert**: `enqueue_orchestration_job` uses `begin_nested` around insert (`db/raw_call_repository.py` 1183-1191) to absorb `IntegrityError` for idempotent `job_key`.

---

## 3. Dual-Write Risks

| Pattern | Order | Failure / recovery | Second write idempotent? |
|--------|--------|---------------------|---------------------------|
| **`run_stage`: DB identity then blob then DB finalize** | (1) Group + run `running` (2) storage write + `CallArtifact` (3) `finalize_db` + `completed` | (2) without (3): failure handler marks run `failed`; **orphan blobs** possible. (1) without (2): committed empty group + running run. | Blob paths keyed by group/step; re-run creates **new** group id -> **not** idempotent at blob layer unless stage short-circuits upstream. |
| **Analyze: `analysis_results` + `provenanced_runs`** | Multiple `record_result` then `record_analysis_execution` in same TX | One TX -> all-or-nothing for DB. | `provenanced_runs`: upsert on `(request_group_id, run_key, run_kind)` (`db/raw_call_repository.py` 1518-1571) with **unique index** (`db/models_v2.py` 787-793). **`analysis_results`**: **no uniqueness** on `(source_group_id, result_key, analysis_group_id, method_definition_id)` — retries can **duplicate** rows. |
| **LangGraph: `complete_orchestration_job` + `record_result` + `record_analysis_execution`** | Single `session_scope` | Commit all together; exception rolls back whole scope. | Same as above for `analysis_results`; provenanced upsert helps canonical side. |
| **Embeddings: `RawCall` + `EmbeddingCacheEntry`** | Insert raw call then upsert cache (`services/embeddings/persistence.py` 106-133) | Same session -> single commit. | `uq_embedding_cache_key` (`db/models_v2.py` 851-852). |
| **Artifacts: blob then DB** | `_write_artifact_bytes` then `_link_artifact_to_group` (`services/artifact_service.py` 505-512) | Blob can exist without `CallArtifact` if DB fails after upload. | Re-upload with same logical path depends on storage backend; DB row is new each time unless deduped elsewhere. |
| **Placeholder `RawCall` per artifact** | `insert_raw_call` then `CallArtifact` (`services/artifact_service.py` 432-463) | Same session in `store_group_blob_artifact` path -> one commit for DB rows. | N/A |

---

## 4. Idempotency Mechanisms

**Pipeline stage keys (from `docs/DATA_PIPELINE.md`)**: `content_fingerprint` (acquire); `(source_dataset_group_id, parser_id, parser_version, dataframe_hash)` (parse); `(source_dataframe_group_id, spec_hash, resolved_index_hash)` (snapshot); embedding matrix lookup `(dataset_key, embedding_engine, provider, entry_max, key_version)` (embed); analyze: dual-write + fingerprint contract including representation and `input_snapshot_group_id`.

**`canonical_run_fingerprint` / `fingerprint_hash`**: Computed in `ProvenancedRunService` (`services/provenanced_run_service.py` 68-101) and stored on `ProvenancedRun`; indexed (`db/models_v2.py` 758-759). Upsert merges fingerprints when updating existing run (`db/raw_call_repository.py` 1565-1568).

**`OrchestrationJob`**: Idempotent enqueue by `job_key` with `begin_nested` + `IntegrityError` recovery (`db/raw_call_repository.py` 1160-1191); unique `job_key` (`db/models_v2.py` 461). Claim path: load candidates, mutate first eligible row in Python (`db/raw_call_repository.py` 1218-1256) — **no `SELECT ... FOR UPDATE SKIP LOCKED`**, so under concurrent workers the same job could be **double-claimed** depending on isolation and timing.

**Embedding cache leases**: `EmbeddingCacheLease.cache_key` is **primary key** (`db/models_v2.py` 884-886); `try_acquire_embedding_cache_lease` read-then-insert (`db/raw_call_repository.py` 920-948) — concurrent inserts can race; second insert fails unless handled (not shown with `begin_nested` here).

**`create_provenanced_run` insert race**: Select-then-insert (`db/raw_call_repository.py` 1518-1596) with **unique** `(request_group_id, run_key, run_kind)` (`db/models_v2.py` 787-793). Unlike `create_group_link`, the **insert path does not catch `IntegrityError` and re-query**, so two workers can force **transaction rollback** on conflict rather than clean upsert.

**Analyze reuse**: Pre-check `get_provenanced_run_by_request_and_key` under `_analysis_lock` (`pipeline/analyze.py` 756-786) — **process-local lock**; cross-process races still rely on DB constraints.

---

## 5. Connection Lifecycle & Isolation

- **Session creation:** `BaseDatabaseConnection.session_scope` creates a **new** `Session` per `with` block (`db/_base_connection.py` 184-186) — **function-scoped**, not request-scoped (CLI/worker model).
- **Engine:** One engine per `DatabaseConnectionV2` instance; **not** a global singleton in `connection_v2.py`. Typical pattern: construct `DatabaseConnectionV2` in `main` / stage entrypoint (e.g. langgraph worker `services/jobs/runtime_workers.py` 68-69).
- **Test isolation:** Risk is **shared DATABASE_URL + shared engine** if tests reuse one connection object; no module-level engine in library code. Pool on the engine can keep connections warm across tests unless fixtures use transactions or separate DBs.
- **Module-level state:** `EmbeddingService` uses in-process L1 cache and inflight dicts (`services/embeddings/service.py` 109-114) — can **bleed across tests** in the same process (not DB isolation, but cross-test coupling).

---

## 6. v1/v2 Cohabitation Hazards

- **Application `src`:** v2 is dominant — `RawCallRepository` imports `models_v2` only (`db/raw_call_repository.py` 15-26). `DatabaseConnection` / `InferenceRepository` are **deprecated** and **not referenced** elsewhere under `src/study_query_llm` (only `db/__init__.py` exports).
- **Tests:** Still import `InferenceRun` from `db.models` (v1) in `tests/test_db/*.py` — **test-only** split-brain vs production v2.
- **Logical "dual surface" (same Postgres v2 schema):**
  - **Reads:** `ProvenancedRunService.list_unified_execution_view` merges real `provenanced_runs` with **synthetic** rows from legacy `clustering_run` groups and `analysis_results` (`services/provenanced_run_service.py` 296-452) — consistency hazard if consumers treat synthetic rows like persisted PK-backed rows (`id=None`).
  - **Writes:** `SweepQueryService.get_unified_execution_runs` delegates to that view (`services/sweep_query_service.py` 220-228).
  - **Metrics:** `get_sweep_metrics_df` still walks **`clustering_run` groups** and metadata JSON (`services/sweep_query_service.py` 264-271) — parallel universe vs `provenanced_runs`-first analytics.

---

## 7. Top 5 ACID Improvements (cheap-while-empty)

1. **What problem:** `run_stage` splits identity, blob, and finalize across **multiple commits** — torn state (e.g. orphaned artifacts, "running" rows) under partial failures.
   **Evidence:** `pipeline/runner.py` 98-167.
   **Cost now (DB nearly empty):** Low — behavioral change + optional cleanup script for orphan groups; no heavy migration.
   **Sketch:** One `session_scope` for DB steps; write blobs **before** `CallArtifact` insert but **after** group id is known, or use **outbox / two-phase**: insert artifact row with `pending` URI then upload, then patch URI in same TX; or compensate by deleting blob on rollback hook.

2. **What problem:** **`analysis_results` duplicates** on retry — no uniqueness for `(method_definition_id, source_group_id, analysis_group_id, result_key)`.
   **Evidence:** `db/models_v2.py` 670-672; `record_result` always inserts (`services/method_service.py` 255-265).
   **Cost now:** Cheap unique index + `ON CONFLICT` upsert while tables are small; expensive later if deduping TB of duplicates.
   **Sketch:** `UNIQUE(method_definition_id, source_group_id, COALESCE(analysis_group_id,-1), result_key)` and upsert in `record_result`.

3. **What problem:** **Orchestration job claim** is read-modify-write without row locking — risk of **double execution** under concurrency.
   **Evidence:** `db/raw_call_repository.py` 1218-1250.
   **Cost now:** Add `SKIP LOCKED` / `FOR UPDATE` claim query or optimistic `version` column + `UPDATE ... WHERE version = ?`; cheap while job count is tiny.
   **Sketch:** PostgreSQL `UPDATE orchestration_jobs SET ... FROM (SELECT id ... FOR UPDATE SKIP LOCKED LIMIT 1)`.

4. **What problem:** **`create_provenanced_run` insert path** races on unique key — no `IntegrityError` recovery (unlike `create_group_link`).
   **Evidence:** `db/raw_call_repository.py` 1518-1596 vs 1027-1049.
   **Cost now:** Small code change; empty DB means few conflicts today, but fixes future worker scale.
   **Sketch:** `begin_nested` + catch `IntegrityError` + re-select existing row and update.

5. **What problem:** **Unified execution view** merges canonical and legacy shapes — callers may **double-count** or follow **stale** legacy paths while `provenanced_runs` is canonical.
   **Evidence:** `services/provenanced_run_service.py` 296-452; sweep metrics still on `clustering_run` (`services/sweep_query_service.py` 269-271).
   **Cost now:** Low — deprecate legacy query paths, migrate metrics to `provenanced_runs`, drop compatibility branches before data grows.
   **Sketch:** Feature flag to return only persisted provenanced rows; one-off backfill from legacy groups into `provenanced_runs` while DB is empty.
