# Subagent 3 — Artifact URI Lifecycle (raw)

Date: 2026-04-24
Scope: complete lifecycle of artifact URIs from creation to DB persistence.
Method: readonly explore subagent.

**Contradiction vs. context block:** `_assert_uri_backend_compatible` is **not** in `src/study_query_llm/services/artifact_service.py`. It is defined and used only in `scripts/ingest_sweep_to_db.py` (`312-329`, invoked at `355`). Jetstream appears in DB connection destructive-guard and ops scripts, not in artifact backend selection.

---

## A. Storage backend inventory and selection logic

| Backend | Class | `backend_type` | Physical URI shape | Definition |
|--------|--------|----------------|-------------------|------------|
| Local | `LocalStorageBackend` | `"local"` | Absolute filesystem path from `Path.resolve()` | `src/study_query_llm/storage/local.py:18, 100-111` |
| Azure Blob | `AzureBlobStorageBackend` | `"azure_blob"` | HTTPS blob URL (`blob_client.url`) | `src/study_query_llm/storage/azure_blob.py:28, 154-165` |

**Protocol:** `StorageBackend` documents `logical_path` and `get_uri()` semantics at `src/study_query_llm/storage/protocol.py:4-30`.

**Construction:** Only these two types are registered in `StorageBackendFactory.create` (`src/study_query_llm/storage/factory.py:31-38`). Package `__init__` exports factory + local + protocol only (`src/study_query_llm/storage/__init__.py:7-15`).

**Selection when `ArtifactService` is constructed without `storage_backend`:** `__init__` calls `_resolve_default_backend(artifact_dir)` (`artifact_service.py:81-84, 86-145`):

1. Read `ARTIFACT_STORAGE_BACKEND`, default **`"local"`** if unset (`93`).
2. Read `ARTIFACT_RUNTIME_ENV`, default **`"dev"`** (`94`).
3. `strict_mode` from `ARTIFACT_STORAGE_STRICT_MODE` truthy env; **forced `True`** if `runtime_env in {"stage", "prod"}` (`95-97`).
4. If `backend_type == "local"` **and** `strict_mode`: **raise** (`99-102`).
5. If `backend_type == "azure_blob"`: build via `StorageBackendFactory.create("azure_blob", ...)` with env-driven args (`104-127`). On `ValueError`/`ImportError`: if `strict_mode` **re-raise** as `RuntimeError` (`128-132`); else **log warning and fall back** by setting `backend_type = "local"` (`133-138`).
6. If `backend_type == "local"` or empty: `StorageBackendFactory.create("local", base_dir=artifact_dir)` (`139-140`).
7. Unknown backend: warning + local fallback (`141-145`).

**Link to DB target (Jetstream): NONE.** `JETSTREAM_DATABASE_URL` is used for **destructive DDL policy** in `DatabaseConnection` (`db/_base_connection.py:17, 147-171`), not for choosing `local` vs `azure_blob`. **Choosing a Jetstream Postgres URL does not force `azure_blob`.** This is the architectural seam at the heart of the audit.

---

## B. All env vars controlling backend selection

| Env var | Role | File:line |
|---------|------|-----------|
| `ARTIFACT_STORAGE_BACKEND` | Backend choice; default local when unset | `artifact_service.py:93` |
| `ARTIFACT_RUNTIME_ENV` | Runtime lane (`dev`/`stage`/`prod`); affects strict default and blob container | `94, 96-97, 106, 160-169` |
| `ARTIFACT_STORAGE_STRICT_MODE` | Truthy → disallow local backend; also used in azure fallback decision | `95, 99-102, 128-132` |
| `ARTIFACT_AUTH_MODE` | Passed to `StorageBackendFactory.create` for azure | `107` |
| `AZURE_STORAGE_ACCOUNT_URL` | Optional account URL for factory | `113` |
| `AZURE_STORAGE_MANAGED_IDENTITY_CLIENT_ID` | Passed to factory | `114-115` |
| `AZURE_STORAGE_PREFIX` | Blob prefix | `117` |
| `AZURE_STORAGE_MAX_RETRIES` | Passed to factory | `118` |
| `AZURE_STORAGE_RETRY_BACKOFF_SECONDS` | Passed to factory | `119-120` |
| `AZURE_STORAGE_VERIFY_UPLOADS` | Truthy default True | `122-125` |
| `AZURE_STORAGE_CONNECTION_STRING` | Used inside `AzureBlobStorageBackend.__init__` | `azure_blob.py:55` |
| `AZURE_STORAGE_CONTAINER` | Default container name segment | `azure_blob.py:56-57; artifact_service.py:161-169` |
| `AZURE_STORAGE_CONTAINER_{RUNTIME_ENV_UPPER}` | Per-lane container override | `artifact_service.py:163-165` |
| `ARTIFACT_ALLOW_CROSS_ENV_CONTAINER` | Allows prod-like container name in non-prod runtime | `173-175` |
| `ARTIFACT_BLOB_MAX_BYTES` / `ARTIFACT_BLOB_MAX_GB` | Quota for azure writes only | `223-251`, enforcement `277-279` |

---

## C. Artifact write sites

**Backend trace (default path):** Any `ArtifactService(..., storage_backend=None)` → `_resolve_default_backend` (`artifact_service.py:81-84, 86-145`) → `storage.write` / `get_uri` via `_write_artifact_bytes` (`296-320`) → `_link_artifact_to_group` persists `CallArtifact.uri` (`452-459`) and may insert `RawCall` with `response_json={"uri": uri}` (`434-439`).

| File:line | Artifact type(s) | Backend trace | Target column / JSON |
|-----------|------------------|---------------|----------------------|
| `artifact_service.py:473-536` | Any `artifact_type` (group blob) | `_write_artifact_bytes` → `storage` | `call_artifacts.uri` (`455`); optional `raw_calls.response_json` (`439`) |
| `artifact_service.py:538-590` | `sweep_results` | same | `call_artifacts.uri` |
| `artifact_service.py:592-658` | `dataset_snapshot_manifest` | same | `call_artifacts.uri` |
| `artifact_service.py:660-708` | `embedding_matrix` | same | `call_artifacts.uri` |
| `artifact_service.py:750-811` | `cluster_labels` | same | `call_artifacts.uri` |
| `artifact_service.py:813-870` | `pca_components` | same | `call_artifacts.uri` |
| `artifact_service.py:872-925` | `metrics` | same | `call_artifacts.uri` |
| `artifact_service.py:927-990` | `representatives` | same | `call_artifacts.uri` |
| `pipeline/acquire.py:138-166` | `dataset_acquisition_file`, `dataset_acquisition_manifest` | `run_stage` → `ArtifactService(repository=..., artifact_dir=...)` (`runner.py:150-153`) → `store_group_blob_artifact` | `call_artifacts.uri` |
| `pipeline/parse.py:308-338` | `canonical_dataset_parquet`, `dataset_dataframe_manifest` | same pattern | `call_artifacts.uri` |
| `pipeline/snapshot.py:303-315` | `dataset_subquery_spec` (`ARTIFACT_TYPE_SUBQUERY_SPEC:22`) | same | `call_artifacts.uri` |
| `pipeline/embed.py:217-231` | `embedding_matrix` | same | `call_artifacts.uri` |
| `pipeline/analyze.py:836-850` | Dynamic per `payload.artifacts` via `_artifact_type_and_content_type` | same | `call_artifacts.uri` |
| `experiments/ingestion.py:161-162` | `sweep_results` | `ArtifactService(repository=repo)` (no `storage_backend`) | `call_artifacts.uri`; **`groups.metadata_json` gets `artifact_uri`** (`181-182`) |
| `experiments/sweep_worker_main.py:126, 155-163` | `embedding_matrix` | `ArtifactService(repository=repo)` | `call_artifacts.uri` |
| `services/embeddings/helpers.py:43, 61-72` | `embedding_matrix` | `ArtifactService(repository=repo)` | `call_artifacts.uri` |
| `services/provenance_service.py:354-365` | Caller-supplied `artifact_type` / `uri` | **No** `ArtifactService`; URI string copied into ORM | `call_artifacts.uri` |
| `panel_app/views/storage_stats.py:278` | (probe only, not pipeline persistence) | `StorageBackendFactory.create("azure_blob", ...)` for listing | N/A |
| `scripts/purge_dataset_acquisition.py:157-159` | N/A (reads storage) | `ArtifactService(repository=None, ...).storage` | N/A |

**No check at any of these sites that storage backend matches `DATABASE_URL` / Jetstream.**

---

## D. DB columns holding artifact URIs

**SQLAlchemy `models_v2.py` (primary schema):**

| Model | Column | Table | Constraints / notes |
|-------|--------|-------|-------------------|
| `CallArtifact` | `uri` | `call_artifacts` | `String(1000)`, `nullable=False` (`227`); indexes on `call_id`,`artifact_type` (`235-237`); **no CHECK / URL validator** |
| `RawCall` | `response_json` | `raw_calls` | JSON (`51`); **may embed `{"uri": ...}`** for artifact placeholder calls (`artifact_service.py:439`) — **no URI constraint** |
| `ProvenancedRun` | `result_ref` | `provenanced_runs` | `String(400)`, nullable (`755`); used for sweep artifact URL in ingestion (`experiments/ingestion.py:298`) — **no https-only constraint** |
| `OrchestrationJob` | `result_ref` | `orchestration_jobs` | `String(200)`, nullable (`433`) — semantic "ref", not validated as blob URL |
| `MethodDefinition` | `code_ref` | `method_definitions` | `String(500)` (`574`) — code path string, not pipeline blob URI |
| `AnalysisResult` | `result_json` | `analysis_results` | JSON (`660`); analyze stage stores **`"uris": artifact_uris`** map (`analyze.py:894-903`) — **nested local paths possible** |
| `Group` | `metadata_json` | `groups` | JSON (`128`); e.g. **`artifact_uri`** on clustering runs (`ingestion.py:181-182`) — **no constraint** |

**Legacy `models.py`:** `InferenceRun` has no artifact URI columns (`15-44`).

**Pydantic / regex validators on ORM `uri`:** None found in `models_v2.py` (only `CheckConstraint` on enums, e.g. `RawCall.status:62`, `SweepRunClaim:359`).

---

## E. Existing consistency validators

| Validator | What it checks | What it does **not** check | Invoked from |
|-----------|----------------|-----------------------------|--------------|
| `_assert_uri_backend_compatible` | Blob URL (`https://` + `blob.core.windows.net`) vs `artifact_service.storage.backend_type` local/azure mismatch (`scripts/ingest_sweep_to_db.py:312-329`) | Does **not** compare URI to `DATABASE_URL` / Jetstream; does **not** prevent **writing** local URIs to a remote DB; does **not** run on pipeline stages | `load_and_ingest_from_artifact:355` in `scripts/ingest_sweep_to_db.py` only |
| `scripts/verify_call_artifact_blob_lanes.py` | For each `call_artifacts.uri`: if parseable as Azure HTTPS, container must match `--expected-container`; optional blob key prefix (`154-170`) | **Explicitly treats non-Azure URIs as "local_or_other" and does not fail** (`180-181`) | CLI `main:72-210` |
| `ArtifactService._resolve_default_backend` strict mode | Blocks **local** backend when strict (`99-102`) | Does not require azure when DB is remote; non-strict allows local with any DB | Construction (`86-145`) |
| `ArtifactService._enforce_quota_before_write` | Azure-only quota (`277-279`) | Local writes; URI shape | `_write_artifact_bytes:306-310` |
| `DatabaseConnection._assert_destructive_operation_allowed` | Jetstream URL match blocks **destructive DDL** (`166-171`) | Artifact backend; DML | `drop_all_tables:174-176` |

---

## F. Lifecycle diagram — snapshot artifacts (Jetstream DB + `ARTIFACT_STORAGE_BACKEND=local` or unset)

1. **Entry:** `snapshot(..., database_url=<Jetstream URL>` or `db` connected to Jetstream) (`snapshot.py:215-223`); DB from `_resolve_db:25-37`.
2. **Read prior dataframe URI from Jetstream:** Session loads `Group` (`227-238`); `find_dataframe_parquet_uri(session, ...)` reads **`call_artifacts.uri`** for canonical parquet (`parse.py:203-213`).
3. **Load parquet bytes:** `_load_dataframe_frame` builds `ArtifactService(artifact_dir=artifact_dir)` **without repository** (`snapshot.py:117-123`) → same `_resolve_default_backend` → **`storage.read_from_uri(dataframe_parquet_uri)`** (`123`). If parquet row in Jetstream already holds a **local** path, read uses **local filesystem** on the runner (may fail or read wrong machine).
4. **Compute payload** in memory (`247-275`).
5. **Reuse path:** If existing snapshot group, returns URIs from `_collect_snapshot_artifact_uris` which reads **`CallArtifact.uri`** (`40-50, 285-290`).
6. **New snapshot write:** `run_stage:320-337`:
   - **Transaction 1:** `create_group` + `group_links` on **Jetstream** (`runner.py:98-121`) → tables `groups`, `group_links`.
   - **Transaction 2:** `ArtifactService(repository=artifact_repo, artifact_dir=artifact_dir)` (`147-153`) → **`store_group_blob_artifact`** (`snapshot.py:303-315`) → `_write_artifact_bytes` → **`LocalStorageBackend.write`** returns **absolute path** (`local.py:38-58, 100-111`).
   - **`CallArtifact`** inserted with that path in **`call_artifacts.uri`** (`artifact_service.py:452-459`); **`raw_calls`** row if placeholder (`434-439`).
7. **No `provenanced_runs` row** for snapshot: `run_stage` only creates a run when `request_group_id` and `run_key` set (`runner.py:124-136`); `snapshot.py` omits both (`320-336`).

**URI shape for new snapshot artifact:** Windows/Linux absolute path string (local backend), **not HTTPS** — and that path lands in Jetstream's `call_artifacts.uri` row.

---

## G. FINDINGS

**Where local-path URIs can land in Jetstream**

- **Any** pipeline stage using `run_stage` + `ArtifactService` with default backend while `DATABASE_URL` points at Jetstream: `runner.py:150-153` + `artifact_service.py:93-140` + `_link_artifact_to_group:452-459`.
- **Snapshot specifically:** `snapshot.py:303-315` → `call_artifacts.uri` on the **same** DB session's target (Jetstream if that's what `db` is).
- **Ingestion / sweep:** `experiments/ingestion.py:161-182` → `call_artifacts.uri` and **`groups.metadata_json['artifact_uri']`**.
- **Sweep worker / embeddings:** `sweep_worker_main.py:155-163`, `services/embeddings/helpers.py:72-80`.
- **Provenance API:** `provenance_service.py:354-365` — any caller-supplied `uri` string is persisted.
- **Analyze follow-on:** `analysis_results.result_json` may embed the same local paths under `"uris"` (`analyze.py:894-903`); `provenanced_runs.result_ref` may point to a primary artifact URI (`analyze.py:907-909, 997-1003`).

**Artifact columns / JSON with no constraint preventing local paths**

- `call_artifacts.uri` (`models_v2.py:227`).
- `raw_calls.response_json` (`51`) when used as artifact placeholder (`artifact_service.py:439`).
- `groups.metadata_json` (`128`) e.g. `artifact_uri` (`ingestion.py:181-182`).
- `analysis_results.result_json` (`660`) nested `uris` (`analyze.py:900-903`).
- `provenanced_runs.result_ref` (`755`) (`ingestion.py:298, analyze.py:1003`).

**Backend selection decoupled from DB target**

- **Entire** `ArtifactService._resolve_default_backend` (`artifact_service.py:86-145`): no `DATABASE_URL`, `JETSTREAM_DATABASE_URL`, or connection string parameter.
- **Jetstream** only tied to **DDL safety**, not artifacts (`_base_connection.py:147-171`).
- **Risk:** `ARTIFACT_STORAGE_BACKEND=azure_blob` with misconfiguration in **non-strict** dev: **silent fallback to local** (`133-138`) while DB may still be remote.

---

## "Artifact tables" (tables that hold URI rows or URI-sized refs)

**Dedicated / routine artifact pointer storage**

- **`call_artifacts`** — canonical `uri` column (`models_v2.py:222-227`).

**Tables that can store artifact locations indirectly**

- **`raw_calls`** — `response_json` (`51`).
- **`analysis_results`** — `result_json` (`660`).
- **`provenanced_runs`** — `result_ref` (`755`).
- **`groups`** — `metadata_json` (`128`).
- **`orchestration_jobs`** — `result_ref` (`433`) (may or may not be a blob URL depending on job).

**Not artifact URI tables for pipeline outputs**

- **`method_definitions.code_ref`** — code reference string (`574`), not blob artifact URI from `ArtifactService`.
