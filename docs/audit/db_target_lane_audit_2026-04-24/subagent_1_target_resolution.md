# Subagent 1 — DB Target URL Resolution Audit (raw)

Date: 2026-04-24
Scope: every site that resolves a database URL or determines which DB target a runtime/script will hit.
Method: readonly explore subagent.

---

## A. Central DB connection chokepoint(s)

| Path | Exposes | Enforces |
|------|---------|----------|
| `src/study_query_llm/config.py:96-121`, `src/study_query_llm/config.py:234-237` | Global `config` with `config.database.connection_string` from `DATABASE_URL` with default `sqlite:///study_query_llm.db` if unset/blank: `src/study_query_llm/config.py:95-104`, `src/study_query_llm/config.py:116-121`. | Loads dotenv: `src/study_query_llm/config.py:50-73`; optional `STUDY_QUERY_LLM_DOTenv` first candidate: `src/study_query_llm/config.py:29-35`. **Does not** validate Jetstream vs local; **only** string selection + default SQLite file path. |
| `src/study_query_llm/db/_base_connection.py:98-197` | `BaseDatabaseConnection`: `create_engine` + `sessionmaker`: `src/study_query_llm/db/_base_connection.py:105-110`, `src/study_query_llm/db/_base_connection.py:174-201`. | Destructive DDL guard: SQLite always allowed: `src/study_query_llm/db/_base_connection.py:136-145`; else requires `SQLLM_ALLOW_DESTRUCTIVE_DDL=1`: `src/study_query_llm/db/_base_connection.py:140-145`, `src/study_query_llm/db/_base_connection.py:16-17`. If that env is `1` and `JETSTREAM_DATABASE_URL` parses as Postgres, **refuses** `drop_all_tables` / `recreate_db` when connection URL matches same host/port/dbname as `JETSTREAM_DATABASE_URL` (non-overridable): `src/study_query_llm/db/_base_connection.py:147-172`. **Does not** read `DATABASE_URL` — caller passes the URL string. |
| `src/study_query_llm/db/connection_v2.py:14-61` | `DatabaseConnectionV2` subclass; `init_db` enables pgvector when possible: `src/study_query_llm/db/connection_v2.py:43-60`. | Inherits all `BaseDatabaseConnection` behavior. |
| `src/study_query_llm/db/connection.py:16-32` | Deprecated `DatabaseConnection` (v1 schema). | Same base class as v2. |
| `panel_app/helpers.py:30-38` | `get_db_connection()` builds `DatabaseConnectionV2(config.database.connection_string)`. | Inherits v2; UI messaging references `DATABASE_URL` in `panel_app/helpers.py:84-85`. |
| `panel_app/views/storage_stats.py:75` | Reads `url = config.database.connection_string` for stats. | Also displays `LOCAL_DATABASE_URL` from env for runbook text: `panel_app/views/storage_stats.py:233-243`. |
| `scripts/db_target_guardrails.py:40-70` | `parse_postgres_target`, `is_loopback_target`, `same_db_target` — **parsing only**, not a connection factory. | No env reads in this file. |

---

## B. All env vars consumed for **DB target resolution**

| Variable | Where read / written |
|----------|----------------------|
| `STUDY_QUERY_LLM_DOTENV` | `src/study_query_llm/config.py:32-35` (dotenv path override) |
| `DATABASE_URL` | `src/study_query_llm/config.py:66-70`, `src/study_query_llm/config.py:118-120`; `panel_app/app.py:32-33` (if blank, load `.env` with `override=True`); pipeline `acquire`/`parse`/`embed`/`snapshot`/`analyze` (e.g. `src/study_query_llm/pipeline/acquire.py:35-38`); `sweep_worker_main.py:53,1326-1331`; `runtime_workers.py:61-68`; `runtime_supervisors.py:199-201,360-368`; `runtime_sweeps.py:297-306`; `mcq_analyze_request.py:311-315`; all `src/study_query_llm/db/migrations/*.py` reading `os.environ.get("DATABASE_URL")` (e.g. `src/study_query_llm/db/migrations/add_group_links.py:28-30`); `docker-compose.yml:14` default `sqlite:////data/...` if unset; `deploy/jetstream/docker-compose.jetstream.yml:25` |
| `JETSTREAM_DATABASE_URL` | Destructive guard comparison only: `src/study_query_llm/db/_base_connection.py:147-172` (via `:17`); `scripts/verify_db_backup_inventory.py:124-134`; `scripts/upload_jetstream_pg_dump_to_blob.py:141`; `scripts/dump_postgres_for_jetstream_migration.py:69-71,83-84`; `scripts/probe_postgres_inventory.py:38-39` (default **name** of env var) |
| `LOCAL_DATABASE_URL` | `scripts/sync_from_online.py:255` (local target); `scripts/init_local_db.py:39-42`; `scripts/restore_pg_dump_to_local_docker.py:90-92`; `scripts/check_active_workers.py:47-54`; `scripts/verify_db_backup_inventory.py:124-125,133-134`; `panel_app/views/storage_stats.py:233-237`; `scripts/archive_defective_data.py:169-170`; `scripts/dump_postgres_for_jetstream_migration.py:81-82` |
| `SOURCE_DATABASE_URL` | `scripts/sync_from_online.py:250-253` (`online_url`); `scripts/dump_postgres_for_jetstream_migration.py:86-88` (generic mode) |
| `NEON_DATABASE_URL` | `scripts/ingest_mcq_probe_json_to_sweep_db.py:139-141` (fallback after `DATABASE_URL`) |
| `SQLLM_ALLOW_DESTRUCTIVE_DDL` | `src/study_query_llm/db/_base_connection.py:16,140-145` |
| `JETSTREAM_SSH_*`, `JETSTREAM_POSTGRES_PASSWORD` | `scripts/start_jetstream_postgres_tunnel.py:24-30,253-261` (operator messaging only) |
| `PANEL_DATABASE_URL` / `TEST_DATABASE_URL` / `SCRATCH_DATABASE_URL` / `PIPELINE_DATABASE_URL` | **Not found in `*.py`** — ledger-only at `docs/review/DOC_PARITY_LEDGER.md:80` (staged contract); `NEONDB_DATABASE_URL` in `.env.example:71` is documentation placeholder only |

---

## C. Resolution precedence (what wins; behavior when none set)

| Context | Precedence / behavior |
|--------|------------------------|
| **`Config` / `config.database.connection_string`** | If `DATABASE_URL` empty/whitespace, `_load_dotenv` loads first existing candidate with `override=True` so file can set it: `src/study_query_llm/config.py:66-70`. If still blank after that, `Config` sets **default** `sqlite:///study_query_llm.db`: `src/study_query_llm/config.py:118-121` and `:100-104`. |
| **If `DATABASE_URL` already set (non-blank) before `Config` import** | Dotenv loaded with `override=False` only: `src/study_query_llm/config.py:71-72` — process env **wins** over `.env` files. |
| **Panel** | If `DATABASE_URL` blank, repo `.env` loaded with `override=True` before other imports: `panel_app/app.py:25-35`. |
| **`ingest_mcq_probe_json_to_sweep_db.py`** | `DATABASE_URL` first, else `NEON_DATABASE_URL`: `scripts/ingest_mcq_probe_json_to_sweep_db.py:139-141`. |
| **`sync_from_online.py`** | Source: `--online-url` → `SOURCE_DATABASE_URL` → `DATABASE_URL`: `scripts/sync_from_online.py:250-254`. Local: `--local-url` → `LOCAL_DATABASE_URL`: `:255`. |
| **`check_active_workers.py`** | CLI `--database-url` else `LOCAL_DATABASE_URL` else `DATABASE_URL`: `scripts/check_active_workers.py:47-54,74-75`. |
| **`dump_postgres_for_jetstream_migration.py`** | `--source-url` wins; else `--from-local` → `LOCAL_DATABASE_URL`; else `--from-jetstream` → `JETSTREAM_DATABASE_URL`; else `SOURCE_DATABASE_URL` or `DATABASE_URL`: `scripts/dump_postgres_for_jetstream_migration.py:79-90`. |
| **`create_bank77_contrast_snapshots._resolve_database_url`** | `--database-url` → `DATABASE_URL` → else **file SQLite** under `artifact_dir`: `scripts/create_bank77_contrast_snapshots.py:56-65`. |

**Diagram (text):**

- App library path (import `config`): `os.environ["DATABASE_URL"]` (if set) > dotenv (first file with `override` rules in `_load_dotenv_files`) > default SQLite file `sqlite:///study_query_llm.db`.
- Long-running entrypoints (`sweep`, workers, supervisors, pipeline helpers): `database_url` parameter OR `os.environ.get("DATABASE_URL")` (required in most) — **no** automatic merge with `JETSTREAM_DATABASE_URL` for primary connection.
- Destructive DDL guard: target URL = constructor argument; Jetstream check uses `JETSTREAM_DATABASE_URL` from env vs that URL (`src/study_query_llm/db/_base_connection.py:133-172`).

---

## D. Direct connection bypasses

- **Reality:** There is no single `get_engine()` service; `BaseDatabaseConnection` **is** the shared constructor (`create_engine` inside it): `src/study_query_llm/db/_base_connection.py:105-110`.
- **Additional `create_engine` call sites (bypass reusing a shared singleton but still SQLAlchemy):**
  - `scripts/purge_dataset_acquisition.py:161-162`
  - `scripts/check_active_workers.py:83-84`
  - `scripts/sync_from_online.py:290-295`
  - `scripts/verify_db_backup_inventory.py:55`
  - `scripts/verify_call_artifact_blob_lanes.py:130`
  - `scripts/probe_postgres_inventory.py:61`
  - `scripts/sanity_check_database_url.py:60`
  - `scripts/upload_jetstream_pg_dump_to_blob.py:50`
- **`psycopg2.connect` / `sqlite3.connect` / `asyncpg.connect`:** no matches in `*.py` in repo.
- **Subprocess to Postgres** (not SQLAlchemy): `restore_pg_dump_to_local_docker.py` uses `pg_restore`/`dropdb`/`createdb` with URL-derived args (`scripts/restore_pg_dump_to_local_docker.py:137-159`).

---

## E. Escape hatches (safety override)

| Mechanism | Effect | Cite |
|-----------|--------|------|
| `SQLLM_ALLOW_DESTRUCTIVE_DDL=1` | Allows `drop_all_tables` / `recreate_db` for **non-sqlite** targets **except** when URL matches `JETSTREAM_DATABASE_URL` (that match still hard-stops): `src/study_query_llm/db/_base_connection.py:133-172` | `:136-145, :166-172` |
| `DatabaseConnectionV2` / `BaseDatabaseConnection` accepts **any** connection string at construction | Callers can point at Jetstream, local Docker, SQLite, in-memory: no validation at `__init__`: `src/study_query_llm/db/_base_connection.py:105-110` | — |
| `sync_from_online.py` `--allow-same-source-target` | Bypasses "same DB" check: `scripts/sync_from_online.py:238-242,269-273` | — |
| `sync_from_online.py` `--allow-remote-target` | Allows non-loopback local writes: `scripts/sync_from_online.py:243-247,276-280` | — |
| `purge_dataset_acquisition.py` `--allow-remote-target` | Allows non-loopback purges: `scripts/purge_dataset_acquisition.py:145-150` | — |
| `restore_pg_dump_to_local_docker.py` `--allow-remote-target` + `--confirm-target-db` | Allows restore to non-local Postgres: `scripts/restore_pg_dump_to_local_docker.py:114-128` | — |

---

## F. Per-script classification (highlights — see source for full table)

- **Sandbox-only by construction:** `scripts/snapshot_inventory.py:306-313` uses `sqlite:///{tmpdir}` (ephemeral).
- **Local mirror by intent:** `scripts/sync_from_online.py` writes only to `--local-url` / `LOCAL_DATABASE_URL` with loopback default; doc says "Only downloads, never uploads" (`:6-7`).
- **DATABASE_URL dependent (silent SQLite default if unset):** `scripts/ingest_sweep_to_db.py:487` uses `config.database.connection_string`, which falls back to `sqlite:///study_query_llm.db` when `DATABASE_URL` is empty.
- **DATABASE_URL strictly required:** `scripts/run_bank77_pipeline.py:323-325`, `src/study_query_llm/experiments/sweep_worker_main.py:1326-1327` raise if missing.
- **`create_bank77_contrast_snapshots`:** has implicit local SQLite fallback under `artifact_dir` (`:64-65`) — confused contract.

---

## G. Findings (ambiguity, silent defaults, doc/code drift)

1. **Single `DATABASE_URL` is the default runtime spine** for app code, workers, and most scripts; `JETSTREAM_DATABASE_URL` is **not** auto-selected for app connections — only for guards + ops scripts: compare `src/study_query_llm/config.py:116-121` (no Jetstream) vs `src/study_query_llm/db/_base_connection.py:147-172` (Jetstream in destructive check only) vs tunnel doc `scripts/start_jetstream_postgres_tunnel.py:24-26`.
2. **Silent SQLite default** when `DATABASE_URL` unset/blank: `src/study_query_llm/config.py:118-121` and `:100-104` — operators importing `config` can write to local file `study_query_llm.db` without Postgres.
3. **Docker compose default** for app container: `docker-compose.yml:14` uses `sqlite:////data/study_query_llm.db` if `DATABASE_URL` not provided.
4. **`JETSTREAM_DATABASE_URL` in destructive guard** can block drop/recreate for **any** connection string that **matches** it (e.g. tunneled `127.0.0.1:5433` to Jetstream), matching test intent in `tests/test_db/test_destructive_guard.py:69-70`.
5. **Staged contract vars** `PANEL_DATABASE_URL`, `TEST_DATABASE_URL`, `SCRATCH_DATABASE_URL` appear in docs (`docs/review/DOC_PARITY_LEDGER.md:80`) but **no Python consumer** in repo (grep: no `*.py` matches) — "prep-only / not wired" per that ledger.
6. **`ingest_sweep_to_db` uses `config.database.connection_string`**, so **inherits SQLite default** if operator forgets `DATABASE_URL` — `scripts/ingest_sweep_to_db.py:487` + `src/study_query_llm/config.py:116-121`, whereas `sweep_worker` **requires** `DATABASE_URL`: `src/study_query_llm/experiments/sweep_worker_main.py:1326-1327` — **inconsistent strictness** between entrypoints.
7. **Neon vs NeonDB naming:** `.env.example:71-72` documents `NEONDB_DATABASE_URL` style; `ingest_mcq_probe_json_to_sweep_db.py:139-140` reads **`NEON_DATABASE_URL`** — **potential name mismatch** if operators follow `.env.example` only.
8. **Alembic:** no `alembic` Python tree under project root (glob 0); migrations live as standalone scripts in `src/study_query_llm/db/migrations/` using `DATABASE_URL`.
9. **`tests/conftest.py`** does not set `DATABASE_URL`; tests typically construct `DatabaseConnectionV2` with explicit `sqlite` URLs.

---

**Limitation (explicit):** The combined `scripts/` grep for resolution tokens is the exhaustive mechanical list used here; for any script not in that list, the claim is "no embedded token in file," not "never loads DB at runtime through imports." For import-time DB use without literals, the caller chain must be read.
