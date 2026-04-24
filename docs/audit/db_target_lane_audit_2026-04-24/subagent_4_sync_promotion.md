# Subagent 4 — Sync / Promotion / Direction (raw)

Date: 2026-04-24
Scope: every script and tool that moves data between Jetstream and any local DB (Docker Postgres / SQLite). Direction, side, idempotency, dry-run.
Method: readonly explore subagent.

---

## A. Inventory of cross-database tools (script-by-script)

### Online → Local (read-only on Jetstream)

#### `scripts/sync_from_online.py`
- **Direction:** Online (Jetstream) → Local Docker Postgres. **Online side never written.**
- **Source:** `--online-url` or `JETSTREAM_DATABASE_URL` (resolved via `start_jetstream_postgres_tunnel.py:start_tunnel`). Lines: source resolution `212-227`; tunnel start `228-243`; `online_engine` build `247`; **transactions `engine.begin()` (read) `277, 289`** (no DML in source-side blocks).
- **Destination:** `--local-url` or `LOCAL_DOCKER_POSTGRES_URL`. Resolution `212`; engine `246`. Writes via `engine.begin()` with **`temp_session.execute(text("CREATE SCHEMA…"))`** `255`, `_drop_dependent_views_in_local` `256`, `_drop_local_tables` `257`, `Base.metadata.create_all(engine)` `271`, **`temp_session.execute(insert(table), batch)`** in `_copy_table_data` `145-149`. **All destructive ops are local-side.**
- **Auth:** Online: SSH via `start_tunnel(...)` (port forward) `228-243`. Local: standard psycopg2 connection.
- **Idempotency:** `--temp-suffix` (default `_sync_temp`) creates side schema `_sync_temp` `203, 252`; success path **renames** `_sync_temp.*` → `public.*` after dropping current public tables `277-284`. Re-run safe by suffix uniqueness.
- **Dry-run / Preflight:** No `--dry-run`. Preflight info via `_jetstream_target_info` printed `223-227`. **Promotion confirmation:** `--yes` skip flag `217`; otherwise interactive `_prompt_yes_no("Continue?", ...)` `230` and a final `_prompt_yes_no("Promote temporary schema...", default_no=True)` `277`. Failure cleanup drops temp schema `283-287`.
- **DDL on Jetstream?** **Never.** Only `engine.connect()`/`engine.begin()` reads from `online_engine`. Code paths confirm.

#### `scripts/sync_jetstream_to_local_via_dump_restore.py`
- **Direction:** Online (Jetstream) → Local Docker Postgres via `pg_dump`/`pg_restore`.
- **Source:** Jetstream URL via tunnel/`JETSTREAM_DATABASE_URL`; `pg_dump --format=custom` writes to a `.dump` file `215-275`.
- **Destination:** Local Postgres. **`DROP DATABASE`/`CREATE DATABASE`** of `public` is invoked **after explicit confirmation**; verifies the URL is `localhost`/`127.0.0.1` `155-181`.
- **Auth:** Source via Jetstream tunnel; destination via local credentials.
- **Idempotency:** Adds `--keep-dump` to retain artifact; default cleans dump after restore.
- **Dry-run / Preflight:** Banner displays both URLs with masked passwords, `_print_preflight()` invoked before any irreversible action; `--yes` to skip confirmation `90-118`.
- **DDL on Jetstream?** **Never** — only `pg_dump` (read).

#### `src/study_query_llm/cli/connect_jetstream.py`
- **Direction:** **Read-only** psql shell to Jetstream via tunnel. Print-only.
- **Source:** Jetstream tunnel; **no SQL executed by the script itself** beyond what the human types.
- **Destination:** None.
- **Auth:** SSH tunnel via `start_jetstream_postgres_tunnel`.
- **Idempotency:** N/A.
- **Dry-run / Preflight:** Yes — preflight banner is the entire purpose of this entrypoint.
- **DDL on Jetstream?** Only what the human types; **no policy enforcement** in code.

### Local → Online (Jetstream-side writes)

**No direct Local → Jetstream DML script exists.** Searches against the `Sync*` and `restore*` script names returned nothing that pushes from local Postgres / SQLite back into Jetstream. The only pathway that places data **into Jetstream** is `scripts/restore_jetstream_db.py` from a `.dump` file (see next).

#### `scripts/restore_jetstream_db.py`
- **Direction:** Dump file → Jetstream (or any target). Effectively **promote** of a snapshot.
- **Source:** A `.dump` file (`--dump-file`) produced by an upstream tool (typically `pg_dump`).
- **Destination:** `--target-url` or `JETSTREAM_DATABASE_URL` (`118-130`).
- **Auth:** Direct Postgres credentials (no tunnel auto-start; expects a routable URL).
- **Idempotency:** **`--clean`** flag passes `--clean` to `pg_restore` (data drop on restore). Default keeps existing data.
- **Dry-run / Preflight:** Banner prints target URL with masked password `131-150`; `--yes` skip `41-58`. `--list` enumerates dump contents without applying. Pre-action banner explicitly warns when the URL hostname looks like Jetstream.
- **DDL on Jetstream?** **Yes** — `pg_restore` may CREATE/DROP/INSERT into the target. Acknowledged path: only safe to use with a fresh, ratified dump.

#### `scripts/dump_jetstream_db.py`
- **Direction:** Jetstream → local `.dump` file.
- **Auth/Source/Destination:** Tunnel/JS URL → local file.
- **Side effects on Jetstream:** Read-only `pg_dump`.

### Local-only and unrelated

- **`scripts/start_jetstream_postgres_tunnel.py`** — opens SSH tunnel; no DML.
- **`scripts/verify_db_backup_inventory.py`** — read-only inventory of dump artifacts.
- **`scripts/snapshot_inventory.py`** — local-only counts; no Jetstream DML.
- **`scripts/run_sweeps.py`**, **`scripts/sweep_worker_main.py`** (under `experiments/`) — write through `ArtifactService` and ORM into whichever DB `DATABASE_URL` resolves to. **Could be Jetstream or local**; **no preflight banner**, **no intent gate**.

---

## B. Direction matrix

| Script | Reads from | Writes to | DDL? | Confirms before destructive op? |
|--------|-----------|-----------|------|--------------------------------|
| `sync_from_online.py` | Jetstream | Local Docker | Local only | Yes (interactive + `--yes` opt-out) |
| `sync_jetstream_to_local_via_dump_restore.py` | Jetstream | Local Docker | Local only | Yes (banner + confirm) |
| `dump_jetstream_db.py` | Jetstream | local file | None | n/a |
| `restore_jetstream_db.py` | local dump | Jetstream / target | Yes (target) | Yes (banner + `--yes`) |
| `connect_jetstream.py` | n/a | n/a | n/a | n/a |

There is **no automated Local → Jetstream DML synchronization** in the repo. The only "local → online" paths are the **explicit** `restore_jetstream_db.py` (dump-replace) and **ad-hoc, accidental** writes via pipeline tools when `DATABASE_URL` happens to be Jetstream while artifact backend is local.

---

## C. Preflight, banners, and assertions

- `scripts/sync_from_online.py:223-227` (`_jetstream_target_info`) and `_prompt_yes_no` calls.
- `scripts/sync_jetstream_to_local_via_dump_restore.py:_print_preflight` (banner with masked passwords, paths, `--yes`).
- `scripts/restore_jetstream_db.py:_format_target_label` and the explicit "Type the database name to confirm" gating.
- `connect_jetstream.py` prints sanitized URL components and the local-bound port.
- `DatabaseConnection._assert_destructive_operation_allowed` (`src/study_query_llm/db/_base_connection.py:147-171`) — Jetstream URL match blocks destructive DDL unless `SQLLM_ALLOW_DESTRUCTIVE_DDL` is truthy.

**Gaps**:
- **No** preflight banner on `run_sweeps.py` / `sweep_worker_main.py` / pipeline `runner.py:run_stage`.
- **No** assertion of artifact backend vs DB target (only sweep ingestion's `_assert_uri_backend_compatible` runs after-the-fact).

---

## D. Direction integrity findings

1. **Online → Local is well-defended**: dedicated tools, banners, and per-call confirmation; destructive operations only on the local schema.
2. **Local → Online is unintentionally possible** through pipeline tools whenever `DATABASE_URL` resolves to Jetstream while artifact backend is local; **no script enforces a "promote/flush" pathway**.
3. **`restore_jetstream_db.py` is the only sanctioned promotion**, and it requires a vetted dump.
4. **No code path emits a "dirty local DB" warning** that would help the operator notice forgotten promotion intent.

These gaps motivate the proposed `WriteIntent` chokepoint and Postgres-side URI constraints captured in the implementation plan.
