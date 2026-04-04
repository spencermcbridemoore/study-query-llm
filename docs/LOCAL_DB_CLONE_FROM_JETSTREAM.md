# Clone Jetstream Postgres into local Docker

This runbook backs up your **current local** Docker Postgres, then replaces it with a **full copy of Jetstream’s** database (same idea as [`deploy/jetstream/MIGRATION_FROM_NEON.md`](../deploy/jetstream/MIGRATION_FROM_NEON.md), but **source = Jetstream**, **target = local**).

## Prerequisites

- PostgreSQL **client** tools on your PATH: `pg_dump`, `pg_restore`, `dropdb`, `createdb` (17.x recommended).
- Repo root `.env` with **`LOCAL_DATABASE_URL`** (e.g. `postgresql://study:study@localhost:5433/study_query_local`) and **`JETSTREAM_DATABASE_URL`** (via tunnel port, e.g. `127.0.0.1:5434`).
- **SSH tunnel** running: `python scripts/start_jetstream_postgres_tunnel.py` (leave it open for the Jetstream dump step).
- Local Docker DB: `docker compose --profile postgres up -d db`  
  The compose file uses **`pgvector/pgvector:pg17`** so restores match Jetstream. If you switch from plain `postgres:17`, **back up first**; changing images may require recreating the volume (only after you have a `.dump`).

## Phase A — Back up local (two copies)

1. **Gitignored artifact in the repo** (custom format, good for `pg_restore`):

   ```bash
   python scripts/dump_postgres_for_jetstream_migration.py --from-local
   ```

   Output: `pg_migration_dumps/local_pre_jetstream_clone_<timestamp>.dump`

   Optional: `--dry-run` to print the `pg_dump` command.

2. **Extra copy** — manually copy that `.dump` to an external drive or folder outside the repo (e.g. `D:\Backups\study-query-llm\`).

3. Optional: note row counts — `python scripts/verify_db_backup_inventory.py` (with `DATABASE_URL` / tunnel settings consistent with what you want to measure).

## Phase B — Dump Jetstream

With the tunnel still up:

```bash
python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream
```

Output: `pg_migration_dumps/jetstream_for_local_<timestamp>.dump`

## Phase C — Stop writers and restore into local

1. Stop anything using local Postgres (Panel with `LOCAL_DATABASE_URL`, other tools connected to `localhost:5433`).
2. Restore (drops and recreates the database named in `LOCAL_DATABASE_URL`, then `pg_restore`):

   ```bash
   python scripts/restore_pg_dump_to_local_docker.py pg_migration_dumps/jetstream_for_local_<timestamp>.dump
   ```

   Dry run: add `--dry-run`.

   If `dropdb` fails, close remaining connections and retry. Alternative: `--skip-recreate` to run only `pg_restore --clean --if-exists` against the existing database (advanced).

`pg_restore` may exit with status **1** with warnings (e.g. missing Neon-only extensions); verify data below.

## Phase D — Verify

Point checks at local only, e.g.:

```bash
set DATABASE_URL=postgresql://study:study@localhost:5433/study_query_local
python scripts/sanity_check_database_url.py
python scripts/verify_db_backup_inventory.py
```

With the tunnel up, **Local** and **Jetstream** table counts should **match**.

## Rollback

Keep the **Phase A** `local_pre_jetstream_clone_*.dump`. To restore the old local DB:

```bash
python scripts/restore_pg_dump_to_local_docker.py pg_migration_dumps/local_pre_jetstream_clone_<timestamp>.dump
```

## Related

- Tunnel: [deploy/jetstream/LOCAL_DEV_TUNNEL.md](../deploy/jetstream/LOCAL_DEV_TUNNEL.md)
- Scripts: [`scripts/dump_postgres_for_jetstream_migration.py`](../scripts/dump_postgres_for_jetstream_migration.py), [`scripts/restore_pg_dump_to_local_docker.py`](../scripts/restore_pg_dump_to_local_docker.py)
