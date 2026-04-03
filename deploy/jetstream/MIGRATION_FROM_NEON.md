# Migrate PostgreSQL from Neon to Jetstream Compose

This runbook moves **all** database content from Neon (or any Postgres URL) to the `db` service in `docker-compose.jetstream.yml`.

## When to point the app at Jetstream

Keep **`DATABASE_URL` in `.env.jetstream` on the VM** aimed at **Neon** only if the Jetstream app should keep using Neon during the dump. For a normal cutover:

1. Dump from Neon (laptop or CI).
2. Copy the `.dump` file to the VM.
3. **Stop the Panel app** on Jetstream (`docker compose ... stop app`) so nothing holds DB sessions.
4. Run `restore_pg_dump_to_compose_db.sh` against the **local** Postgres container.
5. **Then** ensure `.env.jetstream` has  
   `DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}`  
   (already the default in `.env.jetstream.example`).
6. `docker compose ... up -d app` (or `start app`).

If `DATABASE_URL` on the VM was already `...@db:5432/...`, the app was using empty local data until you restore—that is fine; restore replaces content.

## One-time: create a dump (any machine with `pg_dump`)

Install PostgreSQL **client** tools so `pg_dump` is on `PATH`.

From the repo root (with Neon in `.env` as `DATABASE_URL`, or set `SOURCE_DATABASE_URL`):

```bash
python scripts/dump_postgres_for_jetstream_migration.py
```

Optional:

```bash
python scripts/dump_postgres_for_jetstream_migration.py --dry-run
python scripts/dump_postgres_for_jetstream_migration.py --source-url "postgresql://..."
```

Artifacts are written under `pg_migration_dumps/` (gitignored). **Treat dumps as secrets** (they contain full DB content).

## Copy dump to the VM

Use `scp`, WinSCP, or shared storage. Example:

```bash
scp pg_migration_dumps/neon_for_jetstream_*.dump user@jetstream-vm:~/migration.dump
```

## Restore on the Jetstream VM

**Full reset + restore (pgvector image, wipes DB volume):** after `git pull`, from `deploy/jetstream`:

```bash
chmod +x jetstream_pgvector_restore.sh restore_pg_dump_to_compose_db.sh
./jetstream_pgvector_restore.sh ~/migration.dump
```

This uses `docker-compose.jetstream.yml` with **`pgvector/pgvector:pg17`** so `CREATE EXTENSION vector` from a Neon dump succeeds. `pg_session_jwt` may still error (Neon-only); that is expected.

**Manual steps** (same result without the helper script):

```bash
cd /path/to/study-query-llm/deploy/jetstream
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream stop app
chmod +x restore_pg_dump_to_compose_db.sh
./restore_pg_dump_to_compose_db.sh ~/migration.dump
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream start app
```

Verify the app health endpoint and spot-check data in the DB.

## Alternative: incremental row sync

For **incremental** copy of v2 tables (not a full cluster-accurate backup), see `scripts/sync_from_online.py` and pass `--local-url` pointing at Jetstream Postgres if it is reachable from where you run the script. Use `pg_dump` / `pg_restore` when you need a **full** database copy.

## Extensions (e.g. pgvector)

The Jetstream compose file uses **`pgvector/pgvector:pg17`** for `db` so `vector` matches Neon. **`pg_session_jwt`** is Neon-specific and will not install on self-hosted Postgres; restore may log 2 errors for it—safe to ignore unless you rely on Neon JWT features.
