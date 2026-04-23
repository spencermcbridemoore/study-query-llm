# Jetstream Postgres State Timeline (Apr 22 v5 Cutover → Dormant)

Status: living  
Owner: ops-maintainers  
Last reviewed: 2026-04-23

This note captures the Jetstream Postgres lifecycle around the Apr 22 2026 v5 cutover so that future operators can reason about the chronology without re-deriving it from `db-backups` filenames and the `group_graph_audit_log` survivor table. It is informational; the canonical source of truth for any restore decision is the dump set in the `db-backups` Azure container plus the manifests under `backup_pg_dumps/*.manifest.json`.

## Summary

Between Apr 22 01:33Z and Apr 23 03:14Z, Jetstream went through a deliberate cutover wipe, a brief 6-dataset rebuild, and an unexplained second drop that left the schema empty of every SQLAlchemy-managed v2 table. Only `group_graph_audit_log` survives — it was created out-of-band on Jetstream and is not part of `BaseV2.metadata`, so `metadata.drop_all()`-style resets do not touch it. The 25 audit-log rows that remain document the brief rebuild but their referenced application rows are gone. **The HEAD `models_v2.py` schema is byte-identical to the schema that ran during the brief rebuild and to the schema captured in every dated dump in `db-backups`**, so any of the prior dumps can be restored without migration shims.

## Wipe-Vector Status (A+B Guardrails)

The previously observed wipe path is now structurally blocked in code:

- `BaseDatabaseConnection.drop_all_tables()`/`recreate_db()` now enforce a destructive-op guard before any DB I/O.
- SQLite remains allowed by default; non-SQLite requires `SQLLM_ALLOW_DESTRUCTIVE_DDL=1`.
- If the active target matches `JETSTREAM_DATABASE_URL` (normalized host + port + dbname), destructive operations are denied even with override.
- `tests/test_db/test_repository_v2.py` no longer reads `DATABASE_URL`; it is pinned to in-memory SQLite to prevent accidental remote/tunneled teardown paths.
- Guard behavior is regression-tested in `tests/test_db/test_destructive_guard.py` using `.invalid` URLs and explicit branch assertions.

## Timeline (UTC)

| When | Event | Evidence |
|---|---|---|
| 2026-04-18 12:03Z | Last MCQ-rich snapshot | `db-backups/jetstream_for_local_20260418_120348Z.dump` (8.3 MB; manifest table_counts 8 109 raw_calls, 1 527 groups, 309 call_artifacts) |
| 2026-04-22 00:47Z | First pre-cutover snapshot | `db-backups/jetstream_for_local_20260422_004755Z.dump` (235 MB; manifest written post-wipe so table_counts is `{}`) |
| 2026-04-22 01:33Z | Second pre-cutover snapshot (canonical) | `db-backups/jetstream_for_local_20260422_013304Z.dump` (235 MB) |
| ~2026-04-22 01:34Z | **Cutover wipe (intentional)** — application tables dropped, `group_graph_audit_log` preserved | Audit log id sequence resets to 1 immediately after this |
| 2026-04-22 01:35Z → 07:23Z | Brief 6-dataset rebuild (banking77, ausem, sources_uncertainty_qc, estela, semeval2013_sra_5way) | 25 audit-log inserts captured (groups + group_links). Source artifacts in `artifacts-dev/jetstream-v5-cutover-20260422/`. |
| 2026-04-22 07:23:36Z | Last audit-log entry | `group_links` insert for semeval2013_sra_5way |
| 2026-04-22 07:23Z → 2026-04-23 03:14Z | **Second drop (unexplained)** — application tables removed; no bracketing dump taken | Cause not in git: only `BaseDatabaseConnection.drop_all_tables()` / `recreate_db()` produce this shape, and those are only called from unit tests (`tests/test_db/test_connection.py`, `tests/test_db/test_repository_v2.py`). Most consistent with a test fixture or manual psql `DROP TABLE` against `JETSTREAM_DATABASE_URL`. |
| 2026-04-23 03:14Z | Dormant-state snapshot taken to close the chronology gap | `db-backups/jetstream_dormant_post_v5_cutover_20260423_031452Z.dump` (14.7 KB; restoring it produces an empty schema with the audit-log survivor table only) |

## Schema Compatibility

`git log` filtered to `src/study_query_llm/db/models_v2.py`, `src/study_query_llm/db/connection_v2.py`, `src/study_query_llm/db/_base_connection.py`, and `src/study_query_llm/db/migrations/` returns **no commits between 2026-04-22 07:23Z and HEAD**. The only DB-adjacent change in the window is `raw_call_repository.py` (commit `3a3fee6`, audit remediation gates) — repository-helper behavior, not DDL. Practical implication: `pg_restore` of any dump in the table above into a fresh Jetstream produces a schema compatible with HEAD, and `db.init_db()` against the empty Jetstream creates that same schema.

## Recovery Options

If Jetstream needs to come back into active service, pick one:

1. **Restore Apr 18 dump** (`jetstream_for_local_20260418_120348Z.dump`, MCQ-rich) — preserves the MCQ lineage rows that Panel's `SweepQueryService.get_mcq_metrics_df()` needs. Use `scripts/restore_pg_dump_to_local_docker.py` for the local clone path or follow `deploy/jetstream/MIGRATION_FROM_NEON.md` for the Jetstream-side compose-DB restore. Source MCQ blobs are still in `artifacts-dev/...` so the restored URIs resolve.
2. **Restore Apr 22 01:33Z dump** (`jetstream_for_local_20260422_013304Z.dump`, 235 MB) — preserves the most recent pre-cutover state including any embedding vectors that grew between Apr 18 and Apr 22.
3. **`db.init_db()` against the empty Jetstream** — clean schema only. Loses both wipes' worth of data but every artifact (parquet, embedding matrices, downloads) remains in `artifacts*` blob storage, so re-running `acquire`/`parse`/`embed` reproduces the rows. The 6 datasets that ran during the brief rebuild (banking77, ausem, sources_uncertainty_qc, estela, semeval2013_sra_5way) have their source artifacts under `artifacts-dev/jetstream-v5-cutover-20260422/` and can be rebuilt without re-downloading.

Option (3) is the right default if Jetstream's role is being re-evaluated. Options (1) or (2) are appropriate if Panel-visible MCQ data or the embedding matrices need to be live in DB form before any new pipeline work.

## Operator Caveats

- `scripts/verify_db_backup_inventory.py` returns exit code 1 against the current dormant Jetstream because its `_table_counts()` query expects `raw_calls` etc. to exist. The "FAIL — Jetstream table counts could not be loaded" message is the expected signal of dormancy, not a script bug. The same run still validates manifest ↔ blob consistency and prints `OK manifest …` for every dated dump, including the dormant-state entry.
- The dormant-state dump is named `jetstream_dormant_post_v5_cutover_…` rather than the `jetstream_for_local_…` pattern so that `scripts/upload_jetstream_pg_dump_to_blob.py`'s "newest jetstream dump" auto-discovery skips it. This prevents an empty dump from being treated as the latest restore candidate.
- Local Docker Postgres (`LOCAL_DATABASE_URL`) has not been used since Mar 29; recent pipeline work (e.g., `scripts/snapshot_inventory.py`) uses ephemeral SQLite tempdirs and never touches Jetstream.
- The MCQ JSON export at `db-backups/mcq_export_20260418T120421Z.json` is metadata-only (URIs, no blob bytes). The companion blob copy via `scripts/archive_mcq_artifact_blobs.py` was never executed; the source MCQ blobs are still live in `artifacts-dev/...`. If MCQ visibility in Panel is required and a `restore_mcq_db_from_json.py` does not exist by then, prefer restoring the Apr 18 dump (option 1 above) over building a JSON re-ingest path.
