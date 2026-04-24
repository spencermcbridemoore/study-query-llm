# Subagent 5 — Test vs Production Isolation (raw)

Date: 2026-04-24
Scope: how tests / fixtures route DB and storage; potential leakage into Jetstream.
Method: readonly explore subagent.

---

## A. Pytest configuration & global fixtures

- `pytest.ini` (root) sets `testpaths = tests` and `addopts = -ra --strict-markers`. **No global env override** beyond markers.
- `tests/conftest.py` sets `os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")` at import time, **only when unset** (`12-21`). It also imports `_clear_default_singletons` from `services/__init__.py` and clears default `ArtifactService` singletons between tests (`30-58`).
  - **Implication:** if a developer runs `pytest` with `DATABASE_URL=<jetstream>` exported in their shell, **`conftest.py` does not override it**. Tests that build a `DatabaseConnection` from default env will hit Jetstream.
- `tests/integration/conftest.py` selectively spins up Postgres via testcontainers when `RUN_INTEGRATION_DB_TESTS=1`. Otherwise integration tests skip. **Never points at Jetstream in code.**
- No `tests/conftest.py` or fixture sets `JETSTREAM_DATABASE_URL` to a sandbox value.

---

## B. Storage fixtures

- `tests/conftest.py` patches `_resolve_default_backend` to **always return a local backend pointing at `tmp_path`** when invoked inside tests (`60-92`). Confirms tests cannot accidentally write to Azure Blob.
- Some integration tests (`tests/integration/test_artifact_service_dual_backend.py`) intentionally exercise the Azure path with `azurite` — explicit `monkeypatch.setenv(...)` and `requires_azurite` marker.

---

## C. Test write paths under audit

The following test modules exercise pipeline writers directly. Each one has been confirmed to use a sandbox session (in-memory SQLite or testcontainers Postgres):

- `tests/pipeline/test_snapshot_runner.py`
- `tests/pipeline/test_twenty_newsgroups_snapshot.py`
- `tests/pipeline/test_acquire.py`
- `tests/pipeline/test_parse.py`
- `tests/pipeline/test_embed.py`
- `tests/pipeline/test_analyze_runner.py`
- `tests/services/test_artifact_service.py`

None of these use `JETSTREAM_DATABASE_URL` directly; all sessions are constructed via the `db_session` fixture chain that ultimately resolves to either SQLite or testcontainers Postgres.

---

## D. CLI / script invocations from tests

- `tests/scripts/test_sync_from_online.py` mocks `start_tunnel` and uses two SQLite engines for source/destination — no actual Jetstream call.
- `tests/scripts/test_restore_jetstream_db.py` patches `subprocess.run` and verifies argv; no live restore.

---

## E. Holes & risks

1. **Shell-exported `DATABASE_URL` overrides the safe default.** `setdefault` does not reset the value if the developer already exported `JETSTREAM_DATABASE_URL` for a debugging session; pytest will then run pipeline writers against Jetstream.
2. **No assertion in `conftest.py` that the resolved URL is non-Jetstream.** Recommend a hard fail when `_jetstream_url_match(...)` returns true and `RUN_REAL_JETSTREAM_TESTS` is not set.
3. **Pipeline `runner.run_stage` is invoked in unit tests** with sandboxed sessions; if the sandbox check is removed, accidental Jetstream writes during local test runs become possible.
4. **`ArtifactService` singleton state leaks** were already addressed by `_clear_default_singletons`; remains brittle and tied to env vars.

---

## F. Summary

Tests do not currently leak into Jetstream **as long as developers do not export `JETSTREAM_DATABASE_URL` before running pytest**. The default SQLite fallback and patched storage backend protect canonical data, but the protection is purely conventional — no test fixture **asserts** the active DB lane.

The implementation plan's preflight banner + `WriteIntent` enforcement should additionally apply during test setup (e.g., via fixtures that declare `WriteIntent.SANDBOX`) so any drift is caught loudly.
