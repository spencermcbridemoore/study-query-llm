# DB Target / Artifact Lane Audit & Remediation Plan

**Date:** 2026-04-24
**Author:** Claude (Opus) — written for Codex audit
**Scope:** Architectural ambiguity in `DATABASE_URL` resolution and the persistence
of local-filesystem URIs in Jetstream's canonical artifact tables.
**Companion artifacts (this folder):**
- [`subagent_1_target_resolution.md`](./subagent_1_target_resolution.md) — DB URL resolution
- [`subagent_2_write_paths.md`](./subagent_2_write_paths.md) — DB write-path inventory
- [`subagent_3_artifact_uri_lifecycle.md`](./subagent_3_artifact_uri_lifecycle.md) — artifact URI lifecycle
- [`subagent_4_sync_promotion.md`](./subagent_4_sync_promotion.md) — sync / promotion direction
- [`subagent_5_test_isolation.md`](./subagent_5_test_isolation.md) — test vs production isolation
- [`subagent_6_doc_parity.md`](./subagent_6_doc_parity.md) — documentation vs implementation parity
- [`live_count.py`](./live_count.py) — read-only SQL probe used to enumerate live data
- [`live_count_output.txt`](./live_count_output.txt) — captured stdout from the probe (Jetstream, this date)

> **For the auditor (Codex):** Every numeric claim, file path, and line number in
> this document is traceable either to the subagent reports above or directly to
> the code citation provided inline. When a finding overlaps with what was
> reported in a previous Opus↔Codex conversation, the relevant claim is repeated
> here so the audit stands alone. Where the assistant's prior conversation
> contradicted the present synthesis (one example called out below), the present
> synthesis takes precedence and the contradiction is noted.

---

## 0. TL;DR

1. **Code-level reality:** `DATABASE_URL` is a single, free-form switch. Anyone
   who exports a Jetstream URL (or, equivalently, runs through a Jetstream SSH
   tunnel) gets a "real-Jetstream" connection from any pipeline stage,
   ingestion script, worker, or migration — without any preflight that says
   "you are about to write to canonical Jetstream." There is no separate
   `JETSTREAM_DATABASE_URL` consumer for production reads/writes; the variable
   is referenced **only** by the destructive-DDL guardrail and a handful of
   ops scripts (subagent 1 §B, §C; `_base_connection.py:147-172`).
2. **Artifact backend is decoupled from DB target:** `ArtifactService`'s
   `_resolve_default_backend` never reads `DATABASE_URL` /
   `JETSTREAM_DATABASE_URL`. Default is `local` when `ARTIFACT_STORAGE_BACKEND`
   is unset; non-strict azure failure silently falls back to local
   (`artifact_service.py:86-145`; subagent 3 §B). So a process can be writing
   rows to Jetstream while its blob URIs are absolute Windows paths.
3. **Live evidence (Jetstream this date):** **11 of 31** rows in
   `call_artifacts.uri` are local-filesystem paths (35.5%); the same 11 are
   mirrored into `raw_calls.response_json[uri]`. All 11 belong to one dataset
   (`bank77_contrast`). The other artifact-bearing JSON columns currently hold
   no URIs (`live_count_output.txt:8-55`).
4. **No "flush" mechanism exists.** The user's intuition is correct: there is
   no batched-promotion path that takes local-DB writes and applies them to
   Jetstream later. The only sanctioned local→Jetstream promotion is the
   manual two-step `pg_dump` → transfer dump file to Jetstream VM →
   `deploy/jetstream/restore_pg_dump_to_compose_db.sh` (a wholesale replace
   of the target compose DB, not a merge). Subagent 4 §A originally cited
   `scripts/restore_jetstream_db.py` and `scripts/dump_jetstream_db.py` for
   this path; both names are wrong (the files do not exist as of
   commit 8e95253). The actual scripts/workflow is documented in §9 and
   §12.
5. **Contradiction with assistant context block:** the live transcript context
   block claimed `_assert_uri_backend_compatible` lives in
   `src/study_query_llm/services/artifact_service.py`. That is **incorrect**.
   The function is defined and used **only** in
   `scripts/ingest_sweep_to_db.py` lines `312-329`, invoked at `355` (subagent
   3 §E and subagent 1 §F). This contradiction is preserved in this document so
   the auditor can verify the corrected location independently.
6. **Recommended fix (synthesised across subagent findings):** introduce a
   `WriteIntent` enum (`CANONICAL`, `READ_MIRROR`, `SANDBOX`) and a single
   mandatory chokepoint inside `BaseDatabaseConnection.__init__` that:
   - resolves `connection_string` against `CANONICAL_DATABASE_URL` /
     `LOCAL_DATABASE_URL` to determine the actual lane (`CANONICAL_DATABASE_URL`
     defaults to `JETSTREAM_DATABASE_URL` for one-release back-compat — see
     §6.0 on role-vs-identity naming);
   - asserts the caller's declared `WriteIntent` matches the actual lane;
   - prints a verbose preflight banner;
   - for `CANONICAL` writes, asserts that the `ArtifactService` resolves to a
     blob backend.
   In parallel, add a Postgres `CHECK` constraint on `call_artifacts.uri` that
   only allows `https://*.blob.core.windows.net/*` URIs on the canonical lane
   (currently Jetstream) so the wrong code path can fail at the database, not
   silently corrupt the source of truth. The full plan is in §6 below; the
   minimum-viable shipment is highlighted at §6.9; remediation of the 11
   polluted rows is in §7.

---

## 1. Why this audit exists (problem statement)

The user observed that the assistant in a prior session described a "dirty"
state where some snapshot artifacts were "local-path URIs, not blob URIs." The
user's concern was not the cosmetics of the URI string — it was the
architectural implication: **if a local DB run can be confused for canonical
work, then the entire premise of "Jetstream is the source of truth" is broken,
because there is no way to tell after the fact which writes were canonical and
no way to promote local writes that should have been canonical.**

That concern has three independent failure surfaces:

1. **DB target is ambiguous.** Tools agree to use `DATABASE_URL`, but
   `DATABASE_URL` can be Jetstream, local Docker Postgres, or a SQLite file —
   the same code path runs in any of those modes (subagent 1 §C, subagent 2
   §A).
2. **Artifact storage target is independently ambiguous.** The artifact backend
   is selected by a separate env var (`ARTIFACT_STORAGE_BACKEND`) that defaults
   to `local`. There is no code that asserts the storage target is consistent
   with the DB target (subagent 3 §B, §G).
3. **No "flush" / promotion exists.** Local-DB writes are not held in a staging
   area for later Jetstream promotion. They simply live wherever the operator
   pointed `DATABASE_URL` at the moment of the write. The only way work gets
   into Jetstream is to have written it directly to Jetstream (subagent 4 §A
   inventory).

The combination is what makes this a serious architectural defect: a single
forgotten env var can cause real production work to silently land in a local
file with no way to recover or even detect the loss.

---

## 2. Methodology

The audit was performed in five phases:

1. **Six parallel readonly explore subagents** were dispatched concurrently,
   each with a narrowly-scoped investigation. Each subagent produced a raw
   report (`subagent_*.md` in this folder). The subagents were instructed to
   cite `file:line` for every claim, to avoid speculation, and to flag
   contradictions with the prior conversation context. The six topics:
   1. **Target resolution** — every site that resolves a DB URL.
   2. **Write paths** — every code path that writes to a DB.
   3. **Artifact URI lifecycle** — backend selection → DB persistence.
   4. **Sync / promotion direction** — every cross-DB tool.
   5. **Test isolation** — pytest / fixture leakage risks.
   6. **Doc parity** — runbook / contract drift.
2. **Synthesis** — overlaps, contradictions, and gaps were reconciled into the
   single ground-truth picture in this document.
3. **Live count** — a read-only SQL probe ([`live_count.py`](./live_count.py))
   was executed via SSH tunnel against the live Jetstream Postgres to quantify
   the existing damage. The output is captured verbatim in
   [`live_count_output.txt`](./live_count_output.txt).
4. **Audit document (this file)** — written so Codex can independently verify.
5. **Implementation plan (§6)** — folded into this document because the plan is
   only meaningful in the context of the findings.

The subagent reports are intentionally retained as raw artifacts so the
auditor can spot-check whether this synthesis distorts them. If a citation in
this document seems off, the auditor should consult the corresponding
subagent file before consulting the source code.

---

## 3. Live Jetstream count — empirical baseline

Source: `live_count.py` executed against
`postgresql://***@127.0.0.1:5434/study_query_jetstream?sslmode=prefer`
(SSH-tunneled `JETSTREAM_DATABASE_URL`). Output:
[`live_count_output.txt`](./live_count_output.txt) (timestamp on the file is
2026-04-24 12:33 local).

### 3.1 `call_artifacts.uri` (the canonical pointer column)

```
total_rows: 31
by_class: blob=20, local_path=11
```

| `artifact_type` | `class` | count |
|-----------------|---------|-------|
| dataset_subquery_spec | local_path | 6 |
| dataset_subquery_spec | blob | 4 |
| embedding_matrix | blob | 4 |
| dataset_acquisition_file | blob | 3 |
| dataset_acquisition_manifest | blob | 3 |
| dataset_canonical_parquet | blob | 3 |
| dataset_dataframe_manifest | blob | 3 |
| dataset_acquisition_file | local_path | 2 |
| dataset_acquisition_manifest | local_path | 1 |
| dataset_canonical_parquet | local_path | 1 |
| dataset_dataframe_manifest | local_path | 1 |

**All 11 local paths share a common prefix:**

```
C:\Users\spenc\Cursor Repos\study-query-llm\artifacts\bank77_contrast\{15..22}\...
```

(`live_count_output.txt:24-34`).

These are absolute Windows paths on the developer's workstation. Any consumer
that resolves `call_artifacts.uri` — analyze stage, panel app, pipeline
re-runs, downstream studies — that does not run on this exact workstation
**will fail to load these artifacts.** Even on this workstation, if the
artifact directory is moved or cleaned, the pointer is dead.

### 3.2 `raw_calls.response_json[uri]`

```
total_rows: 4381
rows_with_uri_substring: 31
by_class (over extracted URIs): blob=20, local_path=11
```

The 31 affected rows mirror the `call_artifacts.uri` rows one-for-one
(`live_count_output.txt:38-55`). This is consistent with
`artifact_service.py:434-439`, which inserts a placeholder `RawCall` row
carrying `response_json={"uri": uri}` whenever a non-`raw_call`-bound
artifact is persisted (subagent 3 §C, table row "`artifact_service.py:473-536`").

### 3.3 Other artifact-bearing JSON columns

```
groups.metadata_json[artifact_uri]:    rows=22, with-key=0
analysis_results.result_json[uris]:    rows=0
provenanced_runs.result_ref:           rows=0
orchestration_jobs.result_ref:         rows=0
```

(`live_count_output.txt:60-92`). The pollution risk in those columns is real
(subagent 3 §G's "no constraint" finding) but **not yet realized** in
Jetstream. We have a window in which we can add the proposed `WriteIntent`
chokepoint and constraints before any of those columns are populated.

### 3.4 Interpretation

- **Concentration:** the damage is concentrated in `call_artifacts` /
  `raw_calls` for one dataset (`bank77_contrast`).
- **Identity of the writer:** the path prefix
  `C:\Users\spenc\Cursor Repos\study-query-llm\artifacts\bank77_contrast\` and
  the artifact types match the snapshot stage's writer
  (`pipeline/snapshot.py:303-315`) and acquire/parse stages
  (`pipeline/acquire.py:138-166`, `pipeline/parse.py:308-338`) (subagent 3
  §C). The most likely scenario: a developer (almost certainly the user) ran
  the bank77 pipeline locally with `DATABASE_URL` pointing at Jetstream while
  `ARTIFACT_STORAGE_BACKEND` was unset (default `local`) or set to a backend
  that fell back to local non-strictly.
- **Severity:** the rows are not corrupted; they are unreachable from any
  other host. They block anyone except the user (on this workstation, in this
  exact path) from re-resolving these specific snapshot lineages.

This is the smoking gun for the §6 implementation plan. The numbers are small
enough that remediation is feasible (§7); they are large enough to prove the
problem class is not theoretical.

---

## 4. Findings — why this happened

The findings below are organised by the four invariants the system needs but
does not enforce. Each finding is sourced from one or more subagent reports.

### F1. The DB connection accepts any URL silently

`BaseDatabaseConnection.__init__` takes a raw connection string and calls
`create_engine` with no validation about which lane the URL points at:

```105:113:src/study_query_llm/db/_base_connection.py
    def __init__(self, connection_string: str, *, echo: bool = False, **engine_kwargs):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=echo, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info(
            "Initialized database connection: %s", _mask_password(connection_string)
        )
```

There is no caller-side declaration of intent ("I am opening a CANONICAL
session") and no runtime banner saying "the URL you passed resolves to
Jetstream." Subagent 1 §A documents that this class is the **only** chokepoint
for SQLAlchemy connections in the application code path (the script-only
exceptions are listed in subagent 1 §D).

### F2. `Config` silently defaults `DATABASE_URL` to local SQLite

```95:121:src/study_query_llm/config.py
@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str

    def __post_init__(self):
        """Provide default if not set."""
        if not self.connection_string:
            self.connection_string = "sqlite:///study_query_llm.db"


class Config:
    ...
    def __init__(self):
        """Load configuration from environment."""
        raw_db = os.getenv("DATABASE_URL", "") or ""
        if not str(raw_db).strip():
            raw_db = "sqlite:///study_query_llm.db"
        self.database = DatabaseConfig(connection_string=str(raw_db).strip())
```

Subagent 1 §C, §G documents that **any** caller who imports `config` and
forgets to set `DATABASE_URL` quietly writes to a local SQLite file in CWD.
This is the worst possible default for a "Jetstream is source of truth"
project: forgetting an env var produces a silent, zero-error, ungovernable
side branch.

### F3. Pipeline / worker entrypoints disagree on strictness

Subagent 1 §F:

- `scripts/run_bank77_pipeline.py:323-325` — **raises** if `DATABASE_URL`
  unset.
- `scripts/ingest_sweep_to_db.py:487-488` — uses `config.database.connection_string`
  → falls back to **local SQLite** if unset.
- `experiments/sweep_worker_main.py:1326-1327` — **raises** if `DATABASE_URL`
  unset.
- `experiments/runtime_supervisors.py:199-201` — uses `os.environ["DATABASE_URL"]`
  directly (raises on unset, but has no lane check).

The strictness of "must set `DATABASE_URL`" is per-script convention and
inconsistent across the codebase. None of these entrypoints check **which** DB
the URL points at.

### F4. `JETSTREAM_DATABASE_URL` is consulted only by the destructive-DDL guard

Subagent 1 §B and `_base_connection.py:147-172` show that
`JETSTREAM_DATABASE_URL` is read **only** to decide whether `drop_all_tables`
/ `recreate_db` should be refused. There is no positive consumer that says:

> "If `DATABASE_URL` matches `JETSTREAM_DATABASE_URL`, this is a CANONICAL
> session — fail closed unless the caller explicitly asked for canonical."

The destructive-DDL guard is the right model — it should be generalised to
every write site, not limited to drop/recreate.

### F5. Artifact backend selection ignores the DB target entirely

```86:145:src/study_query_llm/services/artifact_service.py
    def _resolve_default_backend(self, artifact_dir: str):
        ...
        backend_type = (os.environ.get("ARTIFACT_STORAGE_BACKEND") or "local").strip().lower()
        runtime_env = (os.environ.get("ARTIFACT_RUNTIME_ENV") or "dev").strip().lower()
        strict_mode = self._is_truthy(os.environ.get("ARTIFACT_STORAGE_STRICT_MODE"))
        if runtime_env in {"stage", "prod"}:
            strict_mode = True

        if backend_type == "local" and strict_mode:
            raise ValueError(...)

        if backend_type == "azure_blob":
            ...
            try:
                return StorageBackendFactory.create(...)
            except (ValueError, ImportError) as e:
                if strict_mode:
                    raise RuntimeError(...) from e
                logger.warning(
                    "ARTIFACT_STORAGE_BACKEND=azure_blob requested but backend unavailable: %s. "
                    "Falling back to local.",
                    e,
                )
                backend_type = "local"
        if backend_type == "local" or not backend_type:
            return StorageBackendFactory.create("local", base_dir=artifact_dir)
```

Subagent 3 §A, §B, §G:

- Default is `local`, not `azure_blob`. Documentation (subagent 6 §B row 2)
  treats `azure_blob` as the production default; reality contradicts.
- "Strict mode" only fires when `ARTIFACT_RUNTIME_ENV` is `stage` / `prod`. A
  developer running `ARTIFACT_RUNTIME_ENV=dev` (the default) with
  `ARTIFACT_STORAGE_BACKEND=azure_blob` who hits an azure SDK error gets a
  silent fallback to local **with the DB still pointing at Jetstream**.
- **No code path** in `_resolve_default_backend` references `DATABASE_URL`,
  `JETSTREAM_DATABASE_URL`, or any DB string. The decision tree is purely
  `ARTIFACT_*` env vars.

This is the direct mechanism that produced the 11 polluted rows in §3.

### F6. URI columns have no type/format constraint

Subagent 3 §D enumerates the affected columns. The relevant facts:

- `call_artifacts.uri` is `String(1000)`, no `CheckConstraint`, no validator
  (`models_v2.py:227`).
- `raw_calls.response_json` is generic JSON.
- `analysis_results.result_json`, `groups.metadata_json`,
  `provenanced_runs.result_ref`, `orchestration_jobs.result_ref` similarly
  unconstrained.

There is no database-level safety net. A misbehaving writer can put any string
into the column, and it will persist.

### F7. The only "consistency" check is the wrong scope

Subagent 3 §E, subagent 1 §F:

`_assert_uri_backend_compatible` exists, but only inside
`scripts/ingest_sweep_to_db.py:312-329`, used during sweep ingestion. It
checks that an Azure URI matches an azure-typed backend (and vice versa). It
**does not**:
- prevent writing local URIs to Jetstream;
- compare URI shape against `DATABASE_URL`;
- run on any pipeline stage (acquire/parse/snapshot/embed/analyze).

(This is the contradiction with the assistant's prior context block — see §0
finding 5.)

### F8. There is no Local→Jetstream "promote" path

Subagent 4 §A, §D (with corrections from Codex audit on commit 8e95253 — see
§12):

The directional inventory shows:
- Online→Local is well-defended:
  - `scripts/sync_from_online.py` — incremental application-level sync from
    canonical (Jetstream) to local Postgres.
  - `scripts/dump_postgres_for_jetstream_migration.py --from-jetstream` —
    `pg_dump` of canonical to a local `.dump` file (used for backup or for
    seeding a fresh local clone).
  - `scripts/restore_pg_dump_to_local_docker.py` — restores a `.dump` file
    into the local Docker Postgres.
  - `scripts/upload_jetstream_pg_dump_to_blob.py` — uploads a `.dump` file
    to Azure blob `db-backups` for offsite retention.
- The only Local→Online path is the **manual two-step**: operator generates
  a `.dump` (e.g. `pg_dump` against the local DB or any target), transfers
  it to the Jetstream VM, and runs
  `deploy/jetstream/restore_pg_dump_to_compose_db.sh` (or the related
  `deploy/jetstream/jetstream_pgvector_restore.sh`) to replace the canonical
  compose DB. This is a **dump-replace**, not a merge.
- There is no script that takes incremental local writes and applies them to
  Jetstream.

> Subagent 4's original report named two scripts that do not exist in this
> repo (`scripts/restore_jetstream_db.py` and `scripts/dump_jetstream_db.py`).
> The Codex audit caught the discrepancy; the corrected script set above
> reflects the actual filesystem at commit 8e95253. See §12 for full
> provenance.

This empirically confirms the user's intuition: there is no batched "flush"
mechanic. If a developer accidentally writes to a local DB while believing
they were writing to Jetstream, the work is **stranded** — the only recovery
is a manual `pg_dump` of the local DB, manual transfer to the Jetstream VM,
and a destructive restore (which would also clobber any concurrent canonical
writes).

### F9. Tests do not assert their own lane

Subagent 5 §A, §E:

- `tests/conftest.py` does `os.environ.setdefault("DATABASE_URL",
  "sqlite:///:memory:")`. **`setdefault` does not override** an exported
  `DATABASE_URL`. A developer with `JETSTREAM_DATABASE_URL` exported and
  `DATABASE_URL=$JETSTREAM_DATABASE_URL` set in their shell will run pytest
  against Jetstream.
- No test fixture asserts that the resolved DB URL is non-Jetstream.

### F10. Documentation describes the desired contract, not the enforced one

Subagent 6 §B summarises the drift in detail. The pattern is consistent:
docs say "Jetstream is the source of truth" and "azure_blob is default for
real runs"; code defaults to SQLite + local backend and trusts operator
discipline.

---

## 5. Root cause

All ten findings reduce to a single architectural principle that is missing
from the codebase:

> **A database connection is opened without a declaration of intent. The
> system has no way of knowing whether a given write is supposed to be
> canonical (Jetstream + blob), a local mirror (clone for dev), or a sandbox
> (test/scratch). Because intent is not declared, no runtime check can fire
> when intent and reality disagree.**

`DATABASE_URL` was originally a 12-factor convenience: "the app talks to
whatever URL you give it." That's appropriate for an app where every database
lane is interchangeable. It is not appropriate for a system where one lane
(Jetstream) is "source of truth" and the others are subordinates.

The fix is therefore not "add more env vars" or "rename `DATABASE_URL`." The
fix is to **make intent a first-class argument at the connection chokepoint**,
and to **make the lane derivable from the URL** so that a mismatch between
declared intent and resolved lane can be detected and refused.

---

## 6. Implementation plan (folded-in Step 5)

This plan is structured to be incrementally landable. Each phase produces a
useful safety improvement on its own and unblocks the next phase. The
defense-in-depth ordering matters:

- **Phase 1 (chokepoint + banner)** prevents the 11-row class of defect at
  the **connection layer** for any code path that constructs a
  `BaseDatabaseConnection`. Had Phase 1 existed when the bank77_contrast run
  fired, the operator would have seen a "lane: LOCAL_POSTGRES, intent:
  CANONICAL" mismatch and the constructor would have refused.
- **Phase 2 (ArtifactService coupling)** catches the same class of defect at
  the **artifact-service layer** for any path that uses `ArtifactService`
  but somehow bypasses Phase 1 (e.g., a future caller that builds its own
  connection wrapper).
- **Phase 3 (Postgres `CHECK` constraint)** catches the same class of defect
  at the **database layer** for any path that bypasses both Phase 1 and
  Phase 2 — for example, any of the direct `create_engine(...)` sites
  enumerated in the §6.10 migration checklist (originally surfaced by
  subagent 1 §D), or any future raw-SQL inserter. This is why Phase 3 is in
  MVP (§6.9): until every site in §6.10 has been migrated to flow through
  the chokepoint, direct-engine writers remain a real bypass surface.

The three layers cover the defect at three independent altitudes; any one
of them in isolation closes the user's stated concern, but shipping all
three (Phases 1 + 2 + 3) leaves no single-point-of-failure escape.

### 6.0 Naming convention: role vs identity

Throughout §6 the plan deliberately separates two concepts that are currently
conflated in the codebase:

- **Role** (what library code reasons about): the runtime concept "this DB is
  the canonical source of truth." Expressed by the env var
  `CANONICAL_DATABASE_URL`, the enum value `Lane.CANONICAL`, and the
  `WriteIntent.CANONICAL` constant. Vendor-agnostic.
- **Identity** (what humans/ops know): the proper-noun name of the current
  incumbent that fills the canonical role. Today that is **Jetstream**.
  Identity surfaces in the env var `JETSTREAM_DATABASE_URL` (kept for
  back-compat), runbooks, the SSH-tunnel script, the `connect_jetstream` CLI,
  and ops-facing dump/restore tooling.

**Resolution rule:** library code reads `CANONICAL_DATABASE_URL` first; if
unset it falls back to `JETSTREAM_DATABASE_URL` (logging a deprecation notice
once per process). Ops scripts may continue to read either name during the
back-compat window. After one release cycle, the fallback is removed.

**Conflict rule (added per Codex audit):** if **both** `CANONICAL_DATABASE_URL`
and `JETSTREAM_DATABASE_URL` are set and they resolve to **different**
host/port/dbname triples (using the existing `same_db_target` helper from
`scripts/db_target_guardrails.py:40-70`), `resolve_lane(...)` raises
`CanonicalIdentityConflict` (new exception) at first call. Rationale: during
the alias window, two env vars naming the canonical lane is a footgun — an
operator who exports both and gets them wrong would have writes silently
routed to whichever the library happens to read first. The conflict rule
removes that ambiguity by failing closed instead of guessing. The exception
message names both URLs (with passwords masked) and tells the operator to
unset whichever one is wrong.

**Conflict-rule caveat (added per Codex audit pass 2): the comparison is
literal.** `same_db_target` parses each URL and compares the resulting
`(host, port, dbname)` triple as opaque strings; it does **not** resolve
aliases. In particular, the two URLs

- `postgresql://…@127.0.0.1:5434/study_query_jetstream?sslmode=prefer`
  (an SSH-tunneled URL pointing at a local-loopback port that forwards
  to canonical Jetstream — exactly what `scripts/start_jetstream_postgres_tunnel.py` produces and what §3 used to run the live count), **and**
- `postgresql://…@<jetstream-public-host>:5432/study_query_jetstream`
  (the direct public URL),

will compare as **different** even though they target the same physical
database. Likewise, two DNS aliases for the same host
(`db.internal` vs `db-primary.internal`) compare as different. The conflict
rule will **false-fail** in those cases. Operator guidance during the
back-compat window: set **only one** of `CANONICAL_DATABASE_URL` and
`JETSTREAM_DATABASE_URL` (preferably `CANONICAL_DATABASE_URL`) and let
the other remain unset; if both must be set transiently (e.g. during a
shell-history migration), make sure they are byte-for-byte identical.
A future enhancement could resolve loopback / hostname aliases via DNS
+ `pg_hba`/`pg_stat_activity` cross-check, but that is out of MVP scope
because the operator-discipline workaround is cheap and the misfire
mode (loud `CanonicalIdentityConflict` at startup) is fail-closed and
recoverable.

**Why split them:**
- The destructive-DDL guard at `_base_connection.py:147-172` is currently the
  *only* place the proper noun "Jetstream" leaks into library code. Splitting
  role from identity makes that block portable to any future SoT vendor with
  zero code changes.
- It also makes the eventual case "we run two regions, both
  canonical-eligible" expressible without renaming anything.
- It matches the `WriteIntent` symmetry: `WriteIntent.CANONICAL` ↔
  `Lane.CANONICAL` ↔ `CANONICAL_DATABASE_URL`. The mental model is one word.

For the remainder of §6, "the canonical lane" means `Lane.CANONICAL` (today:
Jetstream); "the local lane" means `Lane.LOCAL_POSTGRES` / `Lane.SQLITE_*`.
Where a citation references the existing `JETSTREAM_DATABASE_URL` env var or
"Jetstream" by name, that is intentional — it documents the current incumbent
identity, not the role.

### 6.1 Design — `WriteIntent` and `LaneResolver`

Two new modules under `src/study_query_llm/db/`:

- `lane.py`
  - `class Lane(StrEnum): CANONICAL, LOCAL_POSTGRES, SQLITE_FILE, SQLITE_MEMORY, UNKNOWN`
    - `CANONICAL` is the role-named value; whichever vendor URL fills the
      role today (Jetstream) maps to `Lane.CANONICAL`. See §6.0.
  - `def resolve_lane(connection_string: str) -> Lane`
    - Reads the role-named env var first:
      `os.environ.get("CANONICAL_DATABASE_URL") or
       os.environ.get("JETSTREAM_DATABASE_URL")`.
      Logs a deprecation notice once per process if it had to fall back to
      `JETSTREAM_DATABASE_URL`.
    - Compares URL host/port/dbname against the resolved canonical URL and
      against `LOCAL_DATABASE_URL` using the existing helpers in
      `scripts/db_target_guardrails.py:40-70` (`parse_postgres_target`,
      `is_loopback_target`, `same_db_target`) — promote those helpers into
      `src/study_query_llm/db/` so library code may use them.
    - Returns `CANONICAL` only if URL matches the resolved canonical URL.
    - Returns `LOCAL_POSTGRES` if URL is loopback Postgres or matches
      `LOCAL_DATABASE_URL`.
    - Returns `SQLITE_FILE` / `SQLITE_MEMORY` for sqlite URLs.
    - Returns `UNKNOWN` otherwise (and `WriteIntent` enforcement treats this
      as failure unless explicitly opted into).
- `write_intent.py`
  - `class WriteIntent(StrEnum): CANONICAL, READ_MIRROR, SANDBOX`
  - `_ALLOWED_LANES_BY_INTENT = { CANONICAL: {Lane.CANONICAL},
    READ_MIRROR: {Lane.LOCAL_POSTGRES, Lane.SQLITE_FILE},
    SANDBOX: {Lane.SQLITE_MEMORY, Lane.SQLITE_FILE, Lane.LOCAL_POSTGRES} }`
  - `def assert_intent_matches_lane(intent: WriteIntent, lane: Lane) -> None`
    - Raises `LaneIntentMismatch` (new exception) on mismatch with a verbose
      message naming both intent and resolved lane.

### 6.2 Wire the chokepoint into `BaseDatabaseConnection`

Modify the constructor signature (additive, keyword-only with default for
backwards compat during rollout):

```python
def __init__(
    self,
    connection_string: str,
    *,
    echo: bool = False,
    write_intent: WriteIntent | None = None,
    allow_unknown_lane: bool = False,
    quiet: bool = False,
    **engine_kwargs,
) -> None:
    self.lane = resolve_lane(connection_string)
    if write_intent is None:
        write_intent = _infer_intent_from_environment(self.lane)
    self.write_intent = write_intent
    if not allow_unknown_lane and self.lane is Lane.UNKNOWN:
        raise LaneResolutionError(...)
    assert_intent_matches_lane(self.write_intent, self.lane)
    if not quiet:
        _print_preflight_banner(self.lane, self.write_intent, connection_string)
    self.connection_string = connection_string
    self.engine = create_engine(connection_string, echo=echo, **engine_kwargs)
    ...
```

`_infer_intent_from_environment` uses a single env var
`SQLLM_WRITE_INTENT` (values: `canonical`, `read_mirror`, `sandbox`); when
unset, it picks `SANDBOX` for sqlite URLs and **raises** for Postgres URLs (no
silent default for live databases).

The `quiet` flag is only for tests that pre-print their own banner.

### 6.3 Preflight banner

Print to stderr at construction time (not info-level log — the user must see
this when running interactively):

```
================================================================
  STUDY-QUERY-LLM — DB SESSION
  lane:           CANONICAL        (identity: Jetstream)
  source_var:     CANONICAL_DATABASE_URL
  intent:         CANONICAL
  artifact_back:  azure_blob       (container=artifacts-prod)
  destructive:    BLOCKED unless SQLLM_ALLOW_DESTRUCTIVE_DDL=1
================================================================
```

The `lane:` row prints the **role** name (`Lane.CANONICAL`); the
parenthesised identity (`Jetstream`) and the explicit `source_var:` row
together name the current incumbent and the env var that resolved to it
(per §6.0). When the lane is `Lane.LOCAL_POSTGRES` or `Lane.SQLITE_*` the
identity reads `local clone` / `sqlite file` / `sqlite memory` and the
banner adds an extra line:

```
  WARNING:        this work will NOT propagate to the canonical DB
```

That single sentence addresses the user's core concern: it is impossible to
do canonical-intent work without seeing it labelled CANONICAL, and equally
impossible to do local work and not see the warning.

### 6.4 Couple `ArtifactService` to the chokepoint

Add a constructor argument `write_intent` to `ArtifactService` and to
`_resolve_default_backend`. Behaviour:

- If `write_intent is WriteIntent.CANONICAL`, the resolved backend **must** be
  `azure_blob`. Otherwise raise `ArtifactBackendIntentMismatch` (new
  exception). Local fallback for azure SDK errors becomes an immediate hard
  failure when intent is CANONICAL, regardless of `ARTIFACT_STORAGE_STRICT_MODE`.
- For `READ_MIRROR` / `SANDBOX`, current behavior is preserved.

The pipeline `runner.py:run_stage` already constructs `ArtifactService` with
`repository=artifact_repo, artifact_dir=artifact_dir` (subagent 3 §C row
"`pipeline/runner.py:150-153`"). Update that single site to pass
`write_intent=db.write_intent`. Every pipeline stage flows through
`run_stage`, so this single edit propagates to acquire / parse / snapshot /
embed / analyze in one step.

For services constructed outside `run_stage`
(`experiments/ingestion.py:161-162`, `services/embeddings/helpers.py:43`,
etc., subagent 3 §C), threaded edits will be needed; these are listed in
§6.10.

### 6.5 Database-level constraint on `call_artifacts.uri`

Add a Postgres `CHECK` constraint that fires only on the canonical lane
(currently Jetstream). The local clone may legitimately hold local paths if
it was hydrated by `sync_from_online.py` for a not-yet-promoted set of
fields:

```sql
ALTER TABLE call_artifacts
  ADD CONSTRAINT call_artifacts_uri_must_be_blob
  CHECK (
    uri ~ '^https://[A-Za-z0-9-]+\.blob\.core\.windows\.net/'
  )
  NOT VALID;
```

`NOT VALID` lets us add the constraint on a polluted table without immediately
failing existing rows; new inserts will still be checked. Once §7 remediation
is complete, run `ALTER TABLE call_artifacts VALIDATE CONSTRAINT
call_artifacts_uri_must_be_blob;` to lock it in.

This constraint is **only** added to the canonical lane, by gating the
migration with `Lane.CANONICAL`. The local-clone schema may keep the
constraint absent so mirror operations remain unchanged.

A parallel `CHECK` on `raw_calls.response_json` is more invasive (needs JSON
extraction); the recommended pattern is a partial index:

```sql
CREATE INDEX raw_calls_local_uri_sentinel
  ON raw_calls ((response_json->>'uri'))
  WHERE response_json ? 'uri'
    AND response_json->>'uri' NOT LIKE 'https://%';
```

This is not a constraint — it's a sentinel index. A nightly job (or a CI
check) querying that index will instantly surface any new local-path leaks
without blocking writes mid-migration.

### 6.6 CI enforcement

Two new CI jobs:

1. **Static check — connection-construction policy.** Lint that every
   construction of `DatabaseConnectionV2` / `BaseDatabaseConnection` and
   every direct `create_engine(...)` call in the **runtime tree** (per
   §6.10's "Scope of this checklist" — `src/`, `panel_app/`, `scripts/`
   (excluding `scripts/history/`), `experiments/`, `deploy/`) passes
   `write_intent=`. Implemented as a ruff/flake8 plugin, or as a targeted
   regex-based pytest under `tests/static/test_db_connection_intent.py`.
   The `_base_connection.py:107` `create_engine` call is the chokepoint
   itself and is the **only** allow-listed direct `create_engine` site
   inside the runtime tree (per §6.10 §G); all other allow-listed paths
   live outside the runtime tree (`tests/**` covered by §6.7 fixture,
   `docs/audit/**` audit probes, `scripts/history/**` frozen archives,
   `notebooks/**` covered separately, untracked scratch). Subagent 1 §D's
   enumeration of bypass sites (`scripts/purge_dataset_acquisition.py:161-162`,
   `scripts/check_active_workers.py:83-84`, etc.) is the authoritative list of
   call sites that must be migrated.
2. **Live check — canonical-lane URI shape.** Promote `live_count.py` (this
   folder) into `scripts/verify_no_local_uris_on_canonical.py` (back-compat
   alias `verify_no_local_uris_in_jetstream.py` for one release), run it as
   a scheduled CI step against the canonical lane (currently Jetstream), and
   fail the run if any `class=local_path` rows appear in `call_artifacts.uri`.

### 6.7 Test isolation hardening

Subagent 5 §E recommendations:

- `tests/conftest.py` should not use `setdefault`. Replace with an
  unconditional override that sets `DATABASE_URL=sqlite:///:memory:` **unless**
  the test explicitly opts into a real DB via marker
  (`@pytest.mark.requires_real_db`).
- Add a session-scoped fixture that constructs all `DatabaseConnectionV2`
  instances with `write_intent=WriteIntent.SANDBOX`.
- Add a hard assertion in `tests/conftest.py` that `resolve_lane(...)` for
  the resolved URL is **not** `Lane.CANONICAL`. This is the test-only
  equivalent of the constructor's `assert_intent_matches_lane`.

### 6.8 Documentation updates

Per subagent 6 §C:
- `docs/runbooks/README.md` — add an "Operational Lane" section that
  references `WriteIntent` and the preflight banner; include a screenshot of
  the banner.
- `docs/design/database_safety_guardrails.md` — extend to cover
  artifact-backend ↔ DB-target consistency.
- `docs/design/clustering_pipeline_provenance.md` — explicitly assert that
  canonical `uri` columns must be HTTPS blob URLs.
- `.env.example` — promote `CANONICAL_DATABASE_URL`, `LOCAL_DATABASE_URL`,
  `SQLLM_WRITE_INTENT` to required entries with comments. Document
  `JETSTREAM_DATABASE_URL` as the deprecated identity-named alias retained
  for one release of back-compat (per §6.0).
- `AGENTS.md` — codify the rule "no agent action may write canonical data
  unless the preflight banner is observed."

### 6.9 Phasing & rollback

> **Minimum-viable shipment (MVP) — scoped per Codex audit pass 2.**
> Phase 0 + Phase 1 (a/b/c) + Phase 3 + §7 remediation = the only items
> strictly required to close the **observed** pollution defect (the 11
> stranded `call_artifacts` rows in §3 and any future writes through the
> same code path). The MVP claim is deliberately **scoped to the
> `call_artifacts.uri` surface and its `raw_calls.response_json[uri]`
> mirror**; broader URI-column coverage is post-MVP.
>
> What MVP closes, by surface:
>
> - **`call_artifacts.uri` (primary surface — all 11 polluted rows live
>   here):** Phase 1 gates the connection layer (chokepoint refuses
>   canonical-lane sessions without `WriteIntent.CANONICAL`) and Phase 3
>   adds a Postgres `CHECK` at the database layer. Defect class closed at
>   two altitudes.
> - **`raw_calls.response_json[uri]` (mirror of `call_artifacts`):**
>   covered **transitively** for writes that flow through `ArtifactService`.
>   The placeholder `raw_call` insert at `services/artifact_service.py:432-440`
>   and the paired `call_artifacts` insert at `:452-464` share a single
>   `repository.session`; `RawCallRepository.insert_raw_call` only
>   `flush()`es, never commits (`db/raw_call_repository.py:96-118`). A
>   Phase 3 `CHECK` failure on the call_artifacts insert therefore rolls
>   back the paired raw_calls placeholder in the same transaction. Coverage
>   gap: a hypothetical future direct-engine writer that wrote
>   `raw_calls.response_json[uri]` **without** a paired `call_artifacts`
>   row would not be caught by Phase 3. Two direct `RawCall` writers
>   outside `RawCallRepository` already exist today —
>   `scripts/sync_from_online.py:179`
>   (`pg_insert(RawCall).values(...).on_conflict_do_nothing` on a raw
>   `create_engine(local_url)` session at `:295`) and
>   `scripts/archive_defective_data.py:98-112`
>   (`local_session.add(RawCall(...))` on `DatabaseConnectionV2(local_url)`
>   at `:190`) — but both default to `LOCAL_DATABASE_URL`
>   (`sync_from_online.py:255`, `archive_defective_data.py:170`),
>   `sync_from_online.py` additionally blocks non-loopback targets unless
>   `--allow-remote-target` is passed (`:276`), and **each co-inserts the
>   paired `CallArtifact` row in the same session as the `RawCall`**
>   (`sync_from_online.py:206-218`, `archive_defective_data.py:145-155, 217`),
>   so the Phase 3 `CHECK` on `call_artifacts.uri` would reject the
>   COMMIT and roll back the paired `raw_calls` mirror if either script
>   were ever misconfigured against canonical.
>   `archive_defective_data.py` is additionally gated by the Phase 1
>   chokepoint (it uses `DatabaseConnectionV2`);
>   `sync_from_online.py` bypasses Phase 1 (raw `create_engine`) but is
>   still caught at Phase 3 plus its loopback guard. Phase 5's sentinel
>   index covers the residual case of a future writer that emits a
>   `raw_calls` row **without** a paired `call_artifacts` row.
> - **Other artifact-bearing JSON columns**
>   (`groups.metadata_json[artifact_uri]`, `analysis_results.result_json[uris]`,
>   `provenanced_runs.result_ref`, `orchestration_jobs.result_ref`): MVP
>   adds **no DB-layer enforcement** on these columns. Coverage in MVP is
>   Phase 1 only — the chokepoint blocks any session whose declared
>   `WriteIntent` does not match the resolved lane. §3.3 confirms these
>   columns currently hold **zero** URIs in the canonical DB, so the
>   residual risk is narrow: a future first-write that simultaneously
>   (a) bypasses the chokepoint (any of the direct `create_engine(...)`
>   sites enumerated in §6.10, or a brand-new such site) **and** (b) targets
>   one of these JSON paths. Phase 5 (sentinel index) catches the
>   `raw_calls` analogue post-hoc; extending similar sentinels (or JSON
>   `CHECK`/trigger constraints) to the other four columns is tracked as
>   a follow-on RFC, not MVP.
>
> Phases 2 and 4–7 deepen the protections (artifact-backend coupling at
> the service layer, `VALIDATE CONSTRAINT`, sentinel indexes, CI, test
> hardening, docs) and can be calendar-sequenced over weeks without
> losing the primary MVP safety property.
>
> If you only ship one bundle, ship MVP = **0 + 1 + 3 + §7** in that order
> and document in the rollout note that (i) JSON-column protections beyond
> `call_artifacts.uri` and its same-transaction `raw_calls` mirror are
> Phase 5+ work, and (ii) the §6.10 chokepoint migration is the
> long-pole on collapsing the direct-engine bypass surface that Phase 3
> currently insures against.

| Phase | What ships | Risk |
|------|-----------|------|
| 0 | Land `lane.py` and `write_intent.py` modules, **not yet wired**. | None. |
| 1a | Add `write_intent` kwarg to `BaseDatabaseConnection.__init__` with default `None` → falls back to environment-inferred behavior. Print preflight banner unconditionally. | Low — only adds output. |
| 1b | Gate `assert_intent_matches_lane` on `SQLLM_WRITE_INTENT` being set. Operators opt in. | Low — no breakage. |
| 1c | Make `SQLLM_WRITE_INTENT` required for non-sqlite URLs. | Medium — operators must update env. Ship after 2 weeks of opt-in soak. |
| 2 | Wire `ArtifactService(write_intent=...)`. CANONICAL forces `azure_blob`. | Medium — could break a worker with misconfigured azure creds. Mitigation: phase 1 banners surface this earlier. |
| 3 | Add Postgres `CHECK ... NOT VALID` on `call_artifacts.uri` on the canonical lane (currently Jetstream). **In MVP** so direct-engine bypass writers (subagent 1 §D) are caught at the DB layer until they migrate. | Low — new inserts only; existing rows untouched until phase 4. |
| 4 | Run §7 remediation; then `VALIDATE CONSTRAINT`. | Medium — coordinate with anyone holding open transactions. |
| 5 | Promote sentinel index for `raw_calls`. | Low. |
| 6 | CI jobs (§6.6) + test fixture changes (§6.7). | Low. |
| 7 | Docs (§6.8). | None. |

Rollback for any phase: revert the migration (constraints are individually
named) or set the new kwargs back to permissive defaults; the underlying
schema is unchanged. For Phase 3 specifically, rollback is
`ALTER TABLE call_artifacts DROP CONSTRAINT call_artifacts_uri_must_be_blob;`
which is a metadata-only operation.

### 6.10 Migration checklist for in-tree call sites

#### Scope of this checklist

Every **runtime-surface** in-tree `DatabaseConnectionV2(...)` constructor
and every direct `create_engine(...)` site listed below must declare a
`WriteIntent` after the chokepoint lands (§6.2). The list is **complete
for runtime surfaces as of commit 8e95253** — but only for the runtime
surfaces. To re-derive the current set at implementation time, scope the
search to the runtime tree explicitly:

```bash
rg "DatabaseConnectionV2\(" \
  -g 'src/**/*.py' \
  -g 'panel_app/**/*.py' \
  -g 'scripts/**/*.py' \
  -g 'experiments/**/*.py' \
  -g 'deploy/**/*.py' \
  -g '!scripts/history/**' \
  -g '!**/tests/**' \
  -g '!docs/audit/**'

rg "create_engine\(" \
  -g 'src/**/*.py' \
  -g 'panel_app/**/*.py' \
  -g 'scripts/**/*.py' \
  -g 'experiments/**/*.py' \
  -g 'deploy/**/*.py' \
  -g '!scripts/history/**' \
  -g '!**/tests/**' \
  -g '!docs/audit/**'
```

**Operator note on false positives.** The `rg` patterns above match
`DatabaseConnectionV2(` and `create_engine(` as raw substrings, so they
will also surface docstring/comment usage examples that are not real
call sites. The canonical example is `src/study_query_llm/db/connection_v2.py:19`,
which shows `db = DatabaseConnectionV2("postgresql://user:pass@localhost/dbname")`
inside the class docstring. When triaging the output, eyeball each
match for the surrounding `"""` / `#` context and skip docstring,
comment, and type-stub examples; only true call-site invocations need
to flow through the chokepoint and declare a `WriteIntent`.

The original draft of this section claimed exhaustiveness and was based on
subagent 2 §A; the Codex audit found additional sites that subagent 2 missed
and one cited script (`scripts/restore_jetstream_db.py`) that does not exist.
Both defects are corrected below — see §12 for the full provenance note.

**Explicitly out-of-scope** (these may use `DatabaseConnectionV2` /
`create_engine` but must **not** be migrated by the chokepoint sweep, and
the §6.6 CI lint must allow-list them):

- `docs/audit/**` — read-only audit-doc probes. `docs/audit/db_target_lane_audit_2026-04-24/live_count.py:410` is the canonical example: it
  intentionally uses raw `create_engine(...)` because the whole point of
  the script is to bypass the chokepoint and probe the live DB without
  any application-side machinery loaded. Migrating these to the
  chokepoint would defeat their purpose.
- `tests/**` — covered separately by §6.7 (a pytest fixture forces
  `WriteIntent.SANDBOX` and a hard assertion that the resolved lane is
  not `Lane.CANONICAL`).
- `scripts/history/**/*.py` — frozen archives per `.cursorrules`
  "living-docs-only" rule. If any of them are revived for active use
  they must be migrated **before** re-running; running them as-is from
  history is forbidden.
- `notebooks/**/*.ipynb` — construct connections at cell-evaluation time
  and cannot be statically gated by import-time CI. The recommendation
  for notebooks is to expose three thin helpers (`get_canonical_db()`,
  `get_local_db()`, `get_sandbox_db()`) that internally set intent —
  tracked separately from MVP.
- Local untracked scratch (e.g. `scratch/`, `local/`, anything outside
  the runtime tree above) — out of scope by definition.

#### A. Canonical writers (declare `WriteIntent.CANONICAL`)

These sites either write to or perform DDL on the canonical lane.

- `panel_app/helpers.py:38` — Panel UI intends to read/write the canonical
  DB.
- `src/study_query_llm/services/jobs/runtime_supervisors.py:41,367` and
  `src/study_query_llm/services/jobs/runtime_workers.py:68` — supervisors
  and workers consume `DATABASE_URL` directly; declare from
  `SQLLM_WRITE_INTENT`.
- `src/study_query_llm/experiments/sweep_worker_main.py:612,1331` — same
  pattern.
- `src/study_query_llm/experiments/runtime_sweeps.py:305` — sweep runtime
  driver.
- `src/study_query_llm/pipeline/{acquire,parse,embed,snapshot,analyze}.py`
  (`acquire.py:38`, `parse.py:41`, `embed.py:40`, `snapshot.py:35`,
  `analyze.py:88`) — pipeline stage entry points; accept `write_intent` arg,
  default `CANONICAL`.
- `src/study_query_llm/analysis/mcq_analyze_request.py:315` — analysis
  driver writes provenance + analysis results.
- `scripts/ingest_sweep_to_db.py:487` — pass `CANONICAL`; remove fallback to
  `config.database.connection_string` in favour of explicit
  `CANONICAL_DATABASE_URL` (back-compat per §6.0).
- `scripts/ingest_mcq_probe_json_to_sweep_db.py:145` — pass `CANONICAL`;
  resolve from `CANONICAL_DATABASE_URL` first, fall back to `DATABASE_URL`
  then `NEON_DATABASE_URL` (per subagent 1 §B).
- `scripts/register_clustering_methods.py:71`,
  `scripts/register_text_classification_methods.py:72` — register methods on
  the canonical lane.
- `scripts/run_bank77_pipeline.py:336`,
  `scripts/run_pca_kllmeans_sweep.py:108`,
  `scripts/run_pca_kllmeans_sweep_full.py:291` — pipeline drivers; declare
  `CANONICAL`.
- `scripts/backfill_run_fingerprints.py:43`,
  `scripts/validate_and_backfill_run_snapshots.py:111` — backfill writers.
- `scripts/purge_dataset_acquisition.py:161` (`create_engine`) — destructive
  on canonical after explicit confirmation; the existing
  `--allow-remote-target` / loopback gate stays, plus declare `CANONICAL`.
- `scripts/backup_mcq_db_to_json.py:255` (`create_engine`) — reads canonical
  for backup; effectively read-only against canonical (see §10 Q8 — current
  intent vocabulary cannot express this cleanly).

#### B. Read-mirror / diagnostic (declare `WriteIntent.READ_MIRROR`)

These sites only operate on a local mirror.

- `scripts/sync_from_online.py:295` — local engine target. (Note: the
  online engine constructed at `:290` is read-only against canonical — see
  §10 Q8.)
- `scripts/init_local_db.py:51` — initialises the local mirror only.
- `scripts/check_active_workers.py:83` (`create_engine`),
  `scripts/probe_postgres_inventory.py:61` (`create_engine`),
  `scripts/sanity_check_database_url.py:60` (`create_engine`) — diagnostics
  intended for either lane; default `READ_MIRROR` and require explicit
  override to point at canonical.
- `scripts/audit_mcq_method_definitions.py:138`,
  `scripts/audit_last_partial_sweep.py:166`,
  `scripts/check_run_groups.py:28`,
  `scripts/check_orchestration_jobs.py:45`,
  `scripts/check_sweep_requests.py:79`,
  `scripts/reconcile_last_partial_sweep.py:106`,
  `scripts/export_mcq_sweep_option_counts_db.py:464` — read-only audit /
  reconciliation scripts.

#### C. Sandbox / ephemeral (declare `WriteIntent.SANDBOX`)

- `scripts/snapshot_inventory.py:313` — uses ephemeral `sqlite:///{tmpdir}`
  by construction.
- `scripts/create_bank77_contrast_snapshots.py:330` — has implicit local
  SQLite fallback under `artifact_dir`; this fallback should be removed and
  the script forced to declare intent explicitly.

#### D. Dual-target (declare per construction)

Scripts that construct multiple connections; each connection declares its
own intent.

- `scripts/archive_defective_data.py:189-190` —
  `online_db = DatabaseConnectionV2(online_url, write_intent=CANONICAL)` +
  `local_db = DatabaseConnectionV2(local_url, write_intent=READ_MIRROR)`.
- `scripts/verify_db_backup_inventory.py:55` (`create_engine`) — iterates
  over a list that may contain both canonical and local URLs; each engine
  declares intent at construction (see §10 Q8 for the canonical-read case).

#### E. Promotion / backup tooling (effectively read-only against canonical)

These are operator scripts whose write target is **outside** the database
(blob, dump file). They read canonical and produce an artifact for offline
review or restore. Declared intent depends on §10 Q8 resolution; the
recommended follow-on (per pass-2 audit) is the new
`WriteIntent.CANONICAL_READ_ONLY` enum value (option (a) in §10 Q8).
Until that lands they should declare `WriteIntent.CANONICAL` with a code
comment marking them as read-only canonical readers, and operator
discipline must enforce the no-write contract.

- `scripts/upload_jetstream_pg_dump_to_blob.py:50` (`create_engine`) — reads
  canonical for table-count manifest; writes to Azure blob.
- `scripts/verify_call_artifact_blob_lanes.py:130` (`create_engine`) —
  direct-engine bypass already flagged in subagent 1 §D; reads canonical to
  verify URI shape. The `verify_no_local_uris_on_canonical.py` promotion
  in §6.6 should subsume it.

#### F. Migrations (`src/study_query_llm/db/migrations/*.py`)

All nine migration scripts consume `DATABASE_URL` and perform DDL on the
canonical lane:

- `add_provenanced_runs_table.py:34`
- `add_method_analysis_tables.py:38`
- `add_recipe_json_column.py:91`
- `add_sweep_request_indexes.py:42`
- `add_sweep_worker_safety.py:89`
- `add_fingerprint_columns.py:57`
- `add_group_links.py:37`
- `normalize_provenanced_run_kind_execution.py:38`
- `drop_embedding_vectors.py:52` (destructive)

Pattern: declare `WriteIntent` from `SQLLM_WRITE_INTENT` and require
operators to set it explicitly. The destructive ones
(`drop_embedding_vectors.py`, `normalize_provenanced_run_kind_execution.py`)
should additionally require `SQLLM_ALLOW_DESTRUCTIVE_DDL=1` (already
enforced by the existing guard at `_base_connection.py:147-172`).

#### G. The chokepoint itself

- `src/study_query_llm/db/_base_connection.py:107` — `create_engine` call
  inside `BaseDatabaseConnection.__init__`. **This is the only direct
  `create_engine` site that is allowed in library code after MVP**; the CI
  lint job in §6.6 enforces this by allow-listing exactly this file:line.

---

## 7. Remediation of the 11 polluted rows

The §3 inventory tells us exactly which rows are stranded:

| `call_artifacts.id` | `artifact_type` | local path |
|--------------------|-----------------|-----------|
| 21 | dataset_acquisition_file | `…\bank77_contrast\15\acquisition\data_train-00000-of-00001.parquet` |
| 22 | dataset_acquisition_file | `…\bank77_contrast\15\acquisition\data_test-00000-of-00001.parquet` |
| 23 | dataset_acquisition_manifest | `…\bank77_contrast\15\acquisition\acquisition.json` |
| 24 | dataset_canonical_parquet | `…\bank77_contrast\16\parse\dataframe.parquet` |
| 25 | dataset_dataframe_manifest | `…\bank77_contrast\16\parse\dataframe_manifest.json` |
| 26 | dataset_subquery_spec | `…\bank77_contrast\17\snapshot\subquery_spec.json` |
| 27 | dataset_subquery_spec | `…\bank77_contrast\18\snapshot\subquery_spec.json` |
| 28 | dataset_subquery_spec | `…\bank77_contrast\19\snapshot\subquery_spec.json` |
| 29 | dataset_subquery_spec | `…\bank77_contrast\20\snapshot\subquery_spec.json` |
| 30 | dataset_subquery_spec | `…\bank77_contrast\21\snapshot\subquery_spec.json` |
| 31 | dataset_subquery_spec | `…\bank77_contrast\22\snapshot\subquery_spec.json` |

(`live_count_output.txt:24-34`)

Recommended remediation script (new
`scripts/remediate_local_path_call_artifacts.py`, dry-run by default):

1. **Resolve Jetstream connection** with `WriteIntent.CANONICAL`, fail loudly
   if `SQLLM_WRITE_INTENT` not set to `canonical`.
2. **Resolve azure backend** via `ArtifactService(..., write_intent=
   CANONICAL)` (forces `azure_blob`). Confirm container target matches
   `runtime_env=prod` rules from `_resolve_blob_container`
   (`artifact_service.py:160-182`).
3. **For each row in the table above:**
   1. Verify the local file still exists on disk. If missing, mark the row for
      `manual_intervention` and stop processing it.
   2. Compute `sha256` and `byte_size` (matches
      `_integrity_metadata`, `artifact_service.py:154-158`).
   3. Compute the canonical logical path the artifact should have used
      (derive from `call_artifacts.artifact_type` + the existing
      `groups`/`group_links` association — same logic that `run_stage` uses,
      but applied retroactively).
   4. Upload the file to azure under that logical path with
      `storage.write(...)`.
   5. Capture the returned blob URI.
   6. **In a single transaction:** update `call_artifacts.uri` and the
      mirrored `raw_calls.response_json[uri]` to the new blob URI. Log the
      previous (local) value into a new `audit_log` JSON column (or a
      side-table `call_artifacts_remediation_log` if column add is too
      invasive).
3. **After all rows succeed:** re-run `live_count.py`, expect zero
   `local_path` rows.
4. **Then:** `ALTER TABLE call_artifacts VALIDATE CONSTRAINT
   call_artifacts_uri_must_be_blob;` to flip the constraint from `NOT VALID`
   to enforced.

**Failure handling.** If a local file is missing, the artifact is genuinely
lost. Two options, listed in order of preference:
1. Re-run the originating pipeline stage with the proper
   `write_intent=CANONICAL` + `ARTIFACT_STORAGE_BACKEND=azure_blob`. Because
   the `groups`/`group_links` rows already exist (they were Jetstream-bound
   from the start), the new run can use the existing `request_group_id` and
   produce a fresh blob.
2. If the originating inputs are themselves no longer reproducible, **delete**
   the `call_artifacts` row and mark the parent `groups` row as `defective`
   via `data_quality_service.get_or_create_defective_group`
   (`raw_call_repository.py` patterns referenced in subagent 2 §A). This is
   the "correct" outcome — the row was always pointing at unreachable data;
   removing it makes the loss visible.

The remediation script should print a final report:
- N rows successfully replaced.
- N rows requiring re-run (file missing).
- N rows quarantined as defective.

That report should be committed to this audit folder under
`remediation_report.md` so the audit trail is closed.

---

## 8. Why this design solves the user's stated concern

The user's exact phrasing was: *"anything that leaves ambiguous whether we are
working with our Jetstream database or a local database is unacceptable."*

The proposed design eliminates ambiguity at every layer:

| Layer | Today | After full plan | In MVP? |
|-------|-------|-----------------|---------|
| Connection construction | accepts any URL silently | refuses to construct without explicit `WriteIntent`; mismatch raises | **Yes (Phase 1)** |
| Banner | none | mandatory stderr banner naming lane + intent + artifact backend | **Yes (Phase 1)** |
| Artifact backend | independent of DB target | CANONICAL forces `azure_blob`; SDK errors fail closed | No (Phase 2) |
| Schema — `call_artifacts.uri` | no constraint | canonical-lane `CHECK` rejects non-blob URIs | **Yes (Phase 3)** |
| Schema — `raw_calls.response_json[uri]` | no constraint | covered transitively via shared txn for ArtifactService writes; sentinel index for direct-engine writers | **Partial: txn-rollback in MVP; sentinel in Phase 5** |
| Schema — other JSON URI columns (`groups.metadata_json`, `analysis_results.result_json`, `provenanced_runs.result_ref`, `orchestration_jobs.result_ref`) | no constraint | follow-on JSON `CHECK` / sentinel indexes per RFC | No (post-MVP) |
| CI | no live check | scheduled `verify_no_local_uris_on_canonical` job fails on any local path | No (Phase 6) |
| Tests | `setdefault` can be overridden by shell env | hard override + assertion that lane is non-`Lane.CANONICAL` | No (Phase 6) |
| Docs | aspirational | reference the actual `WriteIntent` mechanism | No (Phase 7) |

There is **no expectation of operator discipline** anywhere in the chain. An
operator who tries to do the wrong thing gets, under the **full plan**:
1. A loud banner saying what lane they're on, **before** any write.
2. A type error on `WriteIntent` if they forgot to declare intent.
3. A backend error if their declared CANONICAL intent doesn't get azure.
4. A database constraint error if the wrong URI shape gets through
   (`call_artifacts.uri` always; other URI columns once the follow-on RFC
   ships).
5. A CI failure if any of the above are bypassed.

That is the "no ambiguity" property the user asked for, encoded in five
independently-failing safety layers.

**MVP caveat (per §6.9).** The MVP shipment provides layers 1, 2, and 4
for the `call_artifacts.uri` surface (with the `raw_calls.response_json[uri]`
mirror covered transitively for `ArtifactService` writes via shared
transaction). Layers 3 (artifact-backend coupling) and 5 (CI) plus the
JSON-column extension of layer 4 are post-MVP work. Operationally, this
means MVP closes the **observed defect** (the 11 stranded rows in §3 and
any future writes through the same path) but leaves a narrow window in
which a future direct-engine writer that targets a non-`call_artifacts`
URI column could still pollute the canonical DB. §6.9 details why this
scoping is acceptable given §3.3 (those columns are currently empty).

---

## 9. Why no "flush" mechanic is being proposed

The user explicitly noted in their original prompt: *"NOTE: I am assuming this
is not some sort of 'flush' mechanic where there are prepped changes which are
then sent in batches to the actual database."*

This audit confirms that assumption (subagent 4 §A, §D — there is no such
mechanic), and **the implementation plan deliberately does not add one**.
Justification:

1. **The cost of a flush mechanic is high.** It would require a write-ahead
   log on every local write, conflict resolution on promotion, and a strong
   ordering guarantee against concurrent Jetstream writes. The repo does not
   currently need this and adding it would increase complexity for limited
   benefit.
2. **The need for a flush mechanic disappears once intent is declared.** The
   only reason a flush would be needed is to recover from accidental local
   writes — and the §6 plan **prevents** accidental local writes in the first
   place. If a developer wants to write to the local clone for development
   work, they declare `WriteIntent.READ_MIRROR` or `SANDBOX`; the system never
   confuses that with canonical work, and there is nothing to flush.
3. **The legitimate use cases for "local first, promote later" are already
   served** by the manual two-step:
   `pg_dump` (or `scripts/dump_postgres_for_jetstream_migration.py`)
   → human review of the dump → transfer the `.dump` file to the Jetstream
   VM → restore on canonical via **either** of two operator-facing
   scripts:
   - `deploy/jetstream/jetstream_pgvector_restore.sh` (the high-level
     wrapper — wipes the Postgres volume, starts the pgvector-enabled db,
     **invokes `deploy/jetstream/restore_pg_dump_to_compose_db.sh`** at
     line 80 to load the dump, verifies the `vector` extension, then
     starts the app); **or**
   - `deploy/jetstream/restore_pg_dump_to_compose_db.sh` directly (the
     low-level restore step that the wrapper invokes — used when the
     compose stack is already up and only the data needs to be replaced).
   That is a coarse-grained, dump-replace promotion path, not a
   fine-grained merge, but it matches the actual workflow described in
   `docs/runbooks/README.md`. (The earlier draft of this section cited
   `dump_jetstream_db.py` and `restore_jetstream_db.py`; both names are
   wrong — see §12. The earlier draft of this section also had the
   wrapper/wrappee call direction inverted — Codex audit pass 2 caught it
   and the corrected direction is shown above.)

If, after Phase 6, operators discover a real workflow that requires
incremental promotion, that becomes a follow-on RFC. It is not blocking this
audit.

---

## 10. Open questions and risks

| ID | Question | Recommended resolution |
|----|----------|------------------------|
| Q1 | Should `recreate_db` migrations on a Jetstream-pointed dev clone be allowed? | No. The destructive guard's existing hard-stop on Jetstream URL match is correct; the plan extends it (§6.5) but does not weaken it. |
| Q2 | What about pipeline runs that legitimately use `LOCAL_POSTGRES` for QA? | They declare `WriteIntent.READ_MIRROR`. The banner says "this work will NOT propagate to Jetstream" — the operator sees this every run. |
| Q3 | The `audit_log` column for remediation — schema migration or side-table? | Side-table (`call_artifacts_remediation_log`). Avoids touching the canonical schema for one-time use. |
| Q4 | The `NOT VALID` constraint phase: any race where a slow pipeline writes a local URI between the constraint add and the remediation script? | Schedule remediation within 24h of constraint add; revoke `INSERT` from non-Jetstream-credentialed roles during remediation. |
| Q5 | Is the bank77_contrast snapshot lineage retrievable end-to-end? | Confirmed via `live_count_output.txt:24-34` — the `groups` rows referencing artifact ids 21–31 exist in Jetstream (subagent 2 §A confirms `groups` writes are routine). The remediation script can rebuild blob artifacts deterministically because `groups`/`group_links` are intact. |
| Q6 | Do the same 11 artifacts exist in the local-clone DB? | Likely yes (the local clone was hydrated from Jetstream, which contained them). The local clone is unaffected by the §6.5 constraint because it is added only when `Lane.CANONICAL`. |
| Q7 | Could the same defect have produced legacy `inference_runs` rows in v1 schema? | No. v1's `InferenceRun` has no artifact URI columns (subagent 3 §D, "Legacy `models.py`"). |
| Q8 | The current `WriteIntent` vocabulary (`CANONICAL` / `READ_MIRROR` / `SANDBOX`) cannot cleanly express "this script reads the canonical lane to produce an external artifact (blob upload, dump file, audit report) and never writes to it." Examples: `scripts/upload_jetstream_pg_dump_to_blob.py:50`, `scripts/backup_mcq_db_to_json.py:255`, the online engine in `scripts/sync_from_online.py:290`, the canonical engine in `scripts/verify_db_backup_inventory.py:55`, and the §6.6 verifier scripts themselves. Today, declaring `READ_MIRROR` against a canonical URL would (correctly) raise `LaneIntentMismatch` per §6.1. | Out of MVP. Two options for a follow-on: **(a)** add a fourth intent `CANONICAL_READ_ONLY` whose `_ALLOWED_LANES_BY_INTENT` is `{Lane.CANONICAL}` and which sets `engine = create_engine(url, isolation_level="AUTOCOMMIT", execution_options={"postgresql_readonly": True})`, or **(b)** keep three intents and add a `read_only: bool = False` kwarg to `BaseDatabaseConnection.__init__` that flips the same Postgres session flags and that the chokepoint requires when `intent=CANONICAL` and the caller is in the §6.6 allow-list of "canonical readers." **Recommendation (revised per Codex audit pass 2): (a)** — `CANONICAL_READ_ONLY` is a single first-class enum value that the §6.6 CI lint can match exactly (`grep WriteIntent.CANONICAL_READ_ONLY` is a one-line policy check). Option (b) creates a 3×2 intent×read-only matrix in which only certain combinations are valid (`READ_MIRROR + read_only=True` is meaningless, `SANDBOX + read_only=True` is meaningless, `CANONICAL + read_only=False` is the existing canonical writer), and the chokepoint would have to enforce a more complex rule of the form "if intent=CANONICAL and caller is in `CANONICAL_READERS_ALLOWLIST`, then read_only must be True" — strictly more surface area than option (a). Option (a) also keeps `WriteIntent` as the single concept that callers reason about, mirroring the role/identity discipline of §6.0. Either way, the MVP plan is unaffected: until Q8 lands, the affected scripts in §6.10 §B and §6.10 §E should stay declared as `CANONICAL` with operator discipline (no writes), exactly as they behave today. |

---

## 11. Cross-walk: every claim ↔ source

For Codex's audit convenience. Every claim in this document is sourced from
either a code file (cited inline) or a subagent file. The mapping below
ensures no claim is unsupported.

| Claim ID | Section | Cited in |
|----------|---------|----------|
| C1 — DB target ambiguity exists | §0, §1, F1, F4 | subagent 1 §A, §B, §G |
| C2 — Silent SQLite default | F2 | subagent 1 §C, §G item 2; `config.py:118-121` |
| C3 — Strictness inconsistency across entrypoints | F3 | subagent 1 §F, §G item 6 |
| C4 — Artifact backend ignores DB target | F5 | subagent 3 §B, §G; `artifact_service.py:86-145` |
| C5 — URI columns unconstrained | F6 | subagent 3 §D |
| C6 — Only validator is sweep-only | F7 | subagent 3 §E; `scripts/ingest_sweep_to_db.py:312-329, 355` |
| C7 — No Local→Jetstream promote path | F8, §9 | subagent 4 §A, §B, §D (script names corrected against `scripts/dump_postgres_for_jetstream_migration.py`, `scripts/restore_pg_dump_to_local_docker.py`, `scripts/upload_jetstream_pg_dump_to_blob.py`, `deploy/jetstream/restore_pg_dump_to_compose_db.sh`, `deploy/jetstream/jetstream_pgvector_restore.sh` — see §12) |
| C8 — Tests don't assert lane | F9 | subagent 5 §A, §E |
| C9 — Doc/code drift | F10 | subagent 6 §B |
| L1 — 11 of 31 polluted rows | §0, §3 | `live_count_output.txt:8-9` |
| L2 — All 11 belong to bank77_contrast | §3.4 | `live_count_output.txt:24-34` |
| L3 — `raw_calls` mirrors `call_artifacts` | §3.2 | `live_count_output.txt:38-55` |
| L4 — Other JSON columns currently empty | §3.3 | `live_count_output.txt:60-92` |
| P1 — `WriteIntent` design | §6.1 | new — synthesised from F1+F2+F4 |
| P2 — Constructor wiring | §6.2 | based on `_base_connection.py:105-110` |
| P3 — Preflight banner | §6.3 | new — addresses F1 |
| P4 — `ArtifactService` coupling | §6.4 | based on `artifact_service.py:86-145`, subagent 3 §C "`pipeline/runner.py:150-153`" |
| P5 — DB constraint on `call_artifacts.uri` | §6.5 | based on F6 + L1 |
| P6 — CI enforcement | §6.6 | based on subagent 1 §D + L1 |
| P7 — Test isolation hardening | §6.7 | subagent 5 §E |
| P8 — Doc updates | §6.8 | subagent 6 §C |
| P9 — Migration checklist | §6.10 | subagent 1 §D + subagent 2 §A + Codex audit additions: re-derived via runtime-scoped `rg "DatabaseConnectionV2\("` and `rg "create_engine\("` (scope flags shown in §6.10 — `-g 'src/**/*.py' -g 'panel_app/**/*.py' -g 'scripts/**/*.py' -g 'experiments/**/*.py' -g 'deploy/**/*.py' -g '!scripts/history/**' -g '!**/tests/**' -g '!docs/audit/**'`) against commit 8e95253; subagent 2 §A's "exhaustive" claim was incorrect (missed at least 16 sites in `scripts/`, the `pipeline/snapshot.py` constructor, and `analysis/mcq_analyze_request.py`). Stale citation `scripts/restore_jetstream_db.py` removed. The `docs/audit/**` exclusion in the re-derivation prevents `docs/audit/db_target_lane_audit_2026-04-24/live_count.py:410` from showing up as a false positive. See §12. |
| R1 — 11-row remediation | §7 | live_count_output.txt:24-34 |
| R2 — Failure handling for missing files | §7 | subagent 2 §A `data_quality_service` row + general repo patterns |

---

## 12. Confidence and limitations

**Confidence: high** for the structural findings (§4) — every one is directly
visible in source code and confirmed by at least one subagent and one direct
citation in this document.

**Confidence: high** for the live-count numbers (§3) — the SQL was executed
against the live Jetstream Postgres and the script source is checked in
(`live_count.py`) so the result is reproducible.

**Confidence: medium** for the "writer identity" attribution in §3.4 — the
path prefix and artifact types are unambiguous, but only inspection of the
git log on `bank77_contrast` activity could confirm the exact run that
produced the rows. That extra confirmation is not necessary for the
remediation plan to proceed.

**Limitation:** subagent 1 §H notes that for any file not appearing in the
mechanical resolution-token sweep, the claim is "no embedded token in file,"
not "never loads DB at runtime through imports." A full call-graph audit
would close that gap; the present audit considers it acceptable because the
chokepoint design (§6.2) catches by construction any path that reaches
`BaseDatabaseConnection`.

**Limitation:** the live count was a point-in-time read. New writers between
the read and the constraint add (§6.5) could introduce additional polluted
rows. The phasing in §6.9 mitigates this by adding the constraint **before**
remediation.

**Limitation:** this audit does not cover non-Postgres-Jetstream targets
(e.g., Neon). `NEON_DATABASE_URL` is read in
`scripts/ingest_mcq_probe_json_to_sweep_db.py:139-141` (subagent 1 §B) but
no live data was probed there. If Neon is in active use it should be added to
`Lane` and audited separately.

**Limitation (subagent 4 stale citations).** Subagent 4's report (and
therefore the original drafts of §0 finding 4, §5 F8, and §9) referenced two
script names that do not exist in this repo:
`scripts/restore_jetstream_db.py` and `scripts/dump_jetstream_db.py`. The
Codex audit on commit 8e95253 caught the discrepancy. The actual scripts
that implement the canonical-side dump/restore workflow are:

- `scripts/dump_postgres_for_jetstream_migration.py` — `pg_dump` against any
  source DB (Jetstream or local) into a `.dump` file. Read-only against the
  source.
- `scripts/restore_pg_dump_to_local_docker.py` — restores a `.dump` file
  into the local Docker Postgres.
- `scripts/upload_jetstream_pg_dump_to_blob.py` — pushes a `.dump` file to
  Azure blob `db-backups` for offsite retention.
- `deploy/jetstream/restore_pg_dump_to_compose_db.sh` — low-level restore
  step: restores a `.dump` file into the Jetstream VM compose DB (run on
  the Jetstream VM, not locally). Destructive on canonical. Can be
  invoked directly when the compose stack is already up.
- `deploy/jetstream/jetstream_pgvector_restore.sh` — high-level wrapper
  for a fresh restore on the Jetstream VM. Stops the compose stack,
  removes the Postgres volume, starts the pgvector-enabled db,
  **invokes `restore_pg_dump_to_compose_db.sh`** (line 80) to load the
  dump, verifies the `vector` extension, then starts the app. This is
  the operator-facing entry point for full restores.

§0 finding 4, §5 F8, §6.10, §9, and §11 row C7 were rewritten to use these
corrected script names. The structural conclusions (no incremental flush
mechanic; only coarse-grained dump-replace promotion exists) are unchanged
— only the file paths were wrong.

**Codex audit pass 2 follow-up.** The first pass-1 fix introduced a
secondary defect in §9: it described `restore_pg_dump_to_compose_db.sh`
as calling `jetstream_pgvector_restore.sh`, which is the inverse of the
actual call direction shown in `jetstream_pgvector_restore.sh:66,80`. §9
was rewritten in pass 2 to reflect the correct wrapper-invokes-restore
direction, and the pass-2 §12 entry above documents both scripts'
relationship explicitly so the inversion is harder to reintroduce.

**Limitation (P9 / migration checklist completeness).** The original §6.10
draft claimed to be "exhaustive against subagents 1 §D and 2 §A" but the
Codex audit on commit 8e95253 found at least 16 in-tree
`DatabaseConnectionV2(...)` constructor sites that subagent 2 §A had missed
(notably under `scripts/` and the `pipeline/*.py` stage entry points).
§6.10 was rewritten in pass 1 to: (a) drop the "exhaustive" claim,
(b) include the missing sites grouped by intent, and (c) provide a
re-derivation command (`rg "DatabaseConnectionV2\("` +
`rg "create_engine\("`) so the list can be regenerated as the repo drifts.
The migration plan is still actionable; the defect was in the documentation's
confidence framing, not in the design.

**Pass 2 follow-up (Codex audit pass 2).** The pass-1 re-derivation `rg`
commands were unscoped (`--type py` over the whole repo), which would
have surfaced false positives such as
`docs/audit/db_target_lane_audit_2026-04-24/live_count.py:410` — an
audit-doc probe that intentionally bypasses the chokepoint to read the
live DB. §6.10 was rewritten in pass 2 to (a) restrict the scope of the
checklist to "runtime surfaces" only, (b) attach `-g` glob filters to
the re-derivation commands so they search only the runtime tree, and
(c) add an explicit out-of-scope list (`docs/audit/**`, `tests/**`,
`scripts/history/**`, `notebooks/**`, untracked scratch directories)
with the rationale for each exclusion. The §6.6 CI lint inherits the
same scope: only runtime surfaces are checked for chokepoint compliance.

---

## 13. Summary for Codex

This document plus the eight companion files in this directory should be
sufficient for an independent auditor to:

1. Reproduce the live count (`live_count.py` is self-contained and
   read-only).
2. Verify every code citation by reading the cited line ranges.
3. Cross-check that every recommendation in §6 traces to a finding in §4 and
   to a subagent report.
4. Decide whether to accept the plan, request modifications, or propose an
   alternative.

If Codex finds a citation that does not match the cited file/line, that is a
defect in this synthesis; please flag it specifically and do not generalise
("the synthesis is unreliable") — the subagent files are the raw record and
should be consulted before adjusting confidence in unrelated claims.
