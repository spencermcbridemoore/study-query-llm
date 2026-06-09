# Post-audit hardening plan (v2.2)

## Goal
Close the gaps surfaced by the audits in priority order: make the governance we built actually enforce, repair stale docs inside the trusted "living" set, de-risk the bespoke analysis_request race-fix surface, then resume script cleanup behind tighter gates. D1 (push main to origin) was enacted 2026-06-03, so server CI is live. Revised after a 5-agent audit OF this plan (consensus fixes folded in); approach-forks A-D were resolved 2026-06-03 (see Decisions).

## Revisions log
- v2.1: hook timing corrected (check_living_docs_drift -> pre-push; pre-commit = blocking staged-restricted check + static lints); push-range fix promoted to a hard gate; NEW Phase-0 JETSTREAM reconciliation gate (a runbook says canonical is dormant/restore-freely, but dd38ddc wrote to a live ~3300-group canonical); canonical-write invariant fixed (delta == logged conflict-deletes, not strict conservation); 0.5 migration test de-tautologized; 2.2 decoupled + flagged Postgres-only; 2.3 reframed (analysis_run is a distinct group_type); 1.4 governance-doc rebind + scoped evidence check; 3.1 needs a real mechanism.
- v2.2: Fork decisions recorded - (A) local pre-push hook = primary gate, CI = informational backstop, no mandatory PR/required-status-checks; (B) detect-now / enforce-only-if-triggered for the half-row invariant; (C) ledger append-only in place; (D) a cheap deterministic creation-lint, not convention-only.

## Context
- Effort to date (all on origin/main): mechanical cleanup (2ed773c), living-docs-only governance (e1759d1), notebook retirements (0154d69), analysis_request race fix (dd38ddc), set-A script retirements (cf44269).
- Tier A = `src/study_query_llm/` (canonical, tested); Tier B = `scripts/` (thin CLIs + ops utilities).
- Governance: `.cursor/rules/living-docs-only.mdc`, `scripts/internal/living_docs_governance.py`, CI in `.github/workflows/{living-docs-drift,persistence-contract,docker-smoke}.yml`, evidence in `docs/review/DOC_PARITY_LEDGER.md`.
- Workspace edit constraint: Edit/Write disabled; edit via Bash+Python, preserve CRLF (autocrlf=true), encoding=utf-8.
- Known pre-existing test failures (NOT introduced here): an unregistered bank77 test method; a test coupled to a `scratch/` file by path.
- Race-fix surface: `src/study_query_llm/db/models_v2.py` (dual-dialect index SQL constants), `.../db/migrations/add_analysis_request_unique_index.py`, `.../db/raw_call_repository.py` (insert_or_get), `scripts/remediate_analysis_request_duplicates.py`.

## Approach
Sequence by risk: secure the operating state, repair trusted-set doc drift, collapse the duplicated race-fix identity logic, then resume cleanup. Split hook responsibilities by git timing - staged-path checks at pre-commit, commit-range/message checks at pre-push - because the drift tool reads committed history and messages. Prefer standard tools and shallow deterministic checks over bespoke/semantic ones. Phases 1-3 interleave once Phase 0 is done; canonical-DB, migration, and the JETSTREAM reconciliation are hard-gated.

## Steps

Phase 0 - secure the operating state
- [x] 0.1 Fix the two pre-existing test failures; green fast suite locally. Before any blocking hook.
- [x] 0.2a Enact D1: push main to origin (done 2026-06-03; activates the 3 CI workflows + backs up the commits).
- [x] 0.2b Fix the push-range bug in `.github/workflows/living-docs-drift.yml` (HEAD~1..HEAD -> github.event.before..github.event.after). HARD gate before the next multi-commit push.
- [x] 0.2c Hooks, split by timing:
    - pre-commit (fast, target <2s, static/staged only): a BLOCKING port of `warn_restricted_doc_edits.py` (staged restricted-path check) + `verify_script_path_references.py` + `check_persistence_contract.py`. No heavy tests.
    - pre-push: `check_living_docs_drift.py` over origin/main..HEAD (committed range + messages - the only place it works).
- [x] 0.2d Wire `verify_script_path_references.py` into `.github/workflows/persistence-contract.yml` so CI re-runs it on every push as an INFORMATIONAL backstop (catches a bypassed/uninstalled hook). Fork A decision: the local pre-push hook is the PRIMARY gate; CI is a safety net, NOT a required status check, and there is NO mandatory PR flow for this solo repo. Also delete the now-redundant `origin/fix/analysis-request-dup-race` branch (dd38ddc is on main); glance at the stray `origin/docs/meta-plan-workflow` branch. **(2026-06-04 done):** added the informational verify_script_path_references step to persistence-contract.yml (Fork A: visible-but-non-blocking, not a required check); both branches were fully merged into main (0 unique commits) and deleted.
- [x] 0.3 JETSTREAM reconciliation gate (before the next canonical write): the trusted `JETSTREAM_STATE_TIMELINE.md` says canonical is dormant/empty and to pg_restore a dump, but dd38ddc wrote to a live ~3300-group canonical. Neutralize the "dormant -> restore freely" guidance NOW and capture current canonical row/group-count evidence (what happened to Jetstream after Apr 23?). 0.4's rehearsal premise depends on knowing what canonical actually is. **(2026-06-04 done):** read-only inventory captured in `docs/review/CANONICAL_INVENTORY_2026-06-04.md`; canonical is ACTIVE and clean.
- [ ] 0.4 Canonical-DB-write discipline (gate before the next canonical write): rehearse destructive remediation on the local Docker clone (`scripts/restore_pg_dump_to_local_docker.py` + clone runbook) first; backup/confirm gate before any DELETE; emit a MACHINE-READABLE post-write artifact asserting: duplicates=0; FK-orphan scan clean; row delta == logged conflict-deletes (NOT strict conservation - the remediation deletes rows on natural-key collision); every run that referenced a loser now points at the keeper or is a logged conflict-delete; AND no partial-identity analysis_request rows (the 2.2 detection query). The hook can also assert the artifact. **(2026-06-04):** MOOT for the race-fix - canonical verified clean (index present, 0 duplicates, 0 half-rows), so no canonical write is pending; the clone-rehearsal + verification-artifact tooling stays unbuilt and deferred-as-unneeded until a future canonical write.
- [ ] 0.5 Minimal migration bookkeeping (gate before the next migration): `schema_migrations(name, applied_at)` + a tiny ordered runner that runs each module in a transaction and skips applied ones; record+order+applied-check only. (Cross-site SQL agreement test lives in 2.1, not here.) **(2026-06-04):** the analysis_request unique-index migration is already applied to canonical; no pending migration. Runner deferred-as-unneeded until the next migration.

Phase 1 - repair the trusted living set (interleave; but 1.2 + 1.3 BEFORE 3.3)
- [x] 1.1 Full prose refresh of `JETSTREAM_STATE_TIMELINE.md`: add a "Current state (active)" section, keep the Apr-22 chronology as historical, supersede ledger claim C053. (The urgent safety neutralization is in 0.3.)
- [x] 1.2 Fix `scripts/README.md` Framework-Lanes dead links to `history/README.md` + `deprecated/README.md` AND the `.cursor/rules/living-docs-only.mdc` "lane structure ... authoritative for membership" pointer, in one commit; add an archive-tag recovery one-liner. Do before 3.3.
- [x] 1.3 Add a test asserting the restricted-path TABLE in `living-docs-only.mdc` (the authoritative source) equals `scripts/internal/living_docs_governance.py`. Do before 3.3.
- [x] 1.4 Demote `DOC_PARITY_LEDGER.md` to an append-only decision log AND rebind the governance docs that tell agents to maintain it (`.cursorrules`, the `.mdc` parity-evidence line, `docs/STANDING_ORDERS.md` sync checklist) to `CURRENT_STATE.md` as the living mirror. Scope any evidence-path-exists check to living-set paths only (archived/tombstoned paths are intentionally gone - do not false-red). Fork C decision: append-only in place; revisit moving it to docs/review/history/ only if the 80-row history inside the living set becomes noise.

Phase 2 - de-risk the race-fix surface
- [x] 2.1 Consolidate the 4-site identity logic (PG index SQL, SQLite index SQL, migration `_DUPLICATE_PROBE_SQL`, remediation `find_duplicate_identities`) behind a shared identity-field contract + shared SQL-fragment builders (NOT one mega-abstraction); fixture asserts all four agree on PG and SQLite. Optionally add a static check that scripts touching `groups`/`analysis_request` import `ANALYSIS_REQUEST_UNIQUE_INDEX_NAME` from models_v2. **(2026-06-04 done):** shared `db/analysis_request_identity.py` (json_extract_sql + build_unique_index_sql + build_duplicate_probe_sql + extract_identity); all 4 sites rewired (index DDL byte-identical, pinned by test); agreement fixture `tests/test_db/test_analysis_request_identity.py` (SQLite always + PG opt-in via TEST_POSTGRES_URL). Repository lookup dict left as-is (internally consistent); optional static import-check skipped.
- [x] 2.2 Half-row integrity - Fork B decision: detect now, enforce only if triggered. **(2026-06-04):** detection run on canonical = 0 half-rows (enforcement not triggered); the detector is now committed as `scripts/check_analysis_request_half_rows.py` (read-only operator probe built from the shared `build_half_row_probe_sql(dialect)` contract; logic tested on SQLite in `tests/test_db/test_analysis_request_identity.py`). CI cannot reach canonical, so it is operator-run, not a CI gate.
    - NOW: add a read-only assertion flagging any analysis_request row with a PARTIAL identity (num_nulls(method_name, input_id, run_key) NOT IN (0,3)) to the 0.4 verification artifact and a light periodic/CI check. Zero DDL, no dual-dialect, no migration.
    - LATER (only if a half-row ever appears OR `groups` is being migrated anyway): promote to the enforcing partial CHECK, written as cross-dialect boolean logic over JSON-extracted fields (SQLite has no num_nulls), after a legacy half-row scan. Do NOT bundle with 2.1.
    - Rationale: no current writer can produce a half-row, so near-free detection beats a dual-dialect CHECK migration with no live trigger.
- [x] 2.3 Document intended uniqueness PER group_type: `analysis_run` is a DISTINCT type (`create_analysis_run_group`), not covered by `uq_groups_analysis_request_identity` - decide whether duplicate (method,input,run_key) `analysis_run` groups are intended (today: possible; no insert-or-get). **(2026-06-04 resolved):** on canonical, analysis_run uses a DISTINCT identity shape (snapshot_group_id + embedding_batch_group_id + method/run_key; input_id null in all 15,942 rows), correctly not covered by the index. Duplicate (method,run_key) pairs are rare (53 pairs, max mult 3) and INTENDED (execution-event records; no insert-or-get). Evidence: `docs/review/CANONICAL_INVENTORY_2026-06-04.md`. No index/insert-or-get change.
- [ ] 2.4 Postgres-backed concurrency test for the race fix (SQLite serializes writers; reuse the 0.4 clone); time-box / `workflow_dispatch` if flaky.

Phase 3 - resume cleanup behind tighter gates (after Phase 0; 1.2 + 1.3 first)
- [x] 3.1 Creation gate - Fork D decision: a cheap DETERMINISTIC lint in the static set. For each ADDED `scripts/run_*.py` in the diff, fail if no file under `tests/` references its stem (filename + sibling-test existence only; no AST / import-graph / delegates-to-Tier-A analysis). It is a human-nudging tripwire (shallow + gameable), not a correctness proof; pair it with the AGENTS.md Tier A/B boundary note. Chosen over convention-only because the parallel_fixed_k recurrence already happened once. **(2026-06-04 done):** `scripts/check_new_run_scripts_have_tests.py` (+ `tests/test_scripts/test_check_new_run_scripts_have_tests.py`) wired into the pre-commit static set; flags an ADDED root `scripts/run_*.py` whose stem no `tests/` file references (filename or content). Verified end-to-end (fails on a staged untested run script). AGENTS.md Code Layer Boundary note added.
- [ ] 3.2 Define the canonical MCQ accuracy contract BEFORE promotion: persist components (attempts, errors, abstentions, correct), derive rates. Start as a computed dataclass/function (no persisted metrics table yet). Then extract the token-matcher util and promote diagnostics into a tested `src` analysis module. **(2026-06-03 - DEFERRED/split):** the accuracy contract is captured as documented intent in `.cursor/plans/mcq-method-definition-direction.md`; the token-matcher extraction + `src`-module promotion are deferred together with the (deferred) MCQ method-definition redesign. Legacy MCQ scripts/data stay as-is.
- [x] 3.3 Resume Set B/D under a tiered retirement gate (broken/unreferenced -> delete; operator-facing w/ nonzero refs -> time-boxed deprecation warning then delete). Enumerate the answer-position chain explicitly: `run_mcq_answer_position_probe.py`, `ingest_mcq_probe_json_to_sweep_db.py`, `audit_mcq_method_definitions.py` (+ cross-refs). Decide AFTER 3.2 defines the contract. **(2026-06-03 - RESOLVED: no retirement):** the answer-position chain is Family 1 (authoring/constructor-bias generation), a DISTINCT hypothesis NOT superseded by the Family-2 logprob answering work; keep both, retire nothing, legacy data stays legacy. See `.cursor/plans/mcq-method-definition-direction.md`.
- [ ] 3.4 Consolidate clustering-backfill scripts onto the tested core: reconcile the divergent snapshot-lineage contract FIRST; extract a single-target executor; no new parallelism in core.

## Validation
- 0.1: target suite green; the two known failures gone.
- 0.2: push reflected on origin/main; pre-push hook blocks a restricted edit lacking the token; pre-commit blocks a staged restricted-path edit; push-range uses before/after SHAs; CI re-runs the static checks on push as an informational backstop.
- 0.3: current canonical row/group counts recorded; the "dormant -> restore" guidance no longer reachable as live advice.
- 0.4: a clone dry-run+apply emits the machine-readable artifact (duplicates=0, FK-orphan clean, delta == logged conflict-deletes, no partial-identity rows); canonical untouched during rehearsal.
- 0.5: runner records applied modules, skips on re-run, runs each in a transaction.
- 1.x: verify_script_path_references stays at 0 missing; the mdc<->governance sync test passes; no living doc links to nonexistent paths.
- 2.x: shared-extractor fixture passes on PG + SQLite; the 2.2 detection flags a deliberately-inserted partial-identity row (no enforcing CHECK added now).
- 3.x: the creation-lint fails a new scripts/run_*.py added without a test; MCQ rates derive identically from persisted components across consumers; retirements keep verify green.

## Decisions and open questions
Resolved forks (2026-06-03):
- Fork A (enforcement authority): local pre-push hook = primary gate; CI = informational backstop; no required status checks, no mandatory PR flow (solo repo).
- Fork B (half-row invariant): detect now (read-only assertion in 0.4 + CI); enforce the partial CHECK only if a half-row ever appears or `groups` is migrated anyway.
- Fork C (ledger fate): append-only in place + governance-doc rebind; move to docs/review/history/ only if it becomes noise.
- Fork D (creation gate): a cheap deterministic lint (filename + sibling-test), not convention-only, not semantic.
- 3.3 (MCQ supersession): the answer-position chain is a DISTINCT hypothesis (authoring/constructor-bias = "Family 1"), NOT superseded by the logprob answering family ("Family 2"). Keep both; retire nothing. Detail: `.cursor/plans/mcq-method-definition-direction.md`.

Still open:
- (none currently.)

Deferred (intent captured, not scheduled):
- MCQ method-definition redesign: fold MCQ into the method lane as composed methods (permutation-as-method -> raw_call-emitting inference -> scoring; recipe_json composition). Intent in `.cursor/plans/mcq-method-definition-direction.md`. Legacy MCQ scripts/runners and data stay as-is.

## Out of scope
- Full Alembic (a version table + ordered runner is enough).
- Blanket NOT-NULL CHECK on analysis_request (breaks the container shape).
- AST/import-graph/Dockerfile/entry-point reference tool (ripgrep reverse-pass -> WARN; reach for ripgrep/vulture).
- A ledger-automation framework / keeping the ledger "living".
- A persisted MCQ-metrics table before the analysis module exists.
- Splitting analysis_request into a distinct group_type now (document the two shapes instead).
- An enforcing half-row CHECK now (detection only, per Fork B).
- Required status checks / mandatory PR flow for this solo repo (per Fork A).
