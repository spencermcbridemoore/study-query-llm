# Audit 5

## Metadata
- Source chatbot: Opus
- Source file: `C:\Users\spenc\OneDrive\Documents\Opus.txt`
- Status: imported

## Full Response
Forensic Git Audit Report
Repo: study-query-llm | Branch: main | HEAD: d3e095a | Date: 2026-04-09

1. Executive Summary
Main is 7 commits ahead of origin/main, all authored in a ~20-minute burst (21:24–21:44 EDT on 2026-04-09) — these were reconstructed from a salvage checkpoint, not developed organically, creating forensic ambiguity around per-commit provenance.
Binary .npy artifact files were successfully separated from the push-ready stack onto a dangling tag (artifacts-checkpoint-6ed6785), but .gitignore has no guard against artifacts/ or *.npy being re-tracked.
No secrets or credentials were found in the diff. Config changes reference env vars by name only, and .env files remain gitignored.
A committed notebook contains stale execution state including a captured AttributeError, which will ship noise into the repo if pushed as-is.
Two orphaned stash entries predate the cleanup and overlap with files in the current commit stack, creating re-application risk.
2. Findings (ordered by severity)
F-01 | HIGH | Missing .gitignore guard for generated artifacts
Attribute	Detail
Evidence	.gitignore has zero rules matching artifacts/, *.npy, or embedding_matrix. Binary .npy files (2,592 bytes each) were tracked in salvage commit c43e483 and isolated to tag 6ed6785, but nothing prevents re-tracking.
Why it matters	Any future git add . or careless stage will re-introduce binary blobs into history, permanently inflating the repo.
Blast radius	Entire repo clone size; CI pipelines; any contributor checkout.
Confidence	High — verified by reading .gitignore (no match) and confirming git ls-tree HEAD -- artifacts/ returns empty.
Recommended fix	Add artifacts/ and *.npy to .gitignore before pushing.
F-02 | MEDIUM | Notebook shipped with execution state and error output
Attribute	Detail
Evidence	notebooks/mcq_recent_1000_big_run_visualizer.ipynb (commit 8dcfbda) has execution_count: 2, 3, 4, an output_type: "error" cell with ename: "AttributeError", evalue: "type object 'Group' has no attribute 'deleted_at'".
Why it matters	Ships a broken execution trace implying schema incompatibility (Group.deleted_at missing) — confusing for anyone who opens the notebook, and leaks local schema state.
Blast radius	Notebook users; anyone reviewing data analysis tooling.
Confidence	High — error text confirmed in diff content.
Recommended fix	Clear all outputs (jupyter nbconvert --clear-output --inplace) before push.
F-03 | MEDIUM | Migration script with no rollback path
Attribute	Detail
Evidence	src/study_query_llm/db/migrations/add_provenanced_runs_table.py (commit 3d9a388) — creates provenanced_runs table with checkfirst=True but no drop_table down-migration or version tracking.
Why it matters	If the table definition is wrong or needs schema changes, there's no automated undo. Idempotent create is good, but operational recovery requires manual DDL.
Blast radius	Production database; any environment running the migration.
Confidence	Medium — checkfirst=True mitigates re-run risk, but absence of down path is a process gap.
Recommended fix	Add a remove_provenanced_runs_table() down function, or document manual rollback DDL in a runbook. Low urgency if table is not yet in production.
F-04 | MEDIUM | Reconstructed commit stack obscures true development timeline
Attribute	Detail
Evidence	All 7 commits have author dates within 20 minutes (21:24–21:44). Salvage commit c43e483 (42 files, ~3,049 lines) was split into 7 topical commits. The split happened during recovery in Salvage Flow Debacle.
Why it matters	git blame and git bisect lose meaningful time-ordering; all commits appear simultaneous. Actual development likely spanned multiple sessions across 2026-04-09 (and possibly earlier).
Blast radius	Future debugging, code archaeology, audit trail.
Confidence	High — timestamp clustering is self-evident.
Recommended fix	Acceptable for push as-is (topical split is the right call), but add a note in the PR body acknowledging the batch reconstruction. No git history rewrite needed.
F-05 | LOW | Orphaned stash entries with overlapping file scope
Attribute	Detail
Evidence	stash@{0}: "wip before merge origin/main" (9 files including config.py, factory.py, DOC_PARITY_LEDGER.md). stash@{1}: "wip: before force-sync origin/main 2026-03-29" (45 files, major earlier work). Both overlap with files in the current commit stack.
Why it matters	Accidental git stash pop could introduce conflict or overwrite the cleaned-up versions.
Blast radius	Local workspace only — does not affect remote.
Confidence	Medium — overlap confirmed, but stash content could be superseded.
Recommended fix	After verifying no unique work remains, git stash drop both entries post-push.
F-06 | LOW | Commit d3e095a mixes archival and new content creation
Attribute	Detail
Evidence	Single commit both archives docs/plans/ (adds ARCHIVE_NOTICE.md, modifies README.md) AND creates entirely new docs/history/EXPERIMENT_ACTIVITY_SUMMARY.md + EXPERIMENT_ATTEMPT_TABLE.md.
Why it matters	Minor hygiene issue — "archive old docs" and "add experiment history ledger" are distinct intents.
Blast radius	Minimal — only affects commit provenance clarity.
Confidence	High.
Recommended fix	Acceptable as-is for this push. Note for future process.
3. File Inventory Table
File	Classification	Likely Origin	Risk	Action
scripts/create_bank77_snapshot_and_embeddings.py	Intended feature work	BANK77 bootstrap workstream	Low	Push
scripts/probe_embedding_limits_live.py	Intended feature work	Embedding benchmarking workstream	Low	Push
scripts/export_mcq_sweep_option_counts_db.py	Intended feature work	MCQ analysis workstream	Low	Push
scripts/run_openrouter_mcq_sweep_until_done.py	Intended feature work	MCQ/OpenRouter sweep workstream	Low	Push
scripts/ingest_mcq_probe_json_to_sweep_db.py	Intended feature work	MCQ ingestion workstream	Low	Push
scripts/run_mcq_sweep.py	Intended feature work	MCQ sweep workstream	Low	Push
scripts/validate_and_backfill_run_snapshots.py	Intended feature work	Provenance workstream	Low	Push
src/study_query_llm/providers/factory.py	Intended feature work	OpenRouter provider workstream	Low	Push
src/study_query_llm/providers/base.py	Intended feature work	OpenRouter provider workstream	Low	Push
src/study_query_llm/config.py	Intended feature work	OpenRouter provider workstream	Low	Push
src/study_query_llm/services/embeddings/helpers.py	Intended feature work	BANK77/embedding workstream	Low	Push
src/study_query_llm/services/embeddings/service.py	Intended feature work	OpenRouter embedding workstream	Low	Push
src/study_query_llm/services/model_registry.py	Intended feature work	OpenRouter discovery workstream	Low	Push
src/study_query_llm/services/artifact_service.py	Intended feature work	Storage quota workstream	Low	Push
src/study_query_llm/services/provenance_service.py	Intended feature work	Provenance workstream	Low	Push
src/study_query_llm/services/sweep_query_service.py	Intended feature work	Provenance workstream	Low	Push
src/study_query_llm/services/jobs/job_reducer_service.py	Intended feature work	Provenance workstream	Low	Push
src/study_query_llm/storage/azure_blob.py	Intended feature work	BANK77 resilience workstream	Low	Push
src/study_query_llm/storage/local.py	Intended feature work	Storage quota workstream	Low	Push
src/study_query_llm/algorithms/__init__.py	Intended feature work	Provenance/methods workstream	Low	Push
src/study_query_llm/algorithms/method_plugins.py	Intended feature work	Provenance/methods workstream	Low	Push
src/study_query_llm/db/migrations/add_provenanced_runs_table.py	Intended feature work	Provenance workstream	Medium (F-03)	Push with rollback note
src/study_query_llm/experiments/ingestion.py	Intended feature work	Provenance workstream	Low	Push
src/study_query_llm/experiments/mcq_run_persistence.py	Intended feature work	Provenance workstream	Low	Push
src/study_query_llm/experiments/mcq_answer_position_probe.py	Intended feature work	MCQ workstream	Low	Push
config/mcq_sweep_highschool_college_20q_3456_openrouter_6models.json	Intended feature work	MCQ config workstream	Low	Push
notebooks/mcq_recent_1000_big_run_visualizer.ipynb	Intended feature work	MCQ analysis workstream	Medium (F-02)	Clear outputs first
tests/test_providers/test_factory.py	Intended feature work	OpenRouter provider workstream	Low	Push
tests/test_services/test_embedding_service.py	Intended feature work	OpenRouter embedding workstream	Low	Push
tests/test_services/test_model_registry.py	Intended feature work	OpenRouter discovery workstream	Low	Push
tests/test_services/test_artifact_service_quota.py	Intended feature work	Storage quota workstream	Low	Push
tests/test_services/test_provenance_delta_lineage.py	Intended feature work	Provenance workstream	Low	Push
tests/test_scripts/test_create_bank77_snapshot_and_embeddings.py	Intended feature work	BANK77 workstream	Low	Push
tests/test_scripts/test_probe_embedding_limits_live.py	Intended feature work	Embedding benchmark workstream	Low	Push
tests/test_scripts/test_validate_and_backfill_run_snapshots.py	Intended feature work	Provenance workstream	Low	Push
tests/test_storage/test_azure_blob.py	Intended feature work	BANK77 resilience workstream	Low	Push
docs/DATASET_SNAPSHOT_PROVENANCE.md	Docs/process update	BANK77 workstream	Low	Push
docs/runbooks/README.md	Docs/process update	BANK77 workstream	Low	Push
docs/IMPLEMENTATION_PLAN.md	Docs/process update	Archival workstream	Low	Push
docs/USER_GUIDE.md	Docs/process update	Archival workstream	Low	Push
docs/README.md	Docs/process update	Archival workstream	Low	Push
README.md	Docs/process update	Archival workstream	Low	Push
docs/plans/ARCHIVE_NOTICE.md	Docs/process update	Archival workstream	Low	Push
docs/plans/MASTER_META_PLAN.md	Docs/process update	Archival workstream	Low	Push
docs/plans/README.md	Docs/process update	Archival workstream	Low	Push
docs/plans/STEP-01_master_bootstrap.md	Docs/process update	Archival workstream	Low	Push
docs/history/EXPERIMENT_ACTIVITY_SUMMARY.md	Docs/process update	Experiment history workstream	Low	Push
docs/history/EXPERIMENT_ATTEMPT_TABLE.md	Docs/process update	Experiment history workstream	Low	Push
docs/history/README.md	Docs/process update	Experiment history workstream	Low	Push
artifacts/2/embedding_matrix/embedding_matrix.npy	Generated artifact	BANK77 execution output	High (F-01)	Do NOT push (on tag only)
artifacts/3/embedding_matrix/embedding_matrix.npy	Generated artifact	BANK77 execution output	High (F-01)	Do NOT push (on tag only)
4. Proposed Clean Commit Stack (old → new)
The current 7-commit stack is already well-organized topically. The split from the salvage was done competently. No re-split is required. The recommended pre-push actions are additive:

Order	Existing Commit	Verdict
1	a4c7235 feat(bank77): bootstrap flow + resilient reads	Keep as-is — cohesive BANK77 workstream
2	33d7331 feat(benchmark): embedding limit probe	Keep as-is — self-contained script + test
3	d3d67c0 feat(openrouter): embedding discovery + limits	Keep as-is — cohesive provider workstream
4	3d9a388 feat(provenance): execution linkage + lineage	Keep as-is — cohesive provenance workstream
5	808d0d7 feat(storage): quota guardrail	Keep as-is — tightly scoped
6	8dcfbda feat(mcq): sweep helpers + option exports	Needs fix — clear notebook outputs before push
7	d3e095a docs: archive + experiment history	Keep as-is — minor mixed-intent, acceptable
NEW	(add before push)	*chore: add artifacts/ and .npy to .gitignore
If you want a maximally clean history, commit 6 should be amended to clear notebook outputs, and a new commit 8 should add the .gitignore guard. Since the branch hasn't been pushed, amending is safe.

5. Pre-Push Gate Checklist

Add .gitignore rules
:
artifacts/
and
*.npy
patterns

Clear notebook outputs
:
jupyter nbconvert --clear-output --inplace notebooks/mcq_recent_1000_big_run_visualizer.ipynb

Run full test suite
:
pytest tests/
to confirm nothing is broken

Verify no binary files in push
:
git diff --stat origin/main..HEAD
should show zero binary entries (currently clean)

Verify no secrets in push
: Pattern scan completed — clean (confirmed above)

Verify living docs parity
:
CURRENT_STATE.md
last reviewed 2026-04-09,
DOC_PARITY_LEDGER.md
shows zero open high/medium items — confirmed current

Verify stash is safe to keep
: Review
stash@{0}
and
stash@{1}
contents after push; drop if superseded

PR body note
: Acknowledge that the 7-commit stack was reconstructed from a salvage checkpoint; link to
Salvage Flow Debacle
for provenance context

Verify tags are NOT pushed
:
salvage-backup-c43e483
and
artifacts-checkpoint-6ed6785
are local recovery tags — do not push unless intentionally archiving
6. Unknowns / Assumptions
Item	Status	Notes
True development timeline	Unknown	Commits were batch-reconstructed; actual development may span multiple days/sessions before 2026-04-09. Unable to determine from git evidence alone.
Stash content uniqueness	Uncertain	stash@{0} overlaps 9 files with the current stack. Cannot confirm whether it contains any unique work without a deep diff. Safe assumption: superseded.
Migration script applicability to production	Unknown	add_provenanced_runs_table.py uses checkfirst=True but it's unclear whether the table already exists in production. If it does, the script is a no-op (safe).
Notebook error origin	Low confidence	The Group.deleted_at AttributeError in the notebook suggests it was run against a schema version that doesn't have deleted_at on Group. This may be from stash@{1} era work (which included add_group_graph_audit_and_delete_guards.py — a migration that would add such a column). The notebook execution predates that migration being applied.
Test coverage completeness	Medium confidence	All 7 new feature files have corresponding test files. Coverage depth not audited (would require running pytest with --cov).
Cross-session attribution	Medium confidence	File clustering strongly suggests 5 distinct workstreams (BANK77 bootstrap, embedding benchmarking, OpenRouter providers, provenance unification, MCQ sweep tooling) plus a docs cleanup. These align with the commit messages. Attribution to specific chat sessions beyond the salvage transcript is not possible from git evidence alone.
