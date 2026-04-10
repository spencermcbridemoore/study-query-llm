# Audit 1

## Metadata
- Source chatbot: Codex
- Source file: `C:\Users\spenc\OneDrive\Documents\Codex.txt`
- Status: imported

## Full Response
1) Executive Summary
main is clean at d3e095a, tracking origin/main and ahead by 7 commits; no staged/unstaged/untracked files.
Incident window cc49120..salvage-backup-c43e483 contains 3 commits and 51 changed files; 49 of those remain in cc49120..HEAD, while 2 were artifact binaries.
Highest risk is history/provenance confusion from off-main recovery commits/tags (c43e483, 6ed6785) that include mixed WIP and generated .npy artifacts.
Post-cleanup stack is mostly logical by feature area, but 8dcfbda still bundles config/scripts/notebook/runtime changes with weak direct test coverage for that cluster.
No direct secret-file exposure was found in the incident file set (no .env/key files in changed-file lists), but guardrails are weak enough to merit a pre-push secret scan.
2) Findings (ordered by severity)
Critical — None found in current HEAD push path. Evidence: clean tree, no uncommitted residue, and recovery artifact commits are not ancestors of HEAD.
High — Mixed-concern salvage checkpoint exists off-main and includes generated binaries. Evidence: c43e483 changes 42 files across code/docs/scripts/notebook + artifacts/.../embedding_matrix.npy; 6ed6785 is artifact-only; neither checkpoint commit is contained by local/remote branches and both sit as safety tags. Why it matters: accidental cherry-pick/tag push can reintroduce non-reviewable WIP and artifacts. Blast radius: PR integrity, auditability, repo hygiene. Confidence: high. Recommended fix: keep recovery tags local-only, avoid git push --tags, and document “do-not-merge” for these refs.
High — Artifact/cached binary guardrail is missing in ignore policy. Evidence: .gitignore does not exclude artifacts/ or *.npy, and these files were committed during recovery. Why it matters: easy recurrence of binary/checkpoint commits. Blast radius: repository growth, noisy diffs, merge risk. Confidence: high. Recommended fix: add explicit ignore rules for generated artifact paths and keep canonical artifacts in external artifact storage.
Medium — MCQ/OpenRouter tooling cluster is broad and under-tested as a unit. Evidence: 8dcfbda changes config + multiple scripts + experiment code + notebook, but no dedicated tests were added for several touched script paths. Why it matters: runtime/sweep regressions can slip through despite other service-level tests. Blast radius: sweep execution reliability and result consistency. Confidence: medium-high. Recommended fix: split notebook from runtime code and add focused tests/smoke checks for run_mcq_sweep/ingest/export helpers.
Medium — Docs parity appears incomplete for behavior-changing code commits. Evidence: major runtime/provider/provenance/storage changes in d3d67c0, 3d9a388, 808d0d7; changed-file set vs origin/main does not include docs/living/CURRENT_STATE.md or docs/review/DOC_PARITY_LEDGER.md. Why it matters: operators and future agents may rely on stale “current-state” docs. Blast radius: onboarding, operational decisions, future refactors. Confidence: medium. Recommended fix: add a docs parity commit before push.
Low — Notebook commit likely carries transient output noise. Evidence: notebooks/mcq_recent_1000_big_run_visualizer.ipynb added with multiple output cells. Why it matters: non-deterministic diffs and review burden. Blast radius: low, mostly process/review friction. Confidence: medium. Recommended fix: strip outputs or isolate notebook into standalone commit.
Low — Cross-chat attribution is mostly reconstructable but not perfectly deterministic. Evidence: thematic commit split plus transcript clues in Salvage Flow Debacle, OpenRouter Rollout Work, Snapshot Provenance Work, Reunify Runs Analyses. Why it matters: provenance confidence varies per cluster. Blast radius: forensic certainty, not runtime correctness. Confidence: medium. Recommended fix: preserve per-workstream commit isolation and include transcript ID in commit trailers for future forensics.
3) File Inventory Table
file	classification	likely_origin	risk	action
README.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
artifacts/2/embedding_matrix/embedding_matrix.npy	Generated artifact/cache/temp	salvage-artifact-tag	high	do not merge; keep external/local only
artifacts/3/embedding_matrix/embedding_matrix.npy	Generated artifact/cache/temp	salvage-artifact-tag	high	do not merge; keep external/local only
config/mcq_sweep_highschool_college_20q_3456_openrouter_6models.json	Intended feature work	mcq-openrouter-tooling	med	keep with MCQ tooling; verify runtime
docs/DATASET_SNAPSHOT_PROVENANCE.md	Docs/process update	bank77-bootstrap	low	keep with related feature docs
docs/IMPLEMENTATION_PLAN.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/README.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/USER_GUIDE.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/history/EXPERIMENT_ACTIVITY_SUMMARY.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/history/EXPERIMENT_ATTEMPT_TABLE.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/history/README.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/plans/ARCHIVE_NOTICE.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/plans/MASTER_META_PLAN.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/plans/README.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/plans/STEP-01_master_bootstrap.md	Docs/process update	docs-archive-history	low	keep in docs-only commit
docs/runbooks/README.md	Docs/process update	bank77-bootstrap	low	keep with bank77 docs
notebooks/mcq_recent_1000_big_run_visualizer.ipynb	Intended feature work	mcq-openrouter-tooling	med	split into notebook-only commit; strip outputs
scripts/create_bank77_snapshot_and_embeddings.py	Intended feature work	bank77-bootstrap	med	keep with its tests
scripts/export_mcq_sweep_option_counts_db.py	Intended feature work	mcq-openrouter-tooling	med	keep; add/confirm script test coverage
scripts/ingest_mcq_probe_json_to_sweep_db.py	Intended feature work	mcq-openrouter-tooling	med	keep; add/confirm ingest tests
scripts/probe_embedding_limits_live.py	Intended feature work	embedding-limit-probe	low	keep with existing test
scripts/run_mcq_sweep.py	Intended feature work	mcq-openrouter-tooling	med	keep; add regression/smoke test
scripts/run_openrouter_mcq_sweep_until_done.py	Intended feature work	mcq-openrouter-tooling	med	keep; add minimal smoke coverage
scripts/validate_and_backfill_run_snapshots.py	Intended feature work	provenance-lineage-unification	low	keep with validator tests
src/study_query_llm/algorithms/__init__.py	Intended feature work	provenance-lineage-unification	low	keep
src/study_query_llm/algorithms/method_plugins.py	Intended feature work	provenance-lineage-unification	med	keep with provenance tests
src/study_query_llm/config.py	Intended feature work	openrouter-runtime-discovery	low	keep with provider tests
src/study_query_llm/db/migrations/add_provenanced_runs_table.py	Intended feature work	provenance-lineage-unification	med	keep; verify migration safety/idempotency
src/study_query_llm/experiments/ingestion.py	Intended feature work	provenance-lineage-unification	low	keep with ingestion tests
src/study_query_llm/experiments/mcq_answer_position_probe.py	Intended feature work	mcq-openrouter-tooling	med	keep; add targeted tests
src/study_query_llm/experiments/mcq_run_persistence.py	Intended feature work	provenance-lineage-unification	low	keep
src/study_query_llm/providers/base.py	Intended feature work	openrouter-runtime-discovery	low	keep
src/study_query_llm/providers/factory.py	Intended feature work	openrouter-runtime-discovery	low	keep
src/study_query_llm/services/artifact_service.py	Intended feature work	storage-quota-guardrail	low	keep with quota tests
src/study_query_llm/services/embeddings/helpers.py	Intended feature work	bank77-bootstrap	low	keep with bank77 tests
src/study_query_llm/services/embeddings/service.py	Intended feature work	openrouter-runtime-discovery	low	keep with embedding tests
src/study_query_llm/services/jobs/job_reducer_service.py	Intended feature work	provenance-lineage-unification	low	keep
src/study_query_llm/services/model_registry.py	Intended feature work	openrouter-runtime-discovery	low	keep
src/study_query_llm/services/provenance_service.py	Intended feature work	provenance-lineage-unification	low	keep
src/study_query_llm/services/sweep_query_service.py	Intended feature work	provenance-lineage-unification	low	keep
src/study_query_llm/storage/azure_blob.py	Intended feature work	bank77-bootstrap	low	keep with storage tests
src/study_query_llm/storage/local.py	Intended feature work	storage-quota-guardrail	low	keep with quota tests
tests/test_providers/test_factory.py	Intended feature work	openrouter-runtime-discovery	low	keep
tests/test_scripts/test_create_bank77_snapshot_and_embeddings.py	Intended feature work	bank77-bootstrap	low	keep
tests/test_scripts/test_probe_embedding_limits_live.py	Intended feature work	embedding-limit-probe	low	keep
tests/test_scripts/test_validate_and_backfill_run_snapshots.py	Intended feature work	provenance-lineage-unification	low	keep
tests/test_services/test_artifact_service_quota.py	Intended feature work	storage-quota-guardrail	low	keep
tests/test_services/test_embedding_service.py	Intended feature work	openrouter-runtime-discovery	low	keep
tests/test_services/test_model_registry.py	Intended feature work	openrouter-runtime-discovery	low	keep
tests/test_services/test_provenance_delta_lineage.py	Intended feature work	provenance-lineage-unification	low	keep
tests/test_storage/test_azure_blob.py	Intended feature work	bank77-bootstrap	low	keep
4) Proposed Clean Commit Stack (old -> new)
From cc49120:

feat(bank77): bootstrap snapshot and resilient artifact reads
Files: scripts/create_bank77_snapshot_and_embeddings.py, src/study_query_llm/services/embeddings/helpers.py, src/study_query_llm/storage/azure_blob.py, related tests, plus docs/DATASET_SNAPSHOT_PROVENANCE.md and docs/runbooks/README.md.
feat(benchmark): add live embedding limit probe utility
Files: scripts/probe_embedding_limits_live.py + tests/test_scripts/test_probe_embedding_limits_live.py.
feat(openrouter): add embedding discovery and runtime limits
Files: src/study_query_llm/config.py, src/study_query_llm/providers/base.py, src/study_query_llm/providers/factory.py, src/study_query_llm/services/model_registry.py, src/study_query_llm/services/embeddings/service.py, related tests.
feat(provenance): wire snapshot linkage and lineage checks
Files: scripts/validate_and_backfill_run_snapshots.py, provenance/ingestion/query files, src/study_query_llm/db/migrations/add_provenanced_runs_table.py, related tests.
feat(storage): enforce artifact quota before local writes
Files: src/study_query_llm/services/artifact_service.py, src/study_query_llm/storage/local.py, tests/test_services/test_artifact_service_quota.py.
feat(mcq): add OpenRouter sweep tooling and option-count exports
Files: MCQ config + scripts + probe/runtime glue; exclude notebook from this commit.
chore(notebook): add MCQ run visualizer (optional, standalone)
File: notebooks/mcq_recent_1000_big_run_visualizer.ipynb (prefer output-stripped).
docs: archive planning workflow and add experiment history summaries
Files: README.md, docs/IMPLEMENTATION_PLAN.md, docs/README.md, docs/USER_GUIDE.md, docs/history/*, docs/plans/*.
docs(parity): update living state and parity ledger for runtime changes (new recommended commit before push).
What should NOT be committed

artifacts/** generated matrices/checkpoints (specifically artifacts/*/embedding_matrix/embedding_matrix.npy).
Any cache/temp outputs (__pycache__, .pytest_cache, ad-hoc scratch outputs), and secrets (.env, credentials).
What to isolate to ignore/artifact storage

Add ignore coverage for generated artifact paths (at minimum artifacts/ and *.npy under generated output roots).
Keep canonical matrices/manifests in artifact storage (Azure/local artifact backend), not git.
5) Pre-Push Gate Checklist

git status -sb
shows clean tree and expected branch (
main
ahead expected count).

git diff --name-status origin/main...HEAD
contains only intentional file groups (no
artifacts/
binaries).

Recovery refs remain non-merge inputs (
salvage-backup-c43e483
,
artifacts-checkpoint-6ed6785
not rebased/cherry-picked).

Run targeted tests by workstream, including MCQ script/sweep smoke checks for
8dcfbda
-equivalent content.

Run secret scan over staged diff (
.env
, tokens, keys, connection strings).

Add docs parity updates for feature commits (living status + parity ledger).

Push branch only (avoid pushing recovery tags unless explicitly intended).
6) Unknowns / Assumptions
Assumed the current remote does not have recovery tags (remote tag lookup for salvage-backup-c43e483 and artifacts-checkpoint-6ed6785 returned no matches).
Secret-risk assessment is based on changed-file metadata/pattern checks, not a full semantic secret detector over every line of every file revision.
Cross-chat attribution beyond commit-message themes relies on transcript clues and is strongest for salvage/cleanup flow, weaker for exact per-file origin in mixed commits.
Notebook risk is inferred from structure/output presence, not semantic validation of each output cell’s content.
