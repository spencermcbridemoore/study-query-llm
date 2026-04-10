# Audit 3

## Metadata
- Source chatbot: Gemini
- Source file: `C:\Users\spenc\OneDrive\Documents\Gemini.txt`
- Status: imported

## Full Response
1) Executive Summary
Repository State: The current branch (main) is ahead of origin/main by 7 commits, with a clean working tree. The incident window (cc49120..salvage-backup-c43e483) contained 51 changed files, which have now been cleanly organized into the 7-commit stack.
Artifact Leakage Prevented: The initial salvage checkpoint accidentally tracked binary cache artifacts (artifacts/2/... and artifacts/3/...), which were correctly identified and isolated into the artifacts-checkpoint-6ed6785 tag, keeping the main branch clean.
Logical Grouping Success: The recent commit stack (cc49120..HEAD) successfully splits the monolithic salvage state into 7 distinct, logical commits (e.g., separating bank77 bootstrap, embedding probes, storage guardrails, and documentation archiving).
Documentation Parity: Extensive updates were made to docs/history/ and docs/plans/ to archive the planning workflow and reflect the current state, resolving previous parity drift.
Push Readiness: The current main branch is in a healthy, push-ready state. No secrets or untracked artifacts are present in the final commit stack.
2) Findings
1. High: Accidental Tracking of Binary Artifacts (Resolved in HEAD)

Evidence: artifacts/2/embedding_matrix/embedding_matrix.npy and artifacts/3/embedding_matrix/embedding_matrix.npy were added in the salvage-backup-c43e483 checkpoint.
Why it matters: Tracking generated .npy binaries bloats the git repository history and violates the rule against tracking cache/build artifacts.
Blast radius: Repository size bloat and merge conflicts on generated data.
Confidence: High.
Recommended fix: Ensure .gitignore explicitly ignores the artifacts/ directory. The current main branch has successfully excluded these files, but .gitignore should be verified.
2. Medium: Monolithic "Salvage" Accumulation

Evidence: The salvage-backup-c43e483 commit contained 51 changed files across multiple unrelated domains (storage, embeddings, documentation, scripts, tests).
Why it matters: Accumulating changes across multiple chat sessions without staging/committing violates the user rule: "Whenever possible try to do a stage/commit/push-to-origin when finishing a checklist".
Blast radius: Difficult rollbacks, tangled history, and high risk of losing work during context switches.
Confidence: High.
Recommended fix: Enforce the standing order to stage and commit when completing features/tasks incrementally.
3. Low: Missing .gitignore Coverage for Notebook Checkpoints

Evidence: notebooks/mcq_recent_1000_big_run_visualizer.ipynb was added. While the notebook itself is intended, Jupyter often generates .ipynb_checkpoints/.
Why it matters: Potential future leakage of notebook state.
Blast radius: Minor repository clutter.
Confidence: Medium.
Recommended fix: Verify .ipynb_checkpoints/ is in .gitignore.
3) File Inventory Table (Incident Window: cc49120..salvage-backup-c43e483)
file	classification	likely_origin	risk	action
artifacts/**/*.npy	Generated artifact	Script execution	High	Remove from tracking (Done)
scripts/create_bank77_snapshot...	Intended feature work	Bank77 Bootstrap	Low	Keep
scripts/probe_embedding_limits...	Intended feature work	Embedding Limits	Low	Keep
scripts/export_mcq_sweep...	Intended feature work	MCQ Sweep	Low	Keep
src/study_query_llm/storage/*	Intended feature work	Storage Guardrails	Low	Keep
src/study_query_llm/services/*	Intended feature work	Provenance/Embeddings	Low	Keep
tests/**/*	Intended feature work	Test Coverage	Low	Keep
docs/history/*	Docs/process update	Workflow Archiving	Low	Keep
docs/plans/*	Docs/process update	Workflow Archiving	Low	Keep
notebooks/*.ipynb	Intended feature work	Analysis	Low	Keep
config/*.json	Intended feature work	Configuration	Low	Keep
4) Proposed Clean Commit Stack (old -> new)
The current main branch has already successfully implemented a clean commit stack. Here is the validation of that stack:

a4c7235 feat(bank77): add bootstrap flow and resilient artifact reads
33d7331 feat(benchmark): add live embedding limit probe script
d3d67c0 feat(openrouter): discover embedding models and runtime limits
3d9a388 feat(provenance): unify execution linkage and snapshot lineage checks
808d0d7 feat(storage): enforce artifact quota guardrail before writes
8dcfbda feat(mcq): add OpenRouter sweep helpers and option-count exports
d3e095a docs: archive planning workflow and add experiment history summaries
Note: The binary artifacts (artifacts/**/*.npy) were successfully excluded from this stack.

5) Pre-Push Gate Checklist

Verify working tree is clean (
git status
).

Verify no secrets or API keys are present in the diff (
cc49120..HEAD
).

Verify generated binaries (
.npy
) are excluded from the
main
branch commits.

Verify documentation parity matches the new implementation state (
docs/history/
,
docs/plans/
).

Verify
.gitignore
explicitly ignores the
artifacts/
directory to prevent future leakage.

Run test suite to ensure the split commits did not introduce regressions.
6) Unknowns / Assumptions
Assumption: The artifacts-checkpoint-6ed6785 tag was intentionally created to preserve the .npy files locally without pushing them to the remote main branch.
Unknown: It is unclear if the notebooks/mcq_recent_1000_big_run_visualizer.ipynb contains any hardcoded local paths or sensitive data from the execution environment, as notebook cell outputs were not deeply inspected.
Unknown: Whether the .gitignore file currently covers the artifacts/ directory, as it was not modified in the incident window.
