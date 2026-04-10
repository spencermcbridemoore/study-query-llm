# Audit 4

## Metadata
- Source chatbot: Grok
- Source file: `C:\Users\spenc\OneDrive\Documents\Grok.txt`
- Status: imported

## Full Response
1) Executive Summary

Working tree is clean; main is ahead of origin/main by exactly 7 coherent commits (cc49120..d3e095a); no staged/unstaged/untracked files.
All changes in the post-cleanup window (and incident window cc49120..salvage-backup-c43e483 references) are deliberate feature + documentation work aligned with docs/living/CURRENT_STATE.md, ARCHITECTURE_CURRENT.md, and DOC_PARITY_LEDGER.md (last reviewed 2026-04-09).
No secrets, generated binaries, cache artifacts, or untracked sensitive content were committed (.env and large experimental_results/ remain untracked).
Salvage tags (salvage-backup-c43e483, artifacts-checkpoint-6ed6785) no longer appear in the active tag list; the current stack represents successful recovery and incremental hardening (provenance unification, OpenRouter embeddings, artifact quotas, BANK77 bootstrap, MCQ sweeps).
Hygiene is strong overall; minor mixed-concern commits and large tracked analysis artifacts are the only notable items. No critical risks to merge/push.
2) Findings (ordered by severity)

Medium

Large tracked notebook (notebooks/mcq_recent_1000_big_run_visualizer.ipynb) and 1372-line bootstrap script (scripts/create_bank77_snapshot_and_embeddings.py) increase repo size and contain analysis outputs. Evidence: listed in git diff cc49120..HEAD; recent views confirm scale. Why it matters: bloats clone times and mixes generated data with source. Blast radius: repo size only. Confidence: high. Recommended fix: consider moving heavy outputs to untracked experimental_results/ or artifact storage (already partially done per history summary).
Minor lingering modifications to docs/plans/* files after adding ARCHIVE_NOTICE.md. Evidence: commit d3e095a. Why it matters: slightly blurs historical vs living doc boundary. Blast radius: documentation clarity. Confidence: med. Recommended fix: ensure all plans/ content is frozen or moved to docs/history/.
Low

Several commits mix code + docs/tests (e.g., BANK77 commit updates DATASET_SNAPSHOT_PROVENANCE.md + code + tests; provenance commit mixes migration + backfill script). Evidence: per-commit --name-status output. Violates ideal "one concern per commit" but does not break tests or parity ledger. Blast radius: history readability. Confidence: high.
No updates to .cursorrules, STANDING_ORDERS.md, or CONTRIBUTING.md in this stack despite doc changes. Evidence: absent from diff. Minor process gap. Confidence: med.
Salvage tags and exact incident window commits are not easily visible in current git tag / git log without the referenced transcript. Evidence: tags missing; transcript cited as [Salvage Flow Debacle](b75e5b27-b72f-4fea-8443-0204f3c74a24). Reduces provenance clarity. Confidence: high.
No Critical or High findings. No secret exposure, no tracked caches/binaries, parity ledger confirms alignment with living docs, and v2/provenance patterns were followed.

3) File Inventory Table

file	classification	likely_origin	risk	action
docs/* (CURRENT_STATE.md, ARCHITECTURE_CURRENT.md, DOC_PARITY_LEDGER.md, history/ summaries, plans/ARCHIVE_NOTICE.md, README.md, IMPLEMENTATION_PLAN.md, etc.)	Docs/process update	Post-salvage cleanup + living doc maintenance (aligned with AGENTS.md and transcript)	Low	Keep
scripts/create_bank77_snapshot_and_embeddings.py, tests/test_scripts/test_create_bank77_snapshot_and_embeddings.py, src/study_query_llm/storage/azure_blob.py, src/study_query_llm/services/embeddings/helpers.py, docs/DATASET_SNAPSHOT_PROVENANCE.md	Intended feature work	BANK77 bootstrap workstream (explicitly called out in CURRENT_STATE.md)	Low (large script)	Keep
src/study_query_llm/services/artifact_service.py, src/study_query_llm/storage/local.py, tests/test_services/test_artifact_service_quota.py	Intended feature work	Storage hardening / quota guardrail	Low	Keep
src/study_query_llm/services/provenance_service.py, scripts/validate_and_backfill_run_snapshots.py, new migration, method_plugins.py, ingestion/mcq persistence updates, related tests	Intended feature work	Provenance unification / v2 execution linkage (core to CURRENT_STATE and ARCHITECTURE_CURRENT)	Low	Keep
src/study_query_llm/providers/factory.py, model_registry.py, embeddings/service, config updates, OpenRouter tests	Intended feature work	OpenRouter embedding discovery + limits	Low	Keep
scripts/probe_embedding_limits_live.py, related test	Intended feature work	Benchmark/live probing	Low	Keep
MCQ scripts (run_openrouter_mcq_sweep_until_done.py, export_mcq_sweep_option_counts_db.py, run_mcq_sweep.py, mcq_answer_position_probe.py), config JSON, notebook	Intended feature work	MCQ + OpenRouter sweep enhancements	Low	Keep
notebooks/mcq_recent_1000_big_run_visualizer.ipynb	Generated artifact/cache/temp (analysis output)	Experiment visualization from MCQ runs (see EXPERIMENT_ACTIVITY_SUMMARY.md)	Medium	Monitor size; consider untracking outputs
All other test files	Intended feature work	Accompanying each feature	Low	Keep
(No "Risky/sensitive", "Unrelated carryover", or "Unknown" files; all clusters have clear commit + doc evidence.)

4) Proposed Clean Commit Stack (old → new) The existing 7 commits are already logically grouped and mostly atomic. Proposed refinement (no rebase needed, just for future reference):

feat(bank77): add bootstrap flow and resilient artifact reads (script + helpers + azure_blob + test + provenance doc)
feat(benchmark): add live embedding limit probe script (probe + test)
feat(openrouter): discover embedding models and runtime limits (factory, registry, embeddings, config, tests)
feat(provenance): unify execution linkage and snapshot lineage checks (migration, provenance_service, ingestion, backfill, tests)
feat(storage): enforce artifact quota guardrail before writes (artifact_service, local storage, quota test)
feat(mcq): add OpenRouter sweep helpers and option-count exports (MCQ scripts, notebook, config)
docs: archive planning workflow and add experiment history summaries (all docs/history/plans updates + README)
What should NOT be committed: The large notebook's raw output cells if they grow further; any experimental_results/ or logs/ content (already untracked).
What to isolate: Analysis artifacts → untracked dirs or Azure blob / dedicated artifact storage (already the pattern via artifact_service). No .gitignore changes needed.

5) Pre-Push Gate Checklist


git status
confirms clean working tree

All tests pass (
pytest
, including new quota/provenance/embedding tests; mark API tests appropriately)

docs/living/CURRENT_STATE.md
,
ARCHITECTURE_CURRENT.md
, and
DOC_PARITY_LEDGER.md
remain accurate (they do)

No secrets in diff (
grep -E 'sk-|AKIA|Bearer' --files-with-matches
on changed files returns nothing)

Large files under control (
git ls-files --stage | grep -E '\.(ipynb|pkl|bin)$'
reviewed)

Parity ledger updated for any new claims

Optional:
git log --oneline origin/main..HEAD
matches proposed clean stack

Push only after confirming no regression on Jetstream/local DB clone flows
6) Unknowns / Assumptions

Exact content of the referenced transcript [Salvage Flow Debacle](b75e5b27-b72f-4fea-8443-0204f3c74a24) was not re-read (per READ-ONLY constraint and agent transcript rules); inference based on commit messages, safety tags, and living docs is strong but assumes it documents the pre-cc49120 mess that this stack cleaned.
Whether salvage-backup-c43e483 tag still exists on another branch/ref (not visible in git tag).
Exact size/contents of untracked experimental_results/ (mentioned in history summary) was not enumerated beyond confirmation they are untracked.
Assumption that the 7-commit stack fully represents the "post-cleanup" state with no hidden carryover (supported by clean status and per-commit granularity).
This audit indicates the repository is in good shape for merge/push. The salvage incident appears successfully resolved into intentional, documented, tested feature work that aligns with project standing orders.
