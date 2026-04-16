# Audit 2

## Metadata
- Source chatbot: Composer
- Source file: `C:\Users\spenc\OneDrive\Documents\Composer.txt`
- Status: imported

## Full Response
1) Executive Summary
Working tree is clean (git status -sb: main...origin/main [ahead 7] only). No staged/unstaged/untracked changes reported; merge-base of HEAD and origin/main is cc49120 (same as origin/main), so the 7 commits are exactly origin/main..HEAD.
Push/merge delta is 49 files, +5956 / −135 lines vs origin/main — a coherent feature stack (Bank77 bootstrap → embedding probe → OpenRouter/model registry → provenance + migration → Azure quota → MCQ sweep tooling → docs archive).
Salvage forensics: tag salvage-backup-c43e483 is a single monolithic commit that adds binary NumPy artifacts under artifacts/*/embedding_matrix/embedding_matrix.npy plus application/docs/scripts together; c43e483 is not an ancestor of current main (recovery removed that commit from the line you will push). Tag artifacts-checkpoint-6ed6785 contains only those two .npy files and is also not on main.
High process finding: at cc49120 (origin/main), docs/living/CURRENT_STATE.md already documents BANK77 scripting and Azure blob quota enforcement, but scripts/create_bank77_snapshot_and_embeddings.py is absent and artifact_service.py has no quota logic until a4c7235 / 808d0d7 — so origin/main currently overstates shipped capability until this stack merges.
Hygiene: .gitignore does not generally ignore artifacts/ or *.npy; the salvage path shows how easily generated matrices can be committed if discipline slips.
2) Findings (ordered by severity)
Critical
None observed in origin/main..HEAD — no secrets found via targeted searches; no binary .npy paths in the push range (they appear only on c43e483 / 6ed6785, off main).
High
origin/main living doc vs code mismatch (pre-merge)

Evidence: git show cc49120:docs/living/CURRENT_STATE.md includes BANK77 bootstrap and artifact quota bullets; git ls-tree cc49120 scripts/create_bank77_snapshot_and_embeddings.py is empty (file missing); git show cc49120:src/study_query_llm/services/artifact_service.py has no quota/GiB guard (only unrelated * 1000.0 timing). Implementing commits: a4c7235, 808d0d7.
Why it matters: Anyone using origin/main alone gets a false “implemented now” signal for those capabilities.
Blast radius: Onboarding, operators, and future agents trusting CURRENT_STATE without reading git range.
Confidence: High
Recommended fix: Merge/push this stack promptly; optionally add a ledger row in docs/review/DOC_PARITY_LEDGER.md documenting the temporary drift window (process, not code).
Salvage commit mixed production code + docs + generated binaries

Evidence: git show c43e483 --stat — same tree as incident-range work plus artifacts/2/.../embedding_matrix.npy and artifacts/3/.../embedding_matrix.npy (Git Bin). artifacts-checkpoint-6ed6785 == 6ed6785 with only those two files.
Why it matters: Irreversible repo bloat and unclear provenance if such a commit ever lands on a shared branch.
Blast radius: Clone/fetch size, CI, legal/compliance (binary lineage), confusing history.
Confidence: High
Recommended fix: Keep c43e483 / 6ed6785 off main; store large matrices outside git (blob store / dataset registry); add ignore rules or pre-commit for artifacts/** if policy is “never in repo.”
Medium
Public API doc not updated in this range

Evidence: git diff origin/main..HEAD -- docs/living/API_CURRENT.md is empty while src/study_query_llm/providers/factory.py and embedding surfaces changed (d3d67c0 and related tests).
Why it matters: .cursorrules / project norms call out API_CURRENT.md for API changes; drift raises integration risk for external callers.
Blast radius: Consumers of documented factory/API vs actual signatures.
Confidence: Medium
Recommended fix: After merge, diff factory.py / embedding entrypoints against API_CURRENT.md and update or add ledger entries explaining intentional omission.
feat(mcq) commit bundles multiple deliverables

Evidence: 8dcfbda touches config JSON, new notebook, three scripts (export / ingest / runner), MCQ experiment module.
Why it matters: Harder review, revert, and bisect; looks like multi-workstream residue.
Blast radius: MCQ sweep / OpenRouter ops only, but reviewer fatigue is real.
Confidence: Medium
Recommended fix: Optional interactive rebase split (notebook vs CLI scripts vs library tweak) before push if history purity matters; otherwise accept as one “MCQ sweep tooling” theme.
.gitignore gap for NumPy / artifacts/

Evidence: .gitignore has no artifacts/ or *.npy rule (while *.json is broadly ignored with exceptions for specific configs). Salvage did commit .npy under artifacts/.
Why it matters: No structural guardrail against repeating the mistake.
Blast radius: Future commits (not current push).
Confidence: High
Recommended fix: Policy + ignore / gitattributes / CI size/type check.
Low
Large new scripts without “thin” module extraction

Evidence: scripts/create_bank77_snapshot_and_embeddings.py ~1371 lines; scripts/probe_embedding_limits_live.py ~948 lines — mitigated by new tests in tests/test_scripts/.
Why it matters: Maintainability and review cost; not a security issue by itself.
Confidence: Medium
Recommended fix: Long-term factor into src/ libraries; not blocking merge.
Notebook added to repo

Evidence: notebooks/mcq_recent_1000_big_run_visualizer.ipynb (A in 8dcfbda). Quick grep found no obvious api_key / sk- patterns in the file.
Why it matters: Notebooks often later accumulate outputs with paths or tokens.
Confidence: Low (current snapshot looks clean)
Recommended fix: Pre-push manual open or nbstripout policy.
3) File Inventory Table (incident window cc49120..salvage-backup-c43e483)
Same path set as git diff --name-status cc49120..salvage-backup-c43e483 (51 paths). Classifications reference intent of the work, not the salvage mistake.

file	classification	likely_origin	risk	action
README.md	Docs/process update	Docs cleanup / salvage doc sync	Low	Keep; ensure README matches post-merge reality
artifacts/2/embedding_matrix/embedding_matrix.npy	Generated artifact/cache/temp	Local embedding run outputs (salvage)	High (if merged)	Do not merge; object storage; ignore locally
artifacts/3/embedding_matrix/embedding_matrix.npy	Generated artifact/cache/temp	Same	High (if merged)	Same
config/mcq_sweep_highschool_college_20q_3456_openrouter_6models.json	Intended feature work	MCQ OpenRouter sweep configuration	Low	Keep; verify no secrets (redacted audit: none seen)
docs/DATASET_SNAPSHOT_PROVENANCE.md	Docs/process update	Bank77 / snapshot lineage narrative	Low	Keep
docs/IMPLEMENTATION_PLAN.md	Docs/process update	Historical plan maintenance	Low	Keep
docs/README.md	Docs/process update	Docs index	Low	Keep
docs/USER_GUIDE.md	Docs/process update	User-facing onboarding	Low	Keep
docs/history/EXPERIMENT_ACTIVITY_SUMMARY.md	Docs/process update	Experiment history capture	Low	Keep
docs/history/EXPERIMENT_ATTEMPT_TABLE.md	Docs/process update	Same	Low	Keep
docs/history/README.md	Docs/process update	History index	Low	Keep
docs/plans/ARCHIVE_NOTICE.md	Docs/process update	Planning workflow archival	Low	Keep
docs/plans/MASTER_META_PLAN.md	Docs/process update	Meta-plan	Low	Keep
docs/plans/README.md	Docs/process update	Plans index	Low	Keep
docs/plans/STEP-01_master_bootstrap.md	Docs/process update	Planning milestone edits	Low	Keep
docs/runbooks/README.md	Docs/process update	Runbooks index	Low	Keep
notebooks/mcq_recent_1000_big_run_visualizer.ipynb	Intended feature work (analysis UI)	MCQ visualization workstream	Medium (output/token drift over time)	Keep; strip outputs if policy requires
scripts/create_bank77_snapshot_and_embeddings.py	Intended feature work	Bank77 bootstrap / embeddings pipeline	Medium (size/complexity)	Keep; monitor refactors
scripts/export_mcq_sweep_option_counts_db.py	Intended feature work	MCQ DB export tooling	Low	Keep
scripts/ingest_mcq_probe_json_to_sweep_db.py	Intended feature work	MCQ ingest fixes	Low	Keep
scripts/probe_embedding_limits_live.py	Intended feature work	Live embedding limit benchmarking	Medium (live API usage)	Keep; ensure CI doesn’t call live APIs without markers
scripts/run_mcq_sweep.py	Intended feature work	MCQ sweep runner tweaks	Low	Keep
scripts/run_openrouter_mcq_sweep_until_done.py	Intended feature work	OpenRouter automation	Medium (cost/rate limits)	Keep; ops runbook
scripts/validate_and_backfill_run_snapshots.py	Intended feature work	Provenance / snapshot validation	Medium (data mutation)	Keep; run in controlled env
src/study_query_llm/algorithms/__init__.py	Intended feature work	Provenance algorithm registration	Low	Keep
src/study_query_llm/algorithms/method_plugins.py	Intended feature work	Method plugins	Low	Keep
src/study_query_llm/config.py	Intended feature work	Provider/config wiring	Medium (misconfig risk)	Keep; no secrets in diff (not exhaustively scanned byte-by-byte)
src/study_query_llm/db/migrations/add_provenanced_runs_table.py	Intended feature work	DB migration helper	Medium–High (schema)	Keep; verify migration discipline in deploy
src/study_query_llm/experiments/ingestion.py	Intended feature work	Ingestion / provenance	Medium	Keep
src/study_query_llm/experiments/mcq_answer_position_probe.py	Intended feature work	MCQ experiment	Low	Keep
src/study_query_llm/experiments/mcq_run_persistence.py	Intended feature work	MCQ persistence	Medium	Keep
src/study_query_llm/providers/base.py	Intended feature work	Provider embedding metadata	Low	Keep
src/study_query_llm/providers/factory.py	Intended feature work	OpenRouter discovery / factory	Medium (behavior surface)	Keep; sync API_CURRENT.md if needed
src/study_query_llm/services/artifact_service.py	Intended feature work	Quota guardrail	Medium	Keep
src/study_query_llm/services/embeddings/helpers.py	Intended feature work	Bank77 / embedding helpers	Medium	Keep
src/study_query_llm/services/embeddings/service.py	Intended feature work	Embedding service	Medium	Keep
src/study_query_llm/services/jobs/job_reducer_service.py	Intended feature work	Job reduction hooks	Low	Keep
src/study_query_llm/services/model_registry.py	Intended feature work	Model registry cache	Medium	Keep
src/study_query_llm/services/provenance_service.py	Intended feature work	Provenance unification	Medium–High	Keep
src/study_query_llm/services/sweep_query_service.py	Intended feature work	Sweep reads / linkage	Medium	Keep
src/study_query_llm/storage/azure_blob.py	Intended feature work	Azure resilient reads / quota path	Medium	Keep
src/study_query_llm/storage/local.py	Intended feature work	Local storage (quota-related)	Low	Keep
tests/test_providers/test_factory.py	Intended feature work	Factory tests	Low	Keep
tests/test_scripts/test_create_bank77_snapshot_and_embeddings.py	Intended feature work	Script tests	Low	Keep
tests/test_scripts/test_probe_embedding_limits_live.py	Intended feature work	Probe tests	Low	Keep
tests/test_scripts/test_validate_and_backfill_run_snapshots.py	Intended feature work	Backfill tests	Low	Keep
tests/test_services/test_artifact_service_quota.py	Intended feature work	Quota tests	Low	Keep
tests/test_services/test_embedding_service.py	Intended feature work	Embedding service tests	Low	Keep
tests/test_services/test_model_registry.py	Intended feature work	Registry tests	Low	Keep
tests/test_services/test_provenance_delta_lineage.py	Intended feature work	Lineage tests	Low	Keep
4) Proposed Clean Commit Stack (old → new)
Current main already matches a clean logical order (no execution performed here):

cc49120 — origin/main (refactor(execution): reunify runs and analyses on one substrate)
a4c7235 — feat(bank77): add bootstrap flow and resilient artifact reads
33d7331 — feat(benchmark): add live embedding limit probe script
d3d67c0 — feat(openrouter): discover embedding models and runtime limits
3d9a388 — feat(provenance): unify execution linkage and snapshot lineage checks
808d0d7 — feat(storage): enforce artifact quota guardrail before writes
8dcfbda — feat(mcq): add OpenRouter sweep helpers and option-count exports
d3e095a — docs: archive planning workflow and add experiment history summaries
Optional refinement (only if rewriting history): split 8dcfbda into

feat(mcq): export and ingest tooling (scripts + small lib edits),
chore(notebooks): add MCQ visualizer (notebook only),
chore(config): refresh OpenRouter MCQ sweep preset (JSON only).
Must not enter main: anything matching salvage artifacts/**/**.npy checkpoints (c43e483, 6ed6785).

5) Pre-Push Gate Checklist
git status: clean (confirmed).
git log origin/main..HEAD --oneline: exactly 7 commits; subjects match intended themes.
git diff --name-status origin/main..HEAD: no artifacts/, no unexpected *.npy / large binaries.
Tests: pytest (full or CI-equivalent), with attention to new migration script and provenance tests.
Docs: reconcile docs/living/API_CURRENT.md with ProviderFactory / embedding changes; confirm CURRENT_STATE claims match post-merge tree (they already match HEAD; origin/main was the outlier).
Operational: confirm Azure quota env documented for deploy; OpenRouter scripts understood for cost.
Notebook: open mcq_recent_1000_big_run_visualizer.ipynb and confirm no embedded credentials in outputs.
Tags: decide whether salvage-backup-c43e483 / artifacts-checkpoint-6ed6785 stay local-only or should be documented for recovery (avoid pushing confusing tags without explanation).
6) Unknowns / Assumptions
Transcript-driven attribution 1 was not read here; salvage naming and commit messages are the only chat-linked evidence used.
Secret scan was heuristic (path-based + limited grep), not a full secrets scanner or binary scan.
CI behavior (whether live API tests run) was not verified from workflow files in this pass.
pre-script-cleanup tag exists locally; role and safety relative to this stack were not analyzed.
Assumption: origin/main at cc49120 is the correct merge base for your PR; if remote moved, re-run git fetch (outside this read-only audit) before push.
A) Repository state snapshot (required checklist item)
Item	Value
Branch	main @ d3e095a
Tracking	main...origin/main [ahead 7] (not behind)
Staged / unstaged / untracked	None (clean working tree)
Changed vs origin/main	49 files, +5956 −135 (git diff --stat origin/main..HEAD)
Incident window commits cc49120..salvage-backup-c43e483	a4c7235, 33d7331, c43e483 (git log --oneline)
Post-cleanup stack cc49120..HEAD	a4c7235 → 33d7331 → d3d67c0 → 3d9a388 → 808d0d7 → 8dcfbda → d3e095a
Safety tags	salvage-backup-c43e483 → c43e483 (not on main); artifacts-checkpoint-6ed6785 → 6ed6785 (not on main)
