# Firm up the scripts/ vs src/ orchestration boundary

## Goal

Anchor an unambiguous architectural rule for where new orchestration logic belongs — Tier A under `src/study_query_llm/`, not Tier B under `scripts/` — using the tier hierarchy from the Codex de-confusion map as the operating vocabulary. The Slice 2 Wave 1 bundled-clustering work landed correctly in Tier A, but the ambient bloat and naming residue in `scripts/` made plugging it into Tier B plausible. Anchor the boundary now so the next attempt has guardrails — and do it honestly, naming the known transitional violations rather than pretending the rule is universally true.

## Context

**Tier hierarchy (operating vocabulary):**
- **Tier A — canonical runtime**: `src/study_query_llm/experiments/`, `src/study_query_llm/services/jobs/`, `src/study_query_llm/services/sweep_request_service.py`, `src/study_query_llm/experiments/sweep_request_types.py`, `src/study_query_llm/pipeline/`. Behavioral truth lives here.
- **Tier B — compatibility entrypoints**: root `scripts/run_*.py` wrappers. Stable operator command surface; names do not define architecture.
- **Tier C — historical/reproducibility**: `scripts/history/**`, archived notebooks.
- **Tier D — living docs**: `docs/living/**` and runbooks as policy/intent mirrors.

**Rule:** behavioral truth comes from Tier A. Legacy-flavored campaign names in Tier B (`local_300_2datasets`, `pca_kllmeans_sweep`, etc.) do NOT define architecture.

**Known violations of the boundary (state honestly in the anchor docs, do not pretend they don't exist):**
- [src/study_query_llm/services/jobs/runtime_supervisors.py:259](src/study_query_llm/services/jobs/runtime_supervisors.py:259) and [:503](src/study_query_llm/services/jobs/runtime_supervisors.py:503) subprocess-launch `scripts/run_local_300_2datasets_worker.py` — a Tier A → Tier B execution path that contradicts the rule.
- `scripts/run_local_300_2datasets_worker.py` is **1044 lines** carrying canonical sweep worker logic. It is technically Tier B by location, but functionally Tier A. A "fat compatibility wrapper" pending inlining.
- Campaign-naming residue inside Tier A files: `OUT_PREFIX = "local_300_2d_14e_4s"` and hardcoded `dbpedia + estela` dataset constants in `sweep_worker_main.py`; `LLM_SUMMARIZERS` constants and orchestration-flavored logic in `src/study_query_llm/providers/managers/ollama.py`.
- Doc divergence: [docs/SWEEP_MIGRATION_RUNBOOK.md:172](docs/SWEEP_MIGRATION_RUNBOOK.md:172) still describes standalone in pre-orchestration `sweep_run_claims` terms while [docs/living/CURRENT_STATE.md](docs/living/CURRENT_STATE.md) says standalone is now an orchestration profile.

**`summarizer` is current contract, not legacy.** [src/study_query_llm/experiments/sweep_request_types.py:215](src/study_query_llm/experiments/sweep_request_types.py:215) — `summarizers` is part of `expand_parameter_axes`; every clustering `RunTarget` has a `summarizer` field. Persisted keys/fields stay as `summarizer`. Doc language can describe it as "text transformation variant (stored as `summarizer`)" but the storage contract is unchanged. `None` is a first-class variant, not a special legacy marker. Scripts named for summarizers are NOT automatically legacy.

**Existing AI-binding doc surfaces (where the anchor goes):**
- [AGENTS.md](AGENTS.md) — root-level AI guidance; already lists the 5-stage pipeline contract on line 24; AI tools read this first.
- [.cursorrules](.cursorrules) — Cursor coding conventions.
- [.cursor/rules/living-docs-only.mdc](.cursor/rules/living-docs-only.mdc) — binding doc<->code map (always-on per AGENTS.md).
- [scripts/README.md](scripts/README.md) — local scripts/ convention; acknowledges mid-migration but doesn't draw the boundary explicitly.

**Prior decision:** boundary first, cleanup second. This plan does NOT inline the canonical worker into src/, retire campaign names, restructure modules, or do the broader scripts/ cleanup. Those are deferred to phase 2. This plan ONLY anchors the rule and acknowledges current violations honestly.

## Approach

Adopt the tier vocabulary in the doc surfaces AI tools and humans actually read. Lead with AGENTS.md (read first by AI tools), reinforce in .cursorrules (Cursor surface), ground in scripts/README.md (local context). Add per-file headers to Tier B wrappers so the rule is visible at the point of temptation. Each anchor includes a "Known violations" subsection — a doc that pretends the rule is universally true would be worse than the current ambiguity, because future readers would trust it and be misled. Tradeoff: still doc-only; depends on AI tools actually reading the docs. A lint test would be stronger but adds maintenance — deferred as optional follow-up.

## Steps

- [ ] Inventory the actual Tier B wrapper set. Grep `scripts/` for files that either (a) subprocess-launch into Tier A modules or (b) wrap CLI commands that delegate to `src/study_query_llm/services/jobs/` or `src/study_query_llm/experiments/sweep_worker_main`. Capture exact paths and current LOC. Specifically flag `run_local_300_2datasets_worker.py` (1044 lines) as a fat wrapper to be called out by name in headers.
- [ ] Edit [AGENTS.md](AGENTS.md). Add a new section "Code layer boundary" after the "Essential Constraints" section, containing:
  - Tier hierarchy summary (A/B/C/D as listed in Context above).
  - Boundary rule: new orchestration/job logic → `src/study_query_llm/services/jobs/` or `src/study_query_llm/experiments/`; new pipeline-stage methods → `src/study_query_llm/pipeline/<stage>/`; `scripts/` only for thin wrappers, ops utilities (DB ops, checks, migration helpers), or historical reproduction.
  - Worked counter-example: "Adding a new clustering method? Register under `src/study_query_llm/pipeline/clustering/` (see Slice 2 Wave 1). Do NOT add a new `scripts/run_*.py` for it."
  - Vocabulary lock: `orchestration_job` (schedulable control-plane unit), `worker runtime` (process consuming orchestration work), `standalone mode` (orchestration profile, not an alternate legacy control plane), `summarizer` (text transformation variant — current schema term, `None` is first-class), `compatibility wrapper` (Tier B script for command/path stability, not business logic).
  - Known violations subsection: list the runtime_supervisors → worker subprocess path with line refs, the 1044-line worker, the campaign naming residue in Tier A, the SWEEP_MIGRATION_RUNBOOK divergence. Mark each as "transitional; tracked under phase-2 cleanup."
  - Cross-link to scripts/README.md for the local convention.
- [ ] Edit [.cursorrules](.cursorrules). Add a 3-5 line block referencing the tier vocabulary, the boundary rule's one-line summary, and a pointer to AGENTS.md for the full version (no duplication).
- [ ] Edit [scripts/README.md](scripts/README.md). Add a new top section "What belongs here":
  - Tier B definition (thin wrappers / ops utilities / historical reproduction).
  - Decision tree: "Adding new code? If it's orchestration / a pipeline-stage method / a job runner → `src/`. If it's a one-shot DB op or ad-hoc utility → `scripts/`."
  - Concrete examples on each side from current contents: `register_clustering_methods.py` belongs in scripts/ (DB setup utility); `pipeline/clustering/kmeans_fixed_k_runner.py` belongs in src/ (Tier A method runner).
  - Brief note that `run_local_300_2datasets_worker.py` is a known fat wrapper carrying canonical worker logic, queued for inlining in phase 2 — do NOT use it as a precedent for new orchestration logic in scripts/.
  - Pointer to AGENTS.md as source of truth.
- [ ] Add a top-of-file header docstring to each Tier B wrapper identified in step 1. Header content (4-6 lines):
  - "Tier B compatibility wrapper. Real orchestration lives in `src/study_query_llm/services/jobs/` and `src/study_query_llm/experiments/`."
  - "Do NOT add orchestration logic here. New methods/jobs/runners belong in src/."
  - For `run_local_300_2datasets_worker.py` specifically: an extra line — "Fat wrapper carrying canonical worker logic; pending inlining into src/ in phase 2 cleanup. Do not use as a template for new code."
  - Pointer to AGENTS.md.
- [ ] Doc reconciliation. Edit [docs/SWEEP_MIGRATION_RUNBOOK.md](docs/SWEEP_MIGRATION_RUNBOOK.md) section 8 ("Job runner architecture and langgraph_run") to front-load a one-line pointer: "**Current behavior:** standalone is now an orchestration profile (see [docs/living/CURRENT_STATE.md](docs/living/CURRENT_STATE.md)); the description below reflects pre-orchestration design and is retained for historical context." Do NOT rewrite the section — minimum-disruption pointer.
- [ ] Verify the changes locally (see Validation).
- [ ] Commit as a single change: `docs(boundary): anchor scripts/ vs src/ tier boundary with known violations`.

## Validation

- Open [AGENTS.md](AGENTS.md) and confirm a new reader can answer two questions from that file alone: (1) "Where does new clustering-method code go?" and (2) "Is `summarizer` legacy or current contract?"
- Open one of the wrapper scripts (e.g. `scripts/run_langgraph_job_worker.py`) and confirm the Tier B header is visible without scrolling.
- Open [scripts/README.md](scripts/README.md) and confirm "What belongs here" answers the decision-tree question with concrete examples.
- Sanity check the audit greps. The runtime_supervisors → worker subprocess coupling MUST still exist (we are documenting it, not removing it). `grep -rn "scripts/" src/` should still surface the existing soft mentions in docstrings PLUS the 2 `runtime_supervisors.py` subprocess paths — no new couplings introduced.
- Optional smoke test: in a fresh Cursor / Claude Code session, ask "where should I add a new bundled clustering method?" and confirm the answer points to `src/study_query_llm/pipeline/clustering/`, not `scripts/`. Also ask "is summarizer legacy?" — answer should be "no, current contract."

## Open questions

- Should the tier vocabulary also live in [.cursor/rules/living-docs-only.mdc](.cursor/rules/living-docs-only.mdc) (always-on per AGENTS.md, stronger AI-tool guarantee) or is AGENTS.md sufficient? The .mdc file currently scopes to docs<->code mappings, not folder-layout rules — adding it would be a charter expansion.
- Should per-file headers also be added to historical scripts under `scripts/history/**`, or only to root-level Tier B wrappers? Probably only root, since history is already lane-segregated, but worth confirming.
- For the SWEEP_MIGRATION_RUNBOOK reconciliation: front-loaded pointer only, or also tag the section title with `[Historical context]`? The pointer is minimum-disruption; the tag is clearer at a glance.
- Should DF-011 (`cosine_kllmeans_no_pca` runtime mismatch) get a brief reference in the AGENTS.md Known violations subsection as another worked example, or kept fully in `docs/DESIGN_FLAWS.md`?

## Out of scope

- **Inlining `run_local_300_2datasets_worker.py` into src/** — the 1044-line worker file refactor. Deferred to phase 2 cleanup. The boundary doc names this as a known violation rather than fixing it now.
- **Renaming campaign-flavored files and constants** (`run_local_300_2datasets_*`, `OUT_PREFIX = "local_300_2d_14e_4s"`, etc.) — naming clarity pass, deferred to phase 2 (Codex map section 6, "Naming clarity pass").
- **Scripts/ cleanup** — 78 live root scripts, 21 wrapper stubs, empty `scripts/living/` migration. Deferred per "boundary first, cleanup second"; the boundary rule will become the filter for that cleanup.
- **Cleaning `LLM_SUMMARIZERS` constants out of `src/study_query_llm/providers/managers/ollama.py`** — orchestration-flavored logic in a provider module; legitimate but separate concern.
- **Resolving DF-011** — separate fix tracked in `docs/DESIGN_FLAWS.md`.
- **Reorganizing `src/services/jobs/` vs `src/experiments/`** — boundary describes current reality; restructuring is separate.
- **Migrating storage contract from `summarizer` to `transform_variant`** — Codex's medium-term suggestion (section 4). Schema change, requires deliberate migration. Out of boundary scope.
- **Adding a lint/test that enforces the boundary** (e.g. fails if new files in `scripts/` import certain orchestration-internal modules, or if `runtime_supervisors.py` adds new subprocess-to-scripts paths). Stronger insurance, more maintenance — deferred as optional follow-up after boundary doc lands.
- **Rewording the soft-mention `code_ref="scripts/parse_quiz.py"` docstring examples in `MethodService` and `models_v2.py`** — benign documentation, not couplings.
