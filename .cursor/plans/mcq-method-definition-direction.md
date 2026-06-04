# MCQ experiments — taxonomy, current (legacy) state, deferred method-definition direction

Status: direction note (intent capture only). No code is committed against this;
the redesign below is intentionally deferred. Created 2026-06-03.

This lives in the plans lane on purpose: it stays out of the living-docs set until
(unless) the redesign is actually built, so it adds no governance surface.

## Why this exists

The MCQ code grew two distinct experiment shapes that are easy to conflate. This
note records (1) the taxonomy, (2) the current legacy state and the explicit
decision to leave it as-is, and (3) the deferred direction so the design thinking
is not lost.

## Taxonomy (two families)

### Family 1 — constructor / authoring-position bias (its own narrow class)
- Code: `src/study_query_llm/experiments/mcq_answer_position_probe.py`
  (+ scripts `run_mcq_answer_position_probe.py`, `run_mcq_sweep.py`,
  `src/study_query_llm/experiments/sweep_mcq_standalone.py`).
- The model GENERATES a test + answer key; sampled N times at temperature ~0.7;
  answer-key letters parsed from generated text; studies the DISTRIBUTION of
  correct-answer positions (pooled + per-sample, chi-square vs uniform).
- Question: when AUTHORING a test, does the model bias correct answers toward
  certain positions? Shares no code with Family 2.

### Family 2 — fixed-structure answering (± permutation); output mode = sampling | logprobs
- Question/option structure is a fixed dataset; options are (optionally) permuted
  by a scheme; the OUTPUT MODE is a knob over the same experimental object.
- Logprobs arm (EXISTS): `src/study_query_llm/services/method_runners/mcq_logprob_basic.py`
  (`inference.mcq_logprob.basic@0.1`), orchestrated by
  `scripts/run_mcq_logprob_experiment.py`. One deterministic call per
  (question, permutation): temperature=0, max_tokens=1, logprobs=True,
  top_logprobs=K; read first-token top-K distribution; argmax letter.
- Sampling arm (DOES NOT EXIST yet): answer the same fixed/permuted test by
  SAMPLING realized letters (temp>0, parse letter, N draws). The generic
  `perturbation_then_inference.basic` runner is NOT this (free-text variants, no
  MCQ structure, no permutation, no letter parsing).

## Current (legacy) state — kept as-is

Decisions (2026-06-03):
- The redesign below is DEFERRED. No MCQ code changes now.
- Existing MCQ scripts/runners stay running as-is. Legacy MCQ data stays in legacy
  terms (no backfill / migration).
- The Family-1 answer-position chain is NOT retired: it is a DISTINCT hypothesis
  (authoring bias), not superseded by the Family-2 logprob/answering work. This
  resolves the 3.3 open question in `.cursor/plans/post-audit-hardening-v2.md`.

Documented rough edges (not fixed now):
- The Family-2 logprobs runner is monolithic: dataset load + permutation +
  per-item fan-out + scoring all inside one runner.
- It does NOT persist per-call `raw_calls`: it builds
  `InferenceService(repository=None)` and bakes outputs into one parquet
  `CallArtifact` (`mcq_logprob_rows_parquet`) under one `ProvenancedRun`. The
  per-question outputs are parquet payload, not relationally-addressable rows.
- top_logprobs: the runner DEFAULTS to 20 (API ceiling, `le=20`) but the actual
  experiment pins it to 5 (`run_mcq_logprob_experiment.INFERENCE_TOP_LOGPROBS`;
  "universal cap across providers, Alibaba max-5"). A separate, unrelated 20/5 in
  `probe_ceiling_for_model` / `OPENAI_PROBE_CEILING_MODELS` is the CONCURRENCY
  ceiling, not logprob depth — easy to misread.
- The permutation schemes (`full_120`, `latin_squares_25`,
  `single_latin_square_5`) live inside `mcq_logprob_basic.py` and are shared with
  nothing.

## Deferred direction (intent — not a commitment)

Fold MCQ into the method-definition lane as composed methods, so "sampling vs
logprobs" becomes an ANALYSIS choice over a shared substrate rather than two
runners:

1. Permutation method (transform; no LLM). Structured MCQ items -> a
   hash-identified permuted-prompt snapshot. parameters_schema =
   {permutation_strategy in [full_120, latin_squares_25, single_latin_square_5,
   five_rotary, random_k_of_120, ...], seed, k}. Deterministic schemes get a
   stable identity; "random k of 120" MUST pin its seed (pseudo_deterministic).
   Precedent: `csv_parse.basic` (imported CSV -> parquet).
2. Inference method WITH a raw_call result. A permuted prompt -> the LLM call
   captured as a `raw_call` (response_json = logprobs AND realized token), linked
   into the run group via `GroupMember`. Only real change from today: the runner
   USES the `repository` it is already handed (stop passing `repository=None`).
   One run per (model x scheme) emitting MANY raw_calls (call grain, not run grain).
3. Scoring method (derive accuracy over the raw_calls): attempts, errors,
   abstentions, correct -> rates; re-scorable without re-querying. This is the 3.2
   accuracy contract.
4. Compose via `recipe_json` (canonical_recipe_hash -> run fingerprint), same
   pattern as the clustering composites.

Payoff: raw_calls become the immutable substrate; the permutation snapshot is
reusable across models; the same raw_calls can feed BOTH a logprob analysis and a
sampling/accuracy analysis (the "constant calls, many pipelines" pattern). The v2
schema already supports this end-to-end (RawCall<->Group many-to-many via
GroupMember; typed GroupLink DAG; AnalysisResult/ProvenancedRun keyed but not
exclusive on source) — no schema change required.

Decisions to settle if/when this is built:
- Persistence grain (raw_calls vs blob vs both) and the data-volume cost on the
  live canonical DB (full_120 x 45 questions x 14 models ~ 75k calls).
- Permutation-snapshot identity hash inputs (dataset_version, scheme, seed,
  prompt_template_version, label-set).
- Whether the permuted-prompt set is modeled as a dataset_snapshot-like input the
  inference method depends_on.

## Pointers
- Plan: `.cursor/plans/post-audit-hardening-v2.md` (items 3.2, 3.3).
- Method lane: `docs/living/METHOD_RECIPES.md`; `docs/STANDING_ORDERS.md`
  (Method Definitions and Provenance).
