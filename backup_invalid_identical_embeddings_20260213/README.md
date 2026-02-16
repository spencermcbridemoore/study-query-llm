# Invalid Experimental Data - Identical Embeddings Bug

**Date:** 2026-02-13
**Issue:** All pickle files in this directory have IDENTICAL embeddings across all summarizers

## Problem Description
Due to a bug in `scripts/run_experimental_sweep.py`, embeddings were computed once from original texts and then reused for all summarizers (None, gpt-4o, gpt-4o-mini, gpt-5-chat). This resulted in identical clustering results regardless of summarization strategy.

## Root Cause
The embedding computation was outside the per-summarizer loop:
```python
# BROKEN CODE (before fix):
embeddings = await fetch_embeddings_async(sampled_texts, EMBEDDING_DEPLOYMENT, db)
for llm_deployment in LLM_SUMMARIZERS:
    result = await run_single_sweep(sampled_texts, embeddings, llm_deployment, db)
    # All summarizers use the SAME embeddings!
```

## Fix Applied
Commit: 7c6e07e - "Fix bug: compute embeddings per-summarizer, not shared"
- Moved embedding computation inside per-summarizer loop
- Added text summarization before embedding for LLM summarizers
- Each summarizer now gets embeddings from its own (possibly summarized) texts

## Validation
Validation script confirmed the fix works (commit: 685d366):
- None vs gpt-4o-mini: Max diff = 0.072498 (embeddings are different!)
- None vs gpt-4o: Max diff = 0.072498
- gpt-4o-mini vs gpt-4o: Max diff = 0.057616

## File Count
362 pickle files moved to this backup directory

## What to do with this data
**DO NOT USE FOR ANALYSIS** - All pairwise comparisons will show zero difference because embeddings are identical. This data only shows the effect of different k values and random restarts, not the effect of summarization.

## New sweep runs
After the fix, new sweeps will produce valid data where different summarizers create different embeddings and thus different clustering results.
