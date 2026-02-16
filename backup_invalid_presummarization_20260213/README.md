# Invalid Experimental Data - Pre-Summarization Bug

**Archive Date**: February 13-15, 2026  
**Files Archived**: 558 pickle files  
**Reason**: Critical experimental design flaw in `run_experimental_sweep.py`

## The Bug

The sweep script was **pre-summarizing all texts** with each LLM before embedding them, causing each summarizer to operate in a completely different embedding space.

### What Was Happening (INCORRECT)

```
For each summarizer:
  1. None      → embed(original_texts)           → cluster
  2. gpt-4o    → summarize all → embed(gpt4o_summaries)    → cluster
  3. gpt-4o-mini → summarize all → embed(mini_summaries)   → cluster
  4. gpt-5-chat  → summarize all → embed(gpt5_summaries)   → cluster
```

**Problem**: Each summarizer operated in a different embedding space. This tested "which LLM produces summaries whose embeddings happen to cluster well" rather than "which LLM makes better centroids during clustering."

### Code Evidence

From `run_experimental_sweep.py` (lines 847-875, before fix):

```python
# BUG: Pre-summarize all texts before embedding (WRONG!)
if llm_deployment is not None:
    print(f"        Summarizing {len(sampled_texts)} texts with {summarizer_name}...")
    paraphraser = create_paraphraser_for_llm(llm_deployment, db)
    
    summarized_texts = []
    for i, text in enumerate(sampled_texts):
        summary = paraphraser([text])
        summarized_texts.append(summary)
    
    texts_to_embed = summarized_texts  # Different per summarizer!
else:
    texts_to_embed = sampled_texts

# Fetch embeddings (different space per summarizer!)
embeddings = await fetch_embeddings_async(texts_to_embed, EMBEDDING_DEPLOYMENT, db)
```

## The Fix

All summarizers now use the **same embeddings** from original texts. LLMs only influence **in-loop centroid updates**.

### What Should Happen (CORRECT)

```
1. Embed original texts ONCE (shared across all summarizers)
2. For each summarizer:
   - Pass same embeddings to clustering
   - LLM influences ONLY in-loop centroid updates:
     * Select representative texts from each cluster
     * Summarize them with LLM
     * Re-embed summaries to compute new centroid positions
3. Fair comparison: same starting space, only centroid quality differs
```

### Fixed Code

From `run_experimental_sweep.py` (after fix, commit `ce8a67d`):

```python
# CORRECT: Embed original texts ONCE (shared across all summarizers)
print(f"      Fetching embeddings for {len(sampled_texts)} original texts...")
shared_embeddings = await fetch_embeddings_async(sampled_texts, EMBEDDING_DEPLOYMENT, db)
print(f"      [OK] Fetched {len(shared_embeddings)} embeddings (shared across all summarizers)")

# Run sweep for each LLM summarizer
# NOTE: All summarizers use the SAME embeddings from original texts
# LLM influence is ONLY during in-loop centroid updates via paraphraser
for llm_deployment in LLM_SUMMARIZERS:
    embeddings = shared_embeddings  # Same for all!
    result = await run_single_sweep(sampled_texts, embeddings, llm_deployment, db)
```

And in `run_single_sweep`:

```python
# Pass both paraphraser and embedder to enable in-loop LLM centroid updates
run_sweep(
    texts, embeddings, SWEEP_CONFIG,
    paraphraser=paraphraser,
    embedder=embedder_sync if paraphraser else None  # Re-embed summaries for centroids only
)
```

## Impact on Results

All experimental results in this backup folder are **scientifically invalid**:

1. **Unfair comparison**: Each summarizer operated on different data
2. **Confounded metrics**: High silhouette scores could be artifacts of pre-summarization, not better centroids
3. **Wrong question**: Tested "which summaries embed well" not "which LLM makes better centroids"

### Expected Changes After Fix

With the correct design:
- All LLMs start from same embedding space (fair comparison)
- Previous "high silhouette at k=2 for gpt-5-chat" may disappear (was an artifact)
- Results now measure only centroid quality, not embedding artifacts

## Files in This Archive

- **Total**: 558 pickle files
- **Date range**: 2026-02-07 to 2026-02-14
- **Datasets**: 
  - 20newsgroups_6cat (6 categories)
  - 20newsgroups_10cat (10 categories)
  - dbpedia (14 categories)
  - yahoo_answers (10 categories)
  - news_category (many categories)
- **Configurations**:
  - `entry_max`: 100, 200
  - `label_max`: 1, 2, 3, 4, 5, 6
  - `summarizers`: None, gpt-4o, gpt-4o-mini, gpt-5-chat

## Git Commits

- **Bug introduced**: Earlier in development
- **Bug discovered**: 2026-02-13
- **Fix commits**:
  - `f82135c` - Initial fix (moved embedding outside loop, removed pre-summarization)
  - `ce8a67d` - Corrected fix (restored embedder for in-loop centroid updates)
  - `7c4eaa5` - Pre-fix commit (saved work before fixing)

## Related Documentation

See `BUG_FIX_20260213.md` in the project root for full technical details.

## Next Steps

After archiving this data:
1. ✅ Archive complete (558 files moved)
2. ⬜ Run limited test sweep to verify fix
3. ⬜ Validate embeddings are identical across summarizers
4. ⬜ Run full sweep with corrected experimental design
5. ⬜ Analyze new results and compare to these invalid results (for learning)

---

**DO NOT USE THESE RESULTS FOR SCIENTIFIC ANALYSIS**

These files are kept for historical/debugging purposes only.
