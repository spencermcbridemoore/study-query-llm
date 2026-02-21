# Custom Sweep: Full Categories, Extended K Range

## Configuration

This sweep tests clustering performance with:
- **ALL available categories** (not subsampled)
- **Extended k range** from 1 to 20 (vs default 2-10)
- **Variable entry counts** to test scalability

### Parameters

| Parameter | Value |
|-----------|-------|
| **Datasets** | dbpedia, yahoo_answers |
| **label_max** | **FIXED** at maximum per dataset |
| | - dbpedia: 14 categories (all) |
| | - yahoo_answers: 10 categories (all) |
| **entry_max** | 100, 200, 300, 400, 500 |
| **k range** | 1 to 20 (extended) |
| **Summarizers** | None, gpt-4o-mini, gpt-4o, gpt-5-chat |
| **n_restarts** | 20 (for stability metrics) |

### Total Runs

**40 runs** = 5 entry_max × 2 datasets × 4 summarizers

## Rationale

### Why Full Categories?

Testing with ALL available categories (14 for dbpedia, 10 for yahoo_answers) provides:
1. **Real-world scenarios**: Actual clustering tasks use all categories
2. **Higher complexity**: More categories = harder clustering problem
3. **Better evaluation**: Can test if LLMs help with many categories

### Why Extended K Range (1-20)?

Default k range (2-10) may miss interesting patterns:
1. **k=1**: Baseline (all one cluster)
2. **k>10**: Test if clustering continues to improve or plateaus
3. **Overfitting detection**: Performance at k=15-20 vs true category count
4. **Elbow analysis**: Better data for finding optimal k

### Why Variable Entry Counts?

Testing 100-500 entries shows:
1. **Data efficiency**: How much data do LLMs need to help?
2. **Scalability**: Do improvements hold with more data?
3. **Statistical power**: More data = more reliable metrics

## Research Questions

This sweep can answer:

1. **Do LLMs help with many categories?**
   - Compare summarizer performance at k=10 and k=14

2. **What's the optimal k?**
   - Extended range allows better elbow/silhouette analysis

3. **How does data size affect LLM benefit?**
   - Compare entry_max=100 vs 500 performance differences

4. **Do results generalize across datasets?**
   - dbpedia (encyclopedic, 14 cats) vs yahoo_answers (Q&A, 10 cats)

## Expected Runtime

- ~40 runs total
- ~10-20 minutes per run (with n_restarts=20, k_max=20)
- **Total: 7-13 hours**

## Output Files

Pickle files saved to `experimental_results/`:
```
experimental_sweep_entry{100-500}_dbpedia_labelmax14_{summarizer}_{timestamp}.pkl
experimental_sweep_entry{100-500}_yahoo_answers_labelmax10_{summarizer}_{timestamp}.pkl
```

Each file contains:
- Clustering results for k=1 to k=20
- Stability metrics across 20 restarts
- Ground truth labels (for ARI/NMI evaluation)
- Metadata (config, categories, etc.)

## Scripts

- **Main sweep**: `scripts/run_custom_full_categories_sweep.py`
- **Analysis**: Use `notebooks/sweep_explorer.ipynb` with filters:
  - `label_max=[10, 14]` (to isolate this sweep)
  - Compare across entry_max
  - Extended k range plots

## Status

**Running**: Started 2026-02-15 23:05:10  
**Progress**: Check `custom_sweep_output.log` for real-time status

---

*Note: This sweep uses the CORRECTED experimental design (embeddings computed once, shared across summarizers)*
