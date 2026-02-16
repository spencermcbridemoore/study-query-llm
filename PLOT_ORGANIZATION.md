# Plot Organization - February 16, 2026

## Folder Structure

Organized `experimental_results/plots/` into three subfolders:

```
experimental_results/plots/
├── values/                           # Raw metric value plots (NEW, EMPTY)
├── pairwise/                         # Pairwise comparison plots (NEW, EMPTY)
└── invalid_presummarization/         # Archived invalid plots (25 files)
    └── README.md
```

## What Was Done

1. **Created organized structure** for future plots:
   - `values/` - For raw metric plots (silhouette, ARI, etc. vs k)
   - `pairwise/` - For difference plots (Summarizer_A - Summarizer_B)

2. **Archived 25 invalid plots** generated from pre-summarization bug data:
   - All existing plots moved to `invalid_presummarization/`
   - Added comprehensive README explaining why they're invalid

3. **Added documentation** in each folder:
   - `values/README.md` - Explains raw value plot structure
   - `pairwise/README.md` - Explains comparison plot structure and error propagation
   - `invalid_presummarization/README.md` - Documents the bug

## Plot Naming Conventions

### Values Folder
```
{metric}_{dataset}_subplot_grid.png
```
Example: `silhouette_mean_dbpedia_subplot_grid.png`

### Pairwise Folder
```
{metric}_{dataset}_pairwise_subplot_grid.png
```
Example: `silhouette_mean_dbpedia_pairwise_subplot_grid.png`

## Grid Structure

Both plot types use subplot grids:
- **Columns**: `label_max` values (category counts)
- **Rows**: 
  - Values plots: Summarizers (None, gpt-4o-mini, gpt-4o, gpt-5-chat)
  - Pairwise plots: Comparisons (None-gpt4o, None-mini, None-gpt5, mini-gpt4o, gpt5-gpt4o)

## Error Representation

- **Values plots**: Shaded ±1σ regions + error bars
- **Pairwise plots**: Propagated errors (σ_diff = √(σ_A² + σ_B²)) with bars

## Next Steps

To generate new (valid) plots using `notebooks/sweep_explorer.ipynb`:

1. Run the batch plot generation cells
2. Modify the save path to use:
   - `../experimental_results/plots/values/` for raw plots
   - `../experimental_results/plots/pairwise/` for comparison plots

## Notes

- Plot images are **not version controlled** (in `.gitignore`)
- Only README documentation is tracked in git
- Invalid plots kept for reference/debugging purposes
