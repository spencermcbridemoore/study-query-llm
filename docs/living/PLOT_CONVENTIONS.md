# Plot Conventions (Current)

Status: living  
Owner: experimentation-maintainers  
Last reviewed: 2026-04-06

## Active Conventions

- Values plots:
  - Filename: `{metric}_{dataset}_subplot_grid.png`
  - Example: `silhouette_mean_dbpedia_subplot_grid.png`
- Pairwise plots:
  - Filename: `{metric}_{dataset}_pairwise_subplot_grid.png`
  - Example: `silhouette_mean_dbpedia_pairwise_subplot_grid.png`

## Grid Shape

- Columns: `label_max` values.
- Rows:
  - values plots: summarizer choice.
  - pairwise plots: summarizer deltas.

## Error Representation

- Values: uncertainty intervals and error bars.
- Pairwise: propagated error bars (`sigma_diff = sqrt(sigma_A^2 + sigma_B^2)`).

## Historical Context

Detailed chronology for the 2026-02 plot folder reorganization remains in `docs/PLOT_ORGANIZATION.md`.
