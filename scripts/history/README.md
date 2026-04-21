# Scripts History Lane

This lane stores experiment and chronology scripts that are kept for reproducibility
but are not the primary operational entrypoints.

## Current Sub-Lanes

- `scripts/history/analysis/`: analysis/plotting scripts for prior experiment batches
- `scripts/history/experiments/`: older experimental sweep drivers and harnesses
- `scripts/history/sweep_recovery/`: one-off clustering/sweep incident tooling
  (`archive_pre_fix_runs.py`, `label_pre_fix_runs.py`)

## Compatibility

Several moved scripts still have root-level wrappers under `scripts/` that print a
deprecation notice and forward execution (often via `scripts/deprecated/` first, then
into this lane). Update any saved commands to use the `scripts/history/...` path
directly when practical.
