# Scripts History Lane

This lane stores experiment and chronology scripts that are kept for reproducibility
but are not the primary operational entrypoints.

## Current Sub-Lanes

- `scripts/history/analysis/`: analysis/plotting scripts for prior experiment batches
- `scripts/history/experiments/`: older experimental sweep drivers and harnesses

## Compatibility

Several moved scripts still have root-level wrappers under `scripts/` that print a
deprecation notice and forward execution to this lane. Update any saved commands to
use the `scripts/history/...` path directly.
