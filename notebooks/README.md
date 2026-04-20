# Notebooks Index

Status: living  
Owner: analytics-maintainers  
Last reviewed: 2026-04-19

This directory contains active working notebooks plus archived snapshots used by
history docs.

## Local GPU 300 Sweep Notebook Policy

- Canonical working notebook: `notebooks/local_gpu_300_sweep.ipynb`
- Historical snapshot retained for history docs: `notebooks/archive/local_gpu_300_sweep.ipynb`
- Current diff between the two files is metadata-only (`kernelspec.display_name`
  and `language_info.version`), not analysis logic.

When updating the local GPU 300 sweep workflow, apply edits to the canonical
working notebook first. Keep the archive copy unchanged unless intentionally
capturing a new historical snapshot and updating linked history docs.
