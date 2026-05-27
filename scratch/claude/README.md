# scratch/claude/

Working surface for Claude (AI assistant) sessions — investigation notes, intermediate findings, query-result dumps, working outlines.

Used in read-only investigation sessions where Claude needs a persistent place to write notes without burning conversation context. Not part of the framework (`src/study_query_llm/`) or operator scripts (`scripts/`).

Everything in this directory is gitignored except this README (matches the `scratch/local/` convention). If a note here turns out worth keeping, move it elsewhere (e.g. `docs/`) and commit it explicitly.

Conventions:
- Prefix filenames with date or topic: `2026-05-27_lease_investigation.md`.
- Don't put secrets or production data here — gitignore protects against accidental commit, not accidental copy/paste into a tracked file.
