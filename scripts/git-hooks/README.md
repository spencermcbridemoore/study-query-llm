# Git Hooks (Living-Docs-Only Governance)

Opt-in Git hooks that enforce the living-docs-only rule
(`.cursor/rules/living-docs-only.mdc`) locally, split by git timing.

## Install (per-clone, opt-in)

```sh
git config core.hooksPath scripts/git-hooks
```

Local to the clone (edits `.git/config`, not the repo); no global settings change.

## Uninstall

```sh
git config --unset core.hooksPath
```

## What's here

`pre-commit` (blocking; fast static/staged checks only -- the full suite runs in CI):

- `scripts/check_staged_restricted_paths.py` -- blocks the commit when staged changes
  touch tombstoned restricted paths (`docs/history/**`, `docs/deprecated/**`,
  `docs/plans/**`, `docs/experiments/**`, `scripts/history/**`, `scripts/deprecated/**`,
  plus the named legacy files in `scripts/internal/living_docs_governance.py`).
- `scripts/verify_script_path_references.py` -- fails if a living/CI doc references a
  missing `scripts/*.py`.
- `scripts/check_persistence_contract.py` -- AST lint for the stage persistence contracts.

`pre-push` (blocking; token-aware authoritative check):

- `scripts/check_living_docs_drift.py --base origin/main --head HEAD` -- fails when the
  unpushed range edits restricted paths without `[restricted-doc-edit-ok]` in a commit
  message in that range.

## Intentional tombstone edits

The pre-commit block is deliberately strict. For a genuine tombstone edit or
reintroduction, re-commit with `git commit --no-verify` and include
`[restricted-doc-edit-ok]` in the commit message so the pre-push drift check (and CI)
accept it.

## Hard enforcement (CI)

`scripts/check_living_docs_drift.py` is also wired into
`.github/workflows/living-docs-drift.yml`, the server-side gate on push/PR.

## Notes

- On Windows, Git for Windows / Git Bash runs POSIX-style hooks; no extra setup.
- `pre-commit` inspects the staged/working state; `pre-push` inspects `origin/main..HEAD`.
- `scripts/warn_restricted_doc_edits.py` remains a standalone, non-blocking warning.
