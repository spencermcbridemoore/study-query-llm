# Git Hooks (Living-Docs-Only Governance)

Optional, opt-in Git hooks that surface the living-docs-only rule
(`.cursor/rules/living-docs-only.mdc`) at commit time.

## Install (per-clone, opt-in)

```sh
git config core.hooksPath scripts/git-hooks
```

This is local to the clone (it edits `.git/config`, not the repo). It does not
modify any global git settings.

## Uninstall

```sh
git config --unset core.hooksPath
```

## What's here

- `pre-commit` -- runs `scripts/warn_restricted_doc_edits.py`. Prints a warning
  to stderr when staged changes touch restricted paths
  (`docs/history/**`, `docs/deprecated/**`, `docs/plans/**`,
  `docs/experiments/**`, `scripts/history/**`, `scripts/deprecated/**`, plus the
  named legacy files in `scripts/internal/living_docs_governance.py`). Always
  exits 0 -- this is a heads-up, not a block.

## Hard enforcement

The CI hard check is `scripts/check_living_docs_drift.py`, wired into
`.github/workflows/living-docs-drift.yml`. It fails when a PR/diff range edits
restricted paths without `[restricted-doc-edit-ok]` in any commit message in
the range.

## Notes

- On Windows, Git for Windows / Git Bash runs POSIX-style hooks; no extra
  setup is required.
- The hook only inspects staged paths; running `git commit -- <pathspec>` with
  a narrow pathspec narrows what the hook sees.
