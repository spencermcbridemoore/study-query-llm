---
name: interrupt-checkpoint
description: Handles mid-task interruption safely by pausing work, reporting changes/side effects, and requiring explicit keep-vs-undo confirmation before proceeding.
disable-model-invocation: true
---
# Interrupt Checkpoint

## Goal

Provide a safe, explicit interruption workflow whenever the user pauses or redirects active work.

## Protocol

1. Stop ongoing work immediately.
2. Report current state before doing anything else:
   - Files created/modified/deleted.
   - Commands already run.
   - Background processes started (and current status if known).
   - Non-file side effects (database writes, migrations, API calls, external state changes).
3. Ask the user to choose one option and wait:
   - A) Keep everything and continue
   - B) Keep files but stop/cleanup processes only
   - C) Undo file changes only
   - D) Undo everything possible (files + processes), and list anything non-revertible
   - E) Custom instructions
4. Do not make further changes until the user chooses A/B/C/D/E.
5. If undo is chosen, show a preview of what will be undone before executing.
6. After action, provide a short post-action state summary.

## Reporting rules

- Be explicit and complete; do not assume intent.
- If nothing changed in a category, say "none".
- Call out irreversible side effects clearly.

## Priority

Safety and explicit user confirmation are higher priority than speed.
