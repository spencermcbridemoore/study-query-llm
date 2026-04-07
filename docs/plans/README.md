# Plans Workflow Guide

## Purpose

This folder contains a contract-driven planning workflow for producing many plan documents from one master meta-plan.

Use this when a task is too large to plan in a single document and needs isolated, step-by-step planning outputs.

## Canonical Files

- `docs/plans/MASTER_META_PLAN.md`
- `docs/plans/templates/STEP_EXECUTION_HEADER.md`
- `docs/plans/templates/SECTION_PLAN_TEMPLATE.md`

## Operating Model

1. Maintain one canonical master document (`MASTER_META_PLAN.md`) with:
   - assumptions registry (`A-*`)
   - contract registry (`C-*`)
   - step registry (`STEP-*`)
2. Execute one step at a time.
3. Generate one section plan output per step.
4. Keep cross-step dependencies contract-based only.

## Recommended Chat Strategy

- Use the current chat to establish or update the master meta-plan.
- Use a fresh chat for each section step, with the copy-paste header template.
- Avoid using ambiguous prompts like "refer to step X" without the structured header.

## Quick Start

### Step 1: Verify Master Inputs

Confirm `docs/plans/MASTER_META_PLAN.md` exists and includes:

- assumptions (`A-*`)
- active contracts (`C-*`)
- step definitions (`STEP-*`)

### Step 2: Copy Header Into New Chat

Use `docs/plans/templates/STEP_EXECUTION_HEADER.md` and fill in:

- `STEP-XX`
- allowed assumptions
- required contracts
- deliverable path
- definition of done

### Step 3: Generate Section Plan

The output should follow `docs/plans/templates/SECTION_PLAN_TEMPLATE.md` and remain within the declared scope.

### Step 4: Validate Independence

Before marking complete, verify:

1. assumptions are only referenced by `A-*` IDs
2. dependencies are only referenced by `C-*` IDs
3. no sibling-step implementation details are required
4. non-goals are explicit

## Naming Conventions

- Step outputs: `docs/plans/STEP-XX_<short_name>.md`
- Optional appendix: `docs/plans/appendix/STEP-XX_<short_name>_appendix.md`
- Contracts: `C-XXX@MAJOR.MINOR`

## Update Policy

- Keep IDs stable and append-only.
- For breaking contract changes, publish a new major version.
- Do not remove old IDs from history; mark them deprecated/retired.
- If ownership conflicts appear, resolve using producer ownership declared in the master meta-plan.

## Troubleshooting

- Missing required contract: mark step `blocked` and request clarification.
- Ambiguous assumption: do not proceed until assumption is clarified and registered.
- Scope drift: move out-of-scope items into non-goals and defer to another step.
