# Section Plan Template

> Use this template to produce one independent section plan for a single `STEP-XX`.

## Metadata

- Step ID: `STEP-XX`
- Plan title: `<title>`
- Status: `draft | active | blocked | completed`
- Execution mode: `draft_provisional | finalize_gated`
- Last updated: `YYYY-MM-DD`
- Source meta-plan: `docs/plans/MASTER_META_PLAN.md`

## Objective

Describe the outcome this step must produce, in one paragraph.

## Allowed Assumptions

- `A-...`
- `A-...`

Only include assumptions listed in the master meta-plan.

## Required Input Contracts

| Contract ID | Version | Why Needed |
| --- | --- | --- |
| `C-...` | `X.Y` | `<reason>` |

## Contract Source Resolution

| Contract ID | Selected Source | Why This Source Was Chosen |
| --- | --- | --- |
| `C-...` | `standalone_file | producer_step_output | master_seed_plus_template` | `<reason>` |

## Forbidden Dependencies

- No implementation details from sibling steps.
- No hidden assumptions outside listed `A-*`.
- No references to non-canonical contract IDs.

## Output Artifacts

- Primary output document: `docs/plans/STEP-XX_<short_name>.md`
- Produced/updated contract(s): `C-...`
- Optional supporting appendix: `docs/plans/appendix/STEP-XX_<short_name>_appendix.md`

## Proposed Plan

1. `<action>`
2. `<action>`
3. `<action>`

## Definition Of Done

- [ ] DoD item 1
- [ ] DoD item 2
- [ ] DoD item 3

## Validation Checks

1. All referenced assumptions are valid `A-*` IDs.
2. All dependencies are contract-based (`C-*`), not implementation-detail-based.
3. Output artifact path matches the step and naming conventions.
4. Non-goals are explicit and enforce scope boundaries.

## Non-Goals

- `<what this step intentionally does not solve>`
- `<what is deferred to other steps>`

## Failure And Rollback Behavior

- In `draft_provisional`, if a required contract is missing or ambiguous, proceed with best-available source and log challenged assumptions.
- In `finalize_gated`, if a required contract is missing or ambiguous, set status to `blocked` and request clarification.
- If a contract change is breaking, publish a new major version and retain old contract references.
- If overlap with another step is discovered, defer to producer ownership in the master step registry.

## Assumption Ledger

| ID | Statement | Status | Notes |
| --- | --- | --- | --- |
| `A-...` | `<statement>` | `accepted | challenged` | `<note>` |

## Contract Change Log

| Contract ID | Change Type | Version Impact | Summary |
| --- | --- | --- | --- |
| `C-...` | `add | clarify | deprecate | retire` | `major | minor | none` | `<summary>` |
