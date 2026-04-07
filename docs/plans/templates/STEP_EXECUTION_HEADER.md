# Step Execution Header Template

Use this header at the start of a new chat to execute exactly one step from the master meta-plan.

For prefilled, step-specific versions, use `docs/plans/templates/STEP_STARTERS.md`.

## Copy-Paste Header

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Step to execute: STEP-XX
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-00Y]
Required input contracts: [C-001@1.0, C-00Z@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No implementation details from other step docs
Deliverable: docs/plans/STEP-XX_<short_name>.md
Definition of done:
- DoD item 1
- DoD item 2
- DoD item 3
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## Minimal One-Liner Variant

Use only after the master meta-plan and step registry already exist.

```text
Execute STEP-XX from docs/plans/MASTER_META_PLAN.md in draft_provisional mode and produce only its plan doc output using contract source precedence rules.
```

## Required Validations Before Starting

1. `STEP-XX` exists in `docs/plans/MASTER_META_PLAN.md`.
2. All listed `A-*` and `C-*` identifiers are valid identifiers (even if some contracts are still draft).
3. Deliverable path is unique for this step.
4. Definition of done is explicit and testable.
5. Execution mode is declared and its missing-contract policy is included.
