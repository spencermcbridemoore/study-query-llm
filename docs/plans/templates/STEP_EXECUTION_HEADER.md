# Step Execution Header Template

Use this header at the start of a new chat to execute exactly one step from the master meta-plan.

## Copy-Paste Header

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Step to execute: STEP-XX
Allowed assumptions: [A-001, A-00Y]
Required input contracts: [C-001@1.0, C-00Z@1.0]
Forbidden dependencies: No implementation details from other step docs
Deliverable: docs/plans/STEP-XX_<short_name>.md
Definition of done:
- DoD item 1
- DoD item 2
- DoD item 3
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- If required contract is missing or ambiguous, stop and request clarification
```

## Minimal One-Liner Variant

Use only after the master meta-plan and step registry already exist.

```text
Execute STEP-XX from docs/plans/MASTER_META_PLAN.md and produce only its plan doc output using required contracts and assumptions.
```

## Required Validations Before Starting

1. `STEP-XX` exists in `docs/plans/MASTER_META_PLAN.md`.
2. All listed `A-*` and `C-*` identifiers are valid and active.
3. Deliverable path is unique for this step.
4. Definition of done is explicit and testable.
