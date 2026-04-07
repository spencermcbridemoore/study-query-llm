# Step Starters (Copy-Paste)

These starters apply master defaults to reduce interactive clarification questions.

## STEP-01

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-01
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-009]
Required input contracts: []
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No implementation details from other step docs
Deliverable: docs/plans/STEP-01_master_bootstrap.md
Definition of done:
- Set C-001@1.0 to active and keep C-002@1.0..C-006@1.0 as draft placeholders
- Produce one step plan doc with non-goals, assumption ledger, and contract change log
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-02

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-02
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-009]
Required input contracts: [C-001@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No clustering-specific algorithm choices
Deliverable: docs/plans/STEP-02_run_request_lifecycle.md
Definition of done:
- Define C-002@1.0 with lifecycle semantics, validation checks, and compatibility policy
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-03

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-03
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-008, A-009]
Required input contracts: [C-001@1.0, C-002@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No panel UX implementation specifics
Deliverable: docs/plans/STEP-03_method_plugin_contract.md
Definition of done:
- Define method plugin contract semantics and C-004@1.0 boundaries
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-04

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-04
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-009]
Required input contracts: [C-001@1.0, C-003@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No worker orchestration internals
Deliverable: docs/plans/STEP-04_input_artifact_plan.md
Definition of done:
- Define input ingestion and artifact eligibility planning semantics for C-003@1.0
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-05

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-05
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-008, A-009]
Required input contracts: [C-002@1.0, C-003@1.0, C-004@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No hard-coding framework to clustering-only
Deliverable: docs/plans/STEP-05_clustering_specialization.md
Definition of done:
- Define clustering specialization boundaries using generic contract semantics
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-06

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-06
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-009]
Required input contracts: [C-002@1.0, C-004@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No UI-only assumptions
Deliverable: docs/plans/STEP-06_analysis_provenance.md
Definition of done:
- Define analysis/provenance semantics and C-005@1.0 compatibility constraints
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-07

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-07
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-009]
Required input contracts: [C-002@1.0, C-005@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed, use best available source, and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No algorithm implementation internals
Deliverable: docs/plans/STEP-07_panel_orchestration_ux.md
Definition of done:
- Define panel orchestration and resource-estimation UX semantics with explicit scope boundaries
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```

## STEP-08

```text
Meta-plan source: docs/plans/MASTER_META_PLAN.md
Template source: docs/plans/templates/SECTION_PLAN_TEMPLATE.md
Step to execute: STEP-08
Execution mode: draft_provisional
Allowed assumptions: [A-001, A-002, A-003, A-004, A-005, A-006, A-007, A-008, A-009]
Required input contracts: [C-001@1.0, C-002@1.0, C-003@1.0, C-004@1.0, C-005@1.0, C-006@1.0]
Contract source precedence: standalone contract file -> producer step output -> master seed + section template scaffolding
Missing contract policy:
- draft_provisional: proceed for C-001..C-005 and log challenged assumptions
- finalize_gated: block and request clarification
Forbidden dependencies: No introduction of new core contracts
Deliverable: docs/plans/STEP-08_cutover_risk_policy.md
Definition of done:
- Define full C-006@1.0 cutover policy semantics (sequencing, guardrails, rollback triggers, validation gates)
Output constraints:
- Produce only the requested step plan document
- Include assumption ledger and non-goals
- Honor the missing contract policy based on execution mode
```
