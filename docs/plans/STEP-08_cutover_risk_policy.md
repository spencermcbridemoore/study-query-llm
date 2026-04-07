# STEP-08 Cutover Risk Policy Plan

## Metadata

- Step ID: `STEP-08`
- Plan title: `Cutover sequencing and risk policy contract definition`
- Status: `active`
- Execution mode: `draft_provisional`
- Last updated: `2026-04-07`
- Source meta-plan: `docs/plans/MASTER_META_PLAN.md`

## Objective

Define `C-006@1.0` as the canonical cutover policy contract that composes with established planning contracts (`C-001@1.0` through `C-005@1.0`) and provides full semantics for cutover sequencing, guardrails, rollback triggers, and validation gates. This step creates a contract-driven safety and decision framework for future execution checklists without introducing new core contracts or coupling to implementation internals.

## Allowed Assumptions

- `A-001`
- `A-002`
- `A-003`
- `A-004`
- `A-005`
- `A-006`
- `A-007`
- `A-008`
- `A-009`

## Required Input Contracts

| Contract ID | Version | Why Needed |
| --- | --- | --- |
| `C-001@1.0` | `1.0` | Provides planning artifact governance and metadata discipline used to structure cutover policy artifacts and decision records. |
| `C-002@1.0` | `1.0` | Provides lifecycle semantics and transition expectations required to define phase gates and transition-safe cutover rules. |
| `C-003@1.0` | `1.0` | Provides input/artifact eligibility and state semantics required for readiness gates and publishability checks during cutover. |
| `C-004@1.0` | `1.0` | Provides method specialization compatibility boundaries required to prevent cutover policy from violating method contract ownership. |
| `C-005@1.0` | `1.0` | Provides provenance/evidence semantics required for gate auditability, rollback evidence, and post-cutover validation. |
| `C-006@1.0` | `1.0` | Required by starter definition; this step is the producer and therefore refines the seed placeholder into a full cutover policy contract. |

## Contract Source Resolution

| Contract ID | Selected Source | Why This Source Was Chosen |
| --- | --- | --- |
| `C-001@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-01_master_bootstrap.md` is the highest-priority available source. |
| `C-002@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-02_run_request_lifecycle.md` is the highest-priority available source. |
| `C-003@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-04_input_artifact_plan.md` is the highest-priority available source. |
| `C-004@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-03_method_plugin_contract.md` is the highest-priority available source, with `docs/plans/STEP-05_clustering_specialization.md` specialization clarifications where relevant. |
| `C-005@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-06_analysis_provenance.md` is the highest-priority available source. |
| `C-006@1.0` | `master_seed_plus_template` | As the producer step, no prior standalone file or producer output exists; seed semantics from the master registry are expanded with section template scaffolding into a full contract definition. |

## Forbidden Dependencies

- No introduction of new core contracts.
- No implementation details from sibling steps.
- No hidden assumptions outside listed `A-*`.
- No references to non-canonical contract IDs.

## Output Artifacts

- Primary output document: `docs/plans/STEP-08_cutover_risk_policy.md`
- Produced/updated contract(s): `C-006@1.0` (`active`)
- Optional supporting appendix: _None_

## C-006@1.0 Contract Definition

### Name

- `CutoverPolicyContract`

### Purpose

- Define canonical sequencing phases and decision checkpoints for production cutover progression.
- Define guardrails and rollback conditions that protect contract-level correctness and operational safety boundaries.
- Define validation gate semantics and evidence requirements for pre-cutover, in-cutover, and post-cutover assurance.

### Producer Step(s)

- `STEP-08`

### Consumer Step(s)

- Future execution checklists
- Future release and migration runbooks

### Composition With Upstream Contracts

| Upstream Contract | Composition Rule | C-006 Constraint |
| --- | --- | --- |
| `C-001@1.0` | Cutover artifacts and decisions are documented using canonical planning metadata conventions. | `C-006@1.0` may require policy artifacts and logs, but must not redefine planning artifact ownership semantics. |
| `C-002@1.0` | Cutover phases and action permissions map to lifecycle legality and transition invariants. | `C-006@1.0` may gate progression by lifecycle outcomes, but must not alter lifecycle states or transition rules. |
| `C-003@1.0` | Readiness and publishability checks use normalized input and artifact eligibility semantics. | `C-006@1.0` may require eligibility evidence, but must not redefine artifact states or ingestion ownership. |
| `C-004@1.0` | Cutover scope and compatibility checks rely on method specialization identities and version constraints. | `C-006@1.0` may enforce compatibility gates, but must not redefine method schema ownership or plugin semantics. |
| `C-005@1.0` | Gate outcomes and rollback decisions require provenance-backed evidence and lineage anchors. | `C-006@1.0` may require evidence completeness thresholds, but must not redefine provenance state or lineage ownership. |

### Cutover Sequencing Semantics

Sequencing invariants:

1. Cutover progression is phase-ordered and forward-only unless an explicit rollback transition is triggered.
2. Every phase transition requires a recorded gate decision with timestamp, actor, and evidence references.
3. No phase may be skipped unless an approved exception is recorded in policy metadata.
4. Any hard-stop trigger immediately suspends forward progression and enters rollback evaluation.

Sequencing phases:

| Phase | Entry Criteria | Exit Criteria | Notes |
| --- | --- | --- | --- |
| `preflight` | Scope, contract bindings, and baseline snapshots are defined. | Pre-cutover gate bundle passes with no hard-stop findings. | Establishes deterministic baseline for later comparison. |
| `canary` | `preflight` exit complete; rollback path and owner acknowledgment recorded. | Canary validation gate passes and no rollback trigger threshold is exceeded. | Limited-scope exposure validates policy assumptions. |
| `progressive` | `canary` exit complete and expansion policy is approved. | Progressive gate passes for each ramp step; no sustained guardrail breach. | Gradual expansion phase with continuous gate checks. |
| `commit` | Progressive ramps complete and rollback feasibility remains confirmed. | Commit gate passes and full target scope is switched under policy control. | Point of intended full cutover activation. |
| `stabilize` | Commit gate complete and post-cutover observation window starts. | Stabilization gate passes and residual risk score is within accepted policy bounds. | Confirms durable post-cutover behavior. |
| `completed` | Stabilization evidence accepted by policy owner. | _Terminal success state_ | Policy-conformant cutover complete. |
| `rolled_back` | Rollback trigger criteria met and rollback decision approved/executed. | _Terminal rollback state_ | Captures controlled retreat with required evidence. |

### Guardrails And Policy Constraints

1. Contract binding guardrail: cutover must pin explicit versions for `C-001@1.0` through `C-005@1.0` before entering `preflight`.
2. Scope guardrail: cutover scope must be declared and immutable within a phase unless an approved exception is logged.
3. Evidence guardrail: every gate decision must include non-empty evidence references traceable via `C-005@1.0`.
4. Eligibility guardrail: promotion to `commit` requires no unresolved blocking eligibility issues under `C-003@1.0`.
5. Compatibility guardrail: method/profile compatibility claims must match `C-004@1.0` declarations for all in-scope methods.
6. Lifecycle guardrail: lifecycle outcomes used for gate decisions must remain consistent with `C-002@1.0` transition semantics.
7. Change-freeze guardrail: no breaking contract changes are allowed while policy state is `active_cutover`.

### Rollback Triggers And Decision Thresholds

Rollback trigger classes:

| Trigger Class | Trigger Condition | Decision Threshold | Required Action |
| --- | --- | --- | --- |
| `hard_gate_failure` | Any required validation gate returns `fail`. | Immediate (`>= 1` hard failure). | Halt progression and enter rollback evaluation without advancing phase. |
| `lifecycle_regression` | Lifecycle outcomes violate allowed transition or terminal-state expectations. | Sustained breach across configured observation window. | Freeze ramp progression; evaluate rollback to last stable phase. |
| `eligibility_regression` | Artifact/input eligibility confidence drops below policy threshold. | Breach of configured minimum eligibility score/rate. | Block `commit`; rollback if breach persists past tolerance window. |
| `compatibility_mismatch` | In-scope method/version compatibility diverges from `C-004@1.0` declarations. | Any unresolved mismatch at decision point. | Require correction or execute rollback path. |
| `provenance_gap` | Required evidence/provenance anchors are missing or incomplete. | Any missing mandatory evidence for gate sign-off. | Reject gate sign-off; rollback if unresolved within policy timeout. |

Rollback decision rules:

1. At least one hard-stop trigger class (`hard_gate_failure`) is mandatory in every policy instance.
2. Trigger thresholds must be explicit and machine-readable in the contract payload.
3. Rollback target phase must be declared before entering `canary`.
4. Rollback execution must generate a post-rollback validation bundle before policy state can be closed.

### Validation Gates

Pre-cutover gate requirements:

1. Contract-binding verification for `C-001@1.0` through `C-005@1.0` is complete and version-pinned.
2. Scope inventory and impacted method/version map are complete.
3. Baseline snapshot for lifecycle, eligibility, compatibility, and provenance indicators is recorded.
4. Rollback plan, owner approvals, and communication plan are recorded.

In-cutover gate requirements:

1. Phase transition decisions include complete evidence links and approver identity.
2. Guardrails are continuously checked and no active hard-stop trigger is unresolved.
3. Canary and progressive checks confirm compatibility and eligibility within policy thresholds.
4. Any exception use is explicitly approved and logged with expiration scope.

Post-cutover gate requirements:

1. Stabilization observation window completes with no unresolved high-severity breaches.
2. Provenance and decision artifacts are complete and audit-ready.
3. Residual risk assessment is recorded and accepted by policy owner.
4. Closure record includes either completion evidence or rollback completion evidence.

### Schema

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `contract_id` | `string` | Must equal `C-006@1.0`. |
| `cutover_policy_id` | `string` | Stable identifier for the cutover policy instance. |
| `policy_version` | `string` | Version token for policy semantics and thresholds. |
| `scope_ref` | `string` | Stable reference describing the release/change scope governed by this policy instance. |
| `contract_bindings` | `object` | Explicit version bindings for `C-001@1.0`..`C-005@1.0` used by this cutover policy. |
| `phase_sequence` | `array[string]` | Ordered phase list constrained to valid phase enum values. |
| `phase_criteria` | `array[object]` | Entry/exit criteria for each phase in `phase_sequence`. |
| `validation_gates` | `object` | Structured gate definitions for `pre_cutover`, `in_cutover`, and `post_cutover`. |
| `guardrails` | `array[object]` | Enforceable guardrail definitions with severity and owner fields. |
| `rollback_triggers` | `array[object]` | Trigger definitions including class, condition, threshold reference, and action. |
| `thresholds` | `object` | Explicit numeric/logical thresholds used by rollback and gate decisions. |
| `decision_log` | `array[object]` | Append-only ordered record of gate decisions, approvals, and phase transitions. |
| `policy_state` | `enum` | One of `drafted`, `ready`, `active_cutover`, `completed`, `rolled_back`, `closed`. |
| `evidence_bundle_ref` | `string` | Reference to aggregate evidence set supporting policy decisions. |
| `created_at_utc` | `datetime` | Policy creation timestamp in UTC ISO-8601 format. |
| `updated_at_utc` | `datetime` | Last mutation timestamp in UTC ISO-8601 format. |

Optional fields:

| Field | Type | Description |
| --- | --- | --- |
| `rollback_target_phase` | `string` | Optional explicit rollback landing phase for controlled retreat. |
| `observation_window` | `object` | Optional post-commit observation window definition (duration, sampling interval). |
| `exception_register` | `array[object]` | Optional approved exceptions with reason, owner, and expiry. |
| `communications_plan_ref` | `string` | Optional reference to cutover communication plan artifact. |
| `owner_acknowledgements` | `array[object]` | Optional owner approval records for phase progression. |
| `residual_risk_summary` | `object` | Optional structured summary of residual risks at closeout. |
| `supersedes_policy_id` | `string` | Optional reference to prior policy instance when superseding. |
| `metadata` | `object` | Backward-compatible extension container for non-breaking annotations. |

### Validation Checks

1. `contract_id` equals `C-006@1.0`.
2. `contract_bindings` includes non-empty version pins for `C-001@1.0` through `C-005@1.0`.
3. `phase_sequence` includes required phases in valid order (`preflight`, `canary`, `progressive`, `commit`, `stabilize`) and ends in either `completed` or `rolled_back`.
4. Each `phase_criteria` entry maps to exactly one phase and includes non-empty entry and exit criteria.
5. `validation_gates` contains `pre_cutover`, `in_cutover`, and `post_cutover` sections, each with required check definitions.
6. `rollback_triggers` includes at least one hard-stop class and all referenced thresholds resolve in `thresholds`.
7. `policy_state` transitions are forward-valid (`drafted -> ready -> active_cutover -> completed|rolled_back -> closed`).
8. `decision_log` is append-only and chronologically ordered by UTC timestamps.
9. `evidence_bundle_ref` is non-empty before transitioning to `completed`, `rolled_back`, or `closed`.
10. Unknown optional fields are allowed only under `metadata` to preserve extensibility boundaries.

### Backward Compatibility Policy

1. Minor (`@1.x`) updates may add optional policy fields, clarify gate descriptions, and refine non-breaking threshold semantics.
2. Minor updates must not remove required fields, reorder mandatory phase semantics, or redefine hard-stop rollback behavior.
3. Major (`@2.0+`) updates are required for required-field additions/removals, incompatible phase model changes, or incompatible rollback trigger semantics.
4. Producers and consumers must preserve unknown optional `metadata` fields for forward compatibility.
5. Deprecated policy fields must remain readable for at least one major-version window before retirement.

## Proposed Plan

1. Compose cutover policy semantics from `C-001@1.0` through `C-005@1.0` into one contract-driven governance model without introducing new core contracts.
2. Define full `C-006@1.0` sequencing, guardrail, rollback, and validation-gate semantics with explicit schema and validation rules.
3. Publish bounded, auditable policy semantics that future execution checklists can consume consistently across method families.

## Definition Of Done

- [x] Define full `C-006@1.0` cutover policy semantics covering sequencing, guardrails, rollback triggers, and validation gates.
- [x] Keep dependencies contract-driven (`C-*`) and avoid introducing new core contracts or sibling-step implementation internals.
- [x] Produce one step plan document at `docs/plans/STEP-08_cutover_risk_policy.md`.
- [x] Include explicit non-goals, assumption ledger, and contract change log.
- [x] Honor `draft_provisional` missing-contract behavior for `C-001@1.0`..`C-005@1.0` and record assumptions transparently.

## Validation Checks

1. PASS: All referenced assumptions are valid `A-*` IDs and constrained to the STEP-08 allowed list.
2. PASS: Required contracts are resolved via source precedence (producer outputs for `C-001@1.0`..`C-005@1.0`; producer-step seed expansion for `C-006@1.0`).
3. PASS: `C-006@1.0` semantics explicitly define sequencing, guardrails, rollback triggers, and validation gates.
4. PASS: Forbidden dependency against introducing new core contracts is explicit.
5. PASS: Output artifact path, assumption ledger, and non-goals satisfy STEP-08 output constraints.

## Non-Goals

- Introduce new core contracts beyond `C-006@1.0`.
- Define provider-specific rollout tooling, infrastructure commands, or deployment mechanics.
- Redefine lifecycle, input/artifact, method specialization, or provenance ownership from `C-002@1.0`..`C-005@1.0`.
- Define implementation code paths for automated rollback orchestration.
- Replace `docs/IMPLEMENTATION_PLAN.md` as implementation status source of truth.

## Failure And Rollback Behavior

- In `draft_provisional`, unresolved or ambiguous inputs for `C-001@1.0`..`C-005@1.0` do not block drafting; proceed with best-available producer semantics and log challenged assumptions.
- In `finalize_gated`, if a required contract is missing or ambiguous, set status to `blocked` and request clarification.
- If a cutover hard-stop trigger is activated, halt forward phase progression and execute the declared rollback decision path.
- If a contract change is breaking, publish a new major version and retain prior contract references.
- If ownership overlap is detected, defer to producer ownership defined in `docs/plans/MASTER_META_PLAN.md`.

## Assumption Ledger

| ID | Statement | Status | Notes |
| --- | --- | --- | --- |
| `A-001` | All planning artifacts live under `docs/plans/` unless explicitly noted. | `accepted` | Output and policy scope remain within `docs/plans/`. |
| `A-002` | `docs/IMPLEMENTATION_PLAN.md` remains the implementation status source of truth. | `accepted` | This document defines planning policy semantics only. |
| `A-003` | Cross-step dependencies are expressed only through `C-*` contracts, not implementation details. | `accepted` | All dependencies are contract-referenced (`C-001@1.0`..`C-006@1.0`). |
| `A-004` | Each section plan can be authored in a fresh chat with only the step header and master doc. | `accepted` | The document is self-contained and template-complete. |
| `A-005` | Contract IDs are stable and append-only; breaking changes require a new major contract version. | `accepted` | Compatibility policy enforces major-version handling for breaking changes. |
| `A-006` | Section plans must include explicit non-goals and forbidden dependencies. | `accepted` | Non-goals and forbidden dependency constraints are explicit. |
| `A-007` | Execution mode controls blocking: `draft_provisional` proceeds with fallback semantics; `finalize_gated` blocks on unresolved contracts. | `accepted` | Failure policy mirrors mode behavior exactly. |
| `A-008` | Clustering remains a specialization use case, not a separate planning architecture. | `accepted` | Cutover policy is method-family agnostic and does not privilege clustering architecture. |
| `A-009` | Default mode for new step runs is `draft_provisional` unless explicitly overridden in the step header. | `accepted` | STEP-08 executes in `draft_provisional` per starter constraints. |

## Contract Change Log

| Contract ID | Change Type | Version Impact | Summary |
| --- | --- | --- | --- |
| `C-006@1.0` | `clarify` | `none` | Promoted from STEP-01 draft placeholder to active producer-defined contract with full cutover policy semantics, including sequencing, guardrails, rollback triggers, and validation gates. |
