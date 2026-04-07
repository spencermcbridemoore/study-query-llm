# STEP-02 Run/Request Lifecycle Plan

## Metadata

- Step ID: `STEP-02`
- Plan title: `Generic run/request lifecycle contract definition`
- Status: `active`
- Execution mode: `draft_provisional`
- Last updated: `2026-04-07`
- Source meta-plan: `docs/plans/MASTER_META_PLAN.md`

## Objective

Define `C-002@1.0` as the canonical, method-agnostic lifecycle contract for run/request planning so downstream steps can specialize behavior without redefining core request states, transition rules, validation expectations, or compatibility guarantees. This step uses `C-001@1.0` as the planning artifact baseline and produces an active lifecycle contract that remains independent of clustering-specific algorithm choices.

## Allowed Assumptions

- `A-001`
- `A-002`
- `A-003`
- `A-004`
- `A-005`
- `A-006`
- `A-007`
- `A-009`

## Required Input Contracts

| Contract ID | Version | Why Needed |
| --- | --- | --- |
| `C-001@1.0` | `1.0` | Provides the baseline planning artifact semantics and metadata discipline needed to define `C-002@1.0` in a reusable, contract-first format. |

## Contract Source Resolution

| Contract ID | Selected Source | Why This Source Was Chosen |
| --- | --- | --- |
| `C-001@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-01_master_bootstrap.md` is the highest-priority available source and is augmented by master seed semantics only where additional scaffolding is needed. |

## Forbidden Dependencies

- No clustering-specific algorithm choices.
- No implementation details from sibling steps.
- No hidden assumptions outside listed `A-*`.
- No references to non-canonical contract IDs.

## Output Artifacts

- Primary output document: `docs/plans/STEP-02_run_request_lifecycle.md`
- Produced/updated contract(s): `C-002@1.0` (`active`)
- Optional supporting appendix: _None_

## C-002@1.0 Contract Definition

### Name

- `GenericRunRequestContract`

### Purpose

- Define a generic request-to-run lifecycle model that all method specializations can inherit without redefining core state semantics.

### Producer Step(s)

- `STEP-02`

### Consumer Step(s)

- `STEP-03`
- `STEP-06`
- `STEP-07`

### Lifecycle Semantics

| State | Meaning | Allowed Next States |
| --- | --- | --- |
| `drafted` | Request intent captured but not yet validated. | `validated`, `rejected`, `cancelled` |
| `validated` | Request passes contract and policy checks. | `queued`, `cancelled` |
| `queued` | Request accepted and waiting for execution assignment. | `dispatching`, `cancelled`, `expired` |
| `dispatching` | Execution assignment in progress. | `running`, `failed` |
| `running` | Work is actively executing. | `succeeded`, `failed`, `cancelled` |
| `succeeded` | Terminal success state with outputs finalized. | _None (terminal)_ |
| `failed` | Terminal failure state with reason captured. | _None (terminal)_ |
| `cancelled` | Terminal cancellation state (user/system initiated). | _None (terminal)_ |
| `rejected` | Terminal validation/policy rejection before queueing. | _None (terminal)_ |
| `expired` | Terminal timeout/ttl expiry before dispatch. | _None (terminal)_ |

Lifecycle invariants:

1. State progression is forward-only; backward transitions are invalid.
2. Terminal states are immutable except for backward-compatible metadata enrichment.
3. Every state transition must record transition timestamp and actor/source.
4. `running`, `succeeded`, and `failed` require a bound `run_id`.

### Schema

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `contract_id` | `string` | Must equal `C-002@1.0`. |
| `request_id` | `string` | Stable identifier for the request lifecycle entity. |
| `lifecycle_state` | `enum` | One of the defined lifecycle states in this contract. |
| `created_at_utc` | `datetime` | Request creation timestamp in UTC ISO-8601 format. |
| `updated_at_utc` | `datetime` | Last state mutation timestamp in UTC ISO-8601 format. |
| `requested_by` | `string` | Logical request originator (user/system/automation identity). |
| `input_ref_ids` | `array[string]` | References to input artifacts needed by the request. |
| `method_ref` | `string` | Generic method/plugin reference consumed by specialization steps. |
| `transition_log` | `array[object]` | Ordered list of lifecycle transitions with from/to, timestamp, and actor/source. |

Optional fields:

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | `string` | Execution identifier assigned when request reaches runtime states. |
| `queue_ref` | `string` | Logical queue or lane label used for scheduling. |
| `priority` | `string` | Optional scheduling priority label (`low`, `normal`, `high`, etc.). |
| `retry_of_request_id` | `string` | Request lineage pointer when this request is a retry. |
| `supersedes_request_id` | `string` | Request lineage pointer for replacement semantics. |
| `terminal_reason` | `string` | Human-readable reason for terminal states (`failed`, `cancelled`, `rejected`, `expired`). |
| `provenance_ref` | `string` | Link to provenance semantics formalized by downstream contracts. |
| `metadata` | `object` | Backward-compatible extension container for non-breaking annotations. |

### Validation Checks

1. `contract_id` equals `C-002@1.0`.
2. `request_id` is non-empty and unique within the active planning scope.
3. Required timestamps are valid UTC ISO-8601 values and `updated_at_utc >= created_at_utc`.
4. `lifecycle_state` is a valid enum member and must match the last transition in `transition_log`.
5. Transitions must follow the allowed-next-state matrix; invalid edges fail validation.
6. `run_id` is mandatory for `running`, `succeeded`, and `failed` states.
7. Terminal states require `terminal_reason` when the state is not `succeeded`.
8. `input_ref_ids` must contain at least one entry for `validated` and later states.
9. `transition_log` entries must be time-ordered and append-only.

### Backward Compatibility Policy

1. Minor (`@1.x`) updates may add optional fields, clarify text, and extend enum values only when existing consumers can safely ignore unknown values.
2. Minor updates must not remove required fields or alter existing state meanings.
3. Major (`@2.0+`) updates are required for required-field removals/additions, breaking transition-rule changes, or semantic reinterpretation of existing states.
4. Producers and consumers must preserve unknown optional fields to support forward compatibility across independently authored steps.
5. Deprecated fields must remain readable for at least one major version window before retirement.

## Proposed Plan

1. Establish generic lifecycle states and transition constraints that remain method-agnostic.
2. Define required/optional schema fields that support downstream specialization without coupling to implementation internals.
3. Publish validation and compatibility rules so `STEP-03`, `STEP-06`, and `STEP-07` can consume `C-002@1.0` predictably.

## Definition Of Done

- [x] Define `C-002@1.0` lifecycle semantics with explicit states, transitions, and invariants.
- [x] Define `C-002@1.0` validation checks with enforceable contract rules.
- [x] Define `C-002@1.0` backward compatibility policy covering minor vs major version changes.
- [x] Produce one step plan document at `docs/plans/STEP-02_run_request_lifecycle.md` with assumption ledger and non-goals.

## Validation Checks

1. PASS: All referenced assumptions are valid `A-*` IDs and constrained to the allowed list.
2. PASS: Dependencies remain contract-based (`C-*`) and avoid implementation-detail coupling.
3. PASS: Output artifact path matches `STEP-02` naming and deliverable requirements.
4. PASS: Forbidden dependency on clustering-specific algorithm choices is explicit.
5. PASS: Non-goals and assumption ledger are included and explicit.

## Non-Goals

- Define clustering-specific algorithm behavior, hyperparameter policy, or method internals.
- Specify worker orchestration internals, queue implementation details, or runtime infrastructure.
- Replace downstream contract ownership for `C-003@1.0`, `C-004@1.0`, `C-005@1.0`, or `C-006@1.0`.
- Replace `docs/IMPLEMENTATION_PLAN.md` as implementation status authority.

## Failure And Rollback Behavior

- In `draft_provisional`, if a required contract is missing or ambiguous, proceed with best-available source and log challenged assumptions.
- In `finalize_gated`, if a required contract is missing or ambiguous, set status to `blocked` and request clarification.
- If a contract change is breaking, publish a new major version and retain old contract references.
- If overlap with another step is discovered, defer to producer ownership in the master step registry.

## Assumption Ledger

| ID | Statement | Status | Notes |
| --- | --- | --- | --- |
| `A-001` | All planning artifacts live under `docs/plans/` unless explicitly noted. | `accepted` | Output path and artifact boundaries follow this location rule. |
| `A-002` | `docs/IMPLEMENTATION_PLAN.md` remains the implementation status source of truth. | `accepted` | This document defines planning contracts only, not runtime status tracking. |
| `A-003` | Cross-step dependencies are expressed only through `C-*` contracts, not implementation details. | `accepted` | All upstream/downstream coupling is represented via `C-001@1.0` and `C-002@1.0`. |
| `A-004` | Each section plan can be authored in a fresh chat with only the step header and master doc. | `accepted` | Contract source precedence and required sections are fully captured in this document. |
| `A-005` | Contract IDs are stable and append-only; breaking changes require a new major contract version. | `accepted` | Compatibility policy enforces major-version requirements for breaking changes. |
| `A-006` | Section plans must include explicit non-goals and forbidden dependencies. | `accepted` | Both sections are explicit and enforce scope limits. |
| `A-007` | Execution mode controls blocking: `draft_provisional` proceeds with fallback semantics; `finalize_gated` blocks on unresolved contracts. | `accepted` | Failure/rollback handling mirrors mode policy exactly. |
| `A-009` | Default mode for new step runs is `draft_provisional` unless explicitly overridden in the step header. | `accepted` | This step executes in `draft_provisional` per starter definition. |

## Contract Change Log

| Contract ID | Change Type | Version Impact | Summary |
| --- | --- | --- | --- |
| `C-002@1.0` | `clarify` | `none` | Promoted from STEP-01 draft placeholder to active producer-defined contract with lifecycle semantics, validation checks, and compatibility policy. |
