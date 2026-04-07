# STEP-03 Method Plugin Contract Plan

## Metadata

- Step ID: `STEP-03`
- Plan title: `Method plugin specialization contract definition`
- Status: `active`
- Execution mode: `draft_provisional`
- Last updated: `2026-04-07`
- Source meta-plan: `docs/plans/MASTER_META_PLAN.md`

## Objective

Define `C-004@1.0` as the canonical method/plugin specialization contract that composes with the generic lifecycle in `C-002@1.0` and preserves plan-level independence. This step establishes boundaries for how method plugins declare identity, compatibility, parameters, outputs, and provenance requirements without coupling to panel UX implementation specifics or clustering-only internals.

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
| `C-001@1.0` | `1.0` | Provides baseline planning artifact semantics and lifecycle governance used to define `C-004@1.0` in a reusable, contract-first format. |
| `C-002@1.0` | `1.0` | Provides method-agnostic run/request lifecycle semantics, including `method_ref`, that `C-004@1.0` must specialize without redefining core lifecycle behavior. |

## Contract Source Resolution

| Contract ID | Selected Source | Why This Source Was Chosen |
| --- | --- | --- |
| `C-001@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-01_master_bootstrap.md` is the highest-priority available source and is augmented by master seed semantics only where additional scaffolding is needed. |
| `C-002@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-02_run_request_lifecycle.md` is the highest-priority available source and is consumed as the canonical upstream lifecycle contract. |

## Forbidden Dependencies

- No panel UX implementation specifics.
- No implementation details from sibling steps.
- No hidden assumptions outside listed `A-*`.
- No references to non-canonical contract IDs.

## Output Artifacts

- Primary output document: `docs/plans/STEP-03_method_plugin_contract.md`
- Produced/updated contract(s): `C-004@1.0` (`active`)
- Optional supporting appendix: _None_

## C-004@1.0 Contract Definition

### Name

- `MethodSpecializationContract`

### Purpose

- Define how method plugins specialize generic request/run semantics while keeping lifecycle ownership in `C-002@1.0`.
- Standardize method identity, plugin registration keys, parameter/result schemas, and compatibility guarantees for downstream specialization and provenance steps.

### Producer Step(s)

- `STEP-03` (base specialization contract definition)
- `STEP-05` (domain specialization using this base contract)

### Consumer Step(s)

- `STEP-05`
- `STEP-06`

### Contract Semantics And Boundaries

Boundary invariants:

1. `C-002@1.0` remains the authority for lifecycle states and transition rules; `C-004@1.0` only adds method specialization semantics.
2. `method_ref` in `C-002@1.0` must resolve to one and only one method/plugin definition under `C-004@1.0`.
3. Method plugins may define parameters and outputs, but must not alter generic request identity, lifecycle progression, or terminal-state meanings from `C-002@1.0`.
4. Method specialization metadata must remain framework-agnostic and transportable across orchestration contexts.
5. Clustering is treated as one specialization instance, not a separate planning architecture.

Boundary table:

| Area | In Scope For `C-004@1.0` | Out Of Scope For `C-004@1.0` |
| --- | --- | --- |
| Method identity | Stable `method_ref`, name/family/version semantics | UI naming conventions or display formatting |
| Plugin contract | Registry/entrypoint key and compatibility declarations | Worker scheduling internals or queue mechanics |
| Method parameters | Parameter schema and defaults for specialization logic | Algorithm-specific implementation code paths |
| Method outputs | Structured result schema and artifact expectations | Panel rendering behavior for results |
| Provenance requirements | Required trace fields for method/result lineage | Full analysis policy ownership (deferred to `C-005@1.0`) |

### Schema

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `contract_id` | `string` | Must equal `C-004@1.0`. |
| `method_ref` | `string` | Canonical method/plugin identifier used by `C-002@1.0` requests. |
| `method_name` | `string` | Stable human-readable method name. |
| `method_family` | `string` | Method family/category token (for example: `clustering`, `mcq`, `analysis`). |
| `method_version` | `string` | Version for method behavior and output semantics. |
| `plugin_key` | `string` | Framework-agnostic plugin registry/entrypoint key used to resolve execution logic. |
| `input_contract_refs` | `array[string]` | Contract IDs consumed by the method; must include `C-002@1.0`. |
| `output_contract_refs` | `array[string]` | Contract IDs produced or updated by method execution/results. |
| `parameter_schema` | `object` | Structured parameter schema for method-specific configuration. |
| `result_schema` | `object` | Structured schema describing expected result payload shape. |
| `compatibility` | `object` | Supported request contract versions and lifecycle-state expectations. |
| `provenance_requirements` | `object` | Required provenance keys to preserve method/result lineage. |

Optional fields:

| Field | Type | Description |
| --- | --- | --- |
| `default_parameters` | `object` | Default parameter values compatible with `parameter_schema`. |
| `artifact_policy` | `object` | Optional artifact eligibility/retention hints for method outputs. |
| `resource_hints` | `object` | Optional compute/memory/concurrency guidance for planning-level sizing. |
| `analysis_hooks` | `array[object]` | Optional post-execution analysis registrations expected by downstream steps. |
| `deprecation` | `object` | Optional deprecation metadata (`status`, `sunset_date`, `replacement_method_ref`). |
| `metadata` | `object` | Backward-compatible extension container for non-breaking annotations. |

### Validation Checks

1. `contract_id` equals `C-004@1.0`.
2. `method_ref` is non-empty, stable, and unique within the active planning scope.
3. `method_version` is non-empty and must be updated for semantic output changes.
4. `input_contract_refs` contains `C-002@1.0` and only valid canonical `C-*` IDs.
5. `plugin_key` is non-empty and deterministic for the same method/version pair.
6. `parameter_schema` and `result_schema` are objects; `default_parameters` (if present) must validate against `parameter_schema`.
7. `compatibility` must not redefine or conflict with `C-002@1.0` lifecycle semantics.
8. `provenance_requirements` must include method identity and version trace keys at minimum.
9. Unknown optional fields are allowed only under `metadata` to preserve contract extensibility boundaries.

### Backward Compatibility Policy

1. Minor (`@1.x`) updates may add optional fields, tighten descriptive guidance, and add backward-compatible capability metadata.
2. Minor updates must not remove required fields or reinterpret `method_ref`, `plugin_key`, or lifecycle composition rules with `C-002@1.0`.
3. Major (`@2.0+`) updates are required for required-field additions/removals, breaking schema interpretation changes, or incompatible compatibility rules.
4. Deprecated method refs must remain readable for at least one major-version window before retirement.
5. Producers and consumers must preserve unknown optional fields in `metadata` for forward compatibility across independently authored steps.

## Proposed Plan

1. Establish `C-004@1.0` as the contract layer that binds generic request lifecycle semantics (`C-002@1.0`) to method/plugin specialization metadata.
2. Define a stable schema for method identity, plugin resolution, parameter/result contracts, and provenance requirements.
3. Publish validation and compatibility rules so downstream specialization (`STEP-05`) and provenance consumers (`STEP-06`) can adopt method definitions predictably.

## Definition Of Done

- [x] Define method/plugin contract semantics and explicit `C-004@1.0` boundaries.
- [x] Define `C-004@1.0` schema, validation checks, and backward compatibility policy.
- [x] Keep scope independent from panel UX implementation specifics.
- [x] Produce one step plan document at `docs/plans/STEP-03_method_plugin_contract.md` with assumption ledger and non-goals.

## Validation Checks

1. PASS: All referenced assumptions are valid `A-*` IDs and constrained to the allowed list.
2. PASS: Dependencies remain contract-based (`C-*`) and avoid implementation-detail coupling.
3. PASS: Output artifact path matches `STEP-03` naming and deliverable requirements.
4. PASS: Forbidden dependency on panel UX implementation specifics is explicit.
5. PASS: Non-goals and assumption ledger are included and explicit.

## Non-Goals

- Define panel UX layout, interaction flows, or visualization behavior for method selection/results.
- Define clustering-specific algorithm internals, hyperparameter tuning policy, or execution code paths.
- Define worker orchestration internals, lease/queue mechanics, or runtime scheduler behavior.
- Replace downstream ownership of analysis/provenance semantics formalized by `C-005@1.0`.

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
| `A-003` | Cross-step dependencies are expressed only through `C-*` contracts, not implementation details. | `accepted` | Upstream/downstream coupling is represented via `C-001@1.0`, `C-002@1.0`, and `C-004@1.0`. |
| `A-004` | Each section plan can be authored in a fresh chat with only the step header and master doc. | `accepted` | This output is self-contained for independent downstream execution. |
| `A-005` | Contract IDs are stable and append-only; breaking changes require a new major contract version. | `accepted` | Compatibility policy enforces major-version requirements for breaking changes. |
| `A-006` | Section plans must include explicit non-goals and forbidden dependencies. | `accepted` | Both sections are explicit and enforce scope boundaries. |
| `A-007` | Execution mode controls blocking: `draft_provisional` proceeds with fallback semantics; `finalize_gated` blocks on unresolved contracts. | `accepted` | Failure/rollback handling mirrors mode policy exactly. |
| `A-008` | Clustering remains a specialization use case, not a separate planning architecture. | `accepted` | `C-004@1.0` remains method-generic and does not hard-code clustering-only semantics. |
| `A-009` | Default mode for new step runs is `draft_provisional` unless explicitly overridden in the step header. | `accepted` | This step executes in `draft_provisional` per starter definition. |

## Contract Change Log

| Contract ID | Change Type | Version Impact | Summary |
| --- | --- | --- | --- |
| `C-004@1.0` | `clarify` | `none` | Promoted from STEP-01 draft placeholder to active producer-defined contract with method/plugin specialization semantics and explicit scope boundaries. |
