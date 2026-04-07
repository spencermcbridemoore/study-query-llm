# STEP-04 Input/Artifact Plan

## Metadata

- Step ID: `STEP-04`
- Plan title: `Input ingestion and artifact eligibility contract definition`
- Status: `active`
- Execution mode: `draft_provisional`
- Last updated: `2026-04-07`
- Source meta-plan: `docs/plans/MASTER_META_PLAN.md`

## Objective

Define `C-003@1.0` as the canonical planning contract for input ingestion and artifact eligibility so downstream specialization steps can consume normalized, traceable inputs and predictable artifact readiness semantics without coupling to worker orchestration internals. This step formalizes ingestion boundaries, artifact eligibility rules, validation checks, and compatibility policy while preserving contract-driven independence.

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
| `C-001@1.0` | `1.0` | Provides planning artifact metadata and governance conventions used to define `C-003@1.0` in a reusable contract format. |
| `C-003@1.0` | `1.0` | Required by starter definition; this step is the producer and therefore refines the seed placeholder semantics into an active contract definition. |

## Contract Source Resolution

| Contract ID | Selected Source | Why This Source Was Chosen |
| --- | --- | --- |
| `C-001@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-01_master_bootstrap.md` is the highest-priority available source. |
| `C-003@1.0` | `master_seed_plus_template` | As the producer step, no prior standalone file or producer output exists yet; seed semantics from the master registry are expanded using the section template scaffolding. |

## Forbidden Dependencies

- No worker orchestration internals.
- No implementation details from sibling steps.
- No hidden assumptions outside listed `A-*`.
- No references to non-canonical contract IDs.

## Output Artifacts

- Primary output document: `docs/plans/STEP-04_input_artifact_plan.md`
- Produced/updated contract(s): `C-003@1.0` (`active`)
- Optional supporting appendix: _None_

## C-003@1.0 Contract Definition

### Name

- `InputAndArtifactContract`

### Purpose

- Define method-agnostic ingestion semantics for converting raw input references into normalized, validated planning inputs.
- Define artifact eligibility semantics so downstream specialization steps can consistently decide what outputs are publishable, retained, and traceable.

### Producer Step(s)

- `STEP-04`

### Consumer Step(s)

- `STEP-05`

### Contract Semantics And Boundaries

Ingestion invariants:

1. Every input item must have a stable `input_ref_id` and a declared source descriptor.
2. Normalization must be declarative and reproducible; contract metadata describes what transformation class was applied, not code-level implementation.
3. Input lineage must remain append-only to preserve provenance and replayability planning.
4. Validation outcomes must be explicit per input item (`accepted`, `rejected`, `deferred`) with reason codes.

Artifact eligibility invariants:

1. Artifact eligibility is determined by contract-declared criteria, not orchestration/runtime implementation details.
2. Eligibility and publication states are forward-only for a given artifact version.
3. Artifact identity must include lineage linkage back to contributing normalized inputs.
4. Retention and immutability policies are declared as planning constraints and must be interpretable without infrastructure-specific assumptions.

Boundary table:

| Area | In Scope For `C-003@1.0` | Out Of Scope For `C-003@1.0` |
| --- | --- | --- |
| Input references | Stable IDs, source descriptor minimums, lineage anchors | Transport adapter implementation details |
| Input normalization | Normalization profile declarations and compatibility semantics | Parser/tokenizer/ETL code internals |
| Input validation | Rule classes, status outcomes, reason code requirements | Runtime execution engine behavior |
| Artifact eligibility | Eligibility criteria, readiness states, reason tracking | Worker queueing, lease, and dispatch logic |
| Artifact policy | Retention/immutability planning fields and publication metadata | Storage backend internals and deployment configuration |

### Schema

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `contract_id` | `string` | Must equal `C-003@1.0`. |
| `input_set_id` | `string` | Stable identifier for the logical ingestion batch/set. |
| `input_items` | `array[object]` | Canonical list of inputs with `input_ref_id`, source descriptor, and declared format/type. |
| `normalization_profile` | `object` | Declarative normalization policy/profile identifier and version. |
| `validation_policy` | `object` | Declared validation rule classes and required reason-code taxonomy. |
| `lineage_map` | `array[object]` | Append-only lineage entries linking raw references to normalized inputs. |
| `artifact_candidates` | `array[object]` | Candidate artifact records with category, lineage refs, and eligibility evidence. |
| `eligibility_policy` | `object` | Criteria and thresholds used to classify candidates as eligible/ineligible. |
| `artifact_state` | `enum` | One of `draft`, `eligible`, `ineligible`, `published`, `superseded`. |
| `created_at_utc` | `datetime` | Contract entity creation timestamp in UTC ISO-8601 format. |
| `updated_at_utc` | `datetime` | Last mutation timestamp in UTC ISO-8601 format. |

Optional fields:

| Field | Type | Description |
| --- | --- | --- |
| `publication_metadata` | `object` | Optional publication channel/context metadata when state advances to `published`. |
| `retention_policy` | `object` | Optional retention window and archival hints as planning-level constraints. |
| `immutability_policy` | `object` | Optional immutability guarantees and supersession rules. |
| `quality_signals` | `array[object]` | Optional quality indicators used by eligibility policy. |
| `exclusion_log` | `array[object]` | Optional record of excluded input/artifact items with reason details. |
| `compatibility_tags` | `array[string]` | Optional tags for downstream method specialization compatibility routing. |
| `metadata` | `object` | Backward-compatible extension container for non-breaking annotations. |

### Validation Checks

1. `contract_id` equals `C-003@1.0`.
2. `input_set_id` is non-empty and unique within active planning scope.
3. `input_items` contains at least one item, and each item has a non-empty `input_ref_id` and source descriptor.
4. `lineage_map` entries reference existing `input_ref_id` values and are append-only by timestamp/order.
5. `artifact_candidates` must reference lineage entries and include declared artifact category/type.
6. `artifact_state` transitions are forward-only: `draft -> eligible|ineligible -> published|superseded`.
7. `eligibility_policy` must be present and parseable before `artifact_state` can be `eligible` or `published`.
8. `updated_at_utc >= created_at_utc`, and both timestamps are valid UTC ISO-8601 values.
9. Unknown optional fields are allowed only under `metadata` to preserve contract extensibility boundaries.

### Backward Compatibility Policy

1. Minor (`@1.x`) updates may add optional fields, clarify criteria semantics, and expand non-breaking reason-code vocabularies.
2. Minor updates must not remove required fields or reinterpret required field meanings (`input_set_id`, `input_items`, `artifact_state`, or lineage linkage semantics).
3. Major (`@2.0+`) updates are required for required-field additions/removals, breaking state-transition rule changes, or incompatible eligibility-policy semantics.
4. Producers and consumers must preserve unknown optional fields under `metadata` for forward compatibility across independently authored steps.
5. Deprecated fields must remain readable for at least one major-version window before retirement.

## Proposed Plan

1. Establish `C-003@1.0` ingestion semantics for canonical input references, normalization boundaries, and lineage-preserving validation outcomes.
2. Define artifact eligibility and state progression semantics that determine readiness, publication, and supersession behavior at planning level.
3. Publish schema, validation checks, and compatibility policy so `STEP-05` can consume input/artifact semantics without worker orchestration coupling.

## Definition Of Done

- [x] Define input ingestion and artifact eligibility planning semantics for `C-003@1.0`.
- [x] Produce one step plan document at `docs/plans/STEP-04_input_artifact_plan.md`.
- [x] Include explicit non-goals, assumption ledger, and contract change log.
- [x] Honor `draft_provisional` missing-contract policy via precedence-based source resolution.

## Validation Checks

1. PASS: All referenced assumptions are valid `A-*` IDs and constrained to the allowed list.
2. PASS: Required contracts are resolved via source precedence (`C-001@1.0` from producer output; `C-003@1.0` from seed plus template scaffolding).
3. PASS: Output artifact path matches `STEP-04` naming and deliverable requirements.
4. PASS: Forbidden dependency on worker orchestration internals is explicit.
5. PASS: Non-goals and assumption ledger are included and explicit.

## Non-Goals

- Define worker orchestration internals (queue topology, lease handling, dispatch/retry scheduling).
- Define method-specific specialization logic, clustering algorithm choices, or plugin registration internals.
- Redefine generic run/request lifecycle semantics owned by `C-002@1.0`.
- Specify infrastructure-specific storage backend implementation details.

## Failure And Rollback Behavior

- In `draft_provisional`, if a required contract is missing or ambiguous, proceed with best-available source and log challenged assumptions.
- In `finalize_gated`, if a required contract is missing or ambiguous, set status to `blocked` and request clarification.
- If a contract change is breaking, publish a new major version and retain old contract references.
- If overlap with another step is discovered, defer to producer ownership in the master step registry.

## Assumption Ledger

| ID | Statement | Status | Notes |
| --- | --- | --- | --- |
| `A-001` | All planning artifacts live under `docs/plans/` unless explicitly noted. | `accepted` | Output path and artifact boundary follow this location rule. |
| `A-002` | `docs/IMPLEMENTATION_PLAN.md` remains the implementation status source of truth. | `accepted` | This document defines planning semantics only, not implementation phase status. |
| `A-003` | Cross-step dependencies are expressed only through `C-*` contracts, not implementation details. | `accepted` | Upstream/downstream coupling is represented via `C-001@1.0` and `C-003@1.0`. |
| `A-004` | Each section plan can be authored in a fresh chat with only the step header and master doc. | `accepted` | The document is self-contained for independent downstream use. |
| `A-005` | Contract IDs are stable and append-only; breaking changes require a new major contract version. | `accepted` | Versioning policy enforces major increments for breaking changes. |
| `A-006` | Section plans must include explicit non-goals and forbidden dependencies. | `accepted` | Both sections are explicit and enforce scope boundaries. |
| `A-007` | Execution mode controls blocking: `draft_provisional` proceeds with fallback semantics; `finalize_gated` blocks on unresolved contracts. | `accepted` | Failure and rollback behavior mirrors mode policy exactly. |
| `A-009` | Default mode for new step runs is `draft_provisional` unless explicitly overridden in the step header. | `accepted` | This step runs in `draft_provisional` per starter definition. |

## Contract Change Log

| Contract ID | Change Type | Version Impact | Summary |
| --- | --- | --- | --- |
| `C-003@1.0` | `clarify` | `none` | Promoted from STEP-01 draft placeholder to active producer-defined contract with input ingestion and artifact eligibility planning semantics. |
