# STEP-06 Analysis/Provenance Plan

## Metadata

- Step ID: `STEP-06`
- Plan title: `Analysis and provenance contract definition`
- Status: `active`
- Execution mode: `draft_provisional`
- Last updated: `2026-04-07`
- Source meta-plan: `docs/plans/MASTER_META_PLAN.md`

## Objective

Define `C-005@1.0` as the canonical analysis/provenance planning contract so downstream orchestration can consume reproducible lineage and evidence semantics without redefining request lifecycle ownership (`C-002@1.0`) or method specialization ownership (`C-004@1.0`). This step formalizes provenance boundaries, schema requirements, validation rules, and compatibility constraints while explicitly avoiding UI-only assumptions.

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
| `C-002@1.0` | `1.0` | Provides canonical request/run lifecycle semantics and identifiers (`request_id`, optional `run_id`, transition context) that provenance records must reference without redefining lifecycle behavior. |
| `C-004@1.0` | `1.0` | Provides canonical method specialization identity/version and provenance requirement anchors that analysis provenance must satisfy and preserve for downstream orchestration. |

## Contract Source Resolution

| Contract ID | Selected Source | Why This Source Was Chosen |
| --- | --- | --- |
| `C-002@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-02_run_request_lifecycle.md` is the highest-priority available source. |
| `C-004@1.0` | `producer_step_output` | No standalone contract file exists under `docs/plans/contracts/`; `docs/plans/STEP-03_method_plugin_contract.md` is the highest-priority available source, with clustering-profile clarifications from `docs/plans/STEP-05_clustering_specialization.md` where relevant. |

## Forbidden Dependencies

- No UI-only assumptions.
- No implementation details from sibling steps.
- No hidden assumptions outside listed `A-*`.
- No references to non-canonical contract IDs.

## Output Artifacts

- Primary output document: `docs/plans/STEP-06_analysis_provenance.md`
- Produced/updated contract(s): `C-005@1.0` (`active`)
- Optional supporting appendix: _None_

## C-005@1.0 Contract Definition

### Name

- `AnalysisProvenanceContract`

### Purpose

- Define canonical provenance semantics linking request lifecycle context (`C-002@1.0`) and method specialization context (`C-004@1.0`) to analysis outputs.
- Standardize lineage grouping, reproducibility evidence, and audit-friendly metadata required for downstream orchestration in `STEP-07`.

### Producer Step(s)

- `STEP-06`

### Consumer Step(s)

- `STEP-07`

### Composition With Upstream Contracts

| Upstream Contract | Composition Rule | C-005 Constraint |
| --- | --- | --- |
| `C-002@1.0` | Provenance records reference lifecycle entities using canonical request/run identifiers and state context. | `C-005@1.0` may reference lifecycle context, but must not add, remove, or reinterpret lifecycle states/transition edges owned by `C-002@1.0`. |
| `C-004@1.0` | Provenance records bind to canonical method identity/version and declared method provenance requirements. | `C-005@1.0` must preserve method-declared provenance keys and must not redefine method parameter/result schema ownership from `C-004@1.0`. |

### Contract Semantics And Boundaries

Boundary invariants:

1. Provenance records are append-only lineage snapshots; once a record is `published`, only backward-compatible metadata enrichment is allowed.
2. Each provenance record binds one analysis reference to one request identity and one method identity/version pair.
3. Lineage edges must explicitly connect derived analysis outputs to their source input and artifact references.
4. Provenance grouping must be deterministic via `provenance_group_key` and `grouping_version`.
5. Reproducibility evidence must include enough deterministic fingerprint material to support downstream replay/audit planning.
6. Contract semantics remain orchestration-agnostic and must not encode UI rendering behavior.

Boundary table:

| Area | In Scope For `C-005@1.0` | Out Of Scope For `C-005@1.0` |
| --- | --- | --- |
| Lifecycle linkage | Referencing canonical request/run lifecycle context and anchors | Defining or changing lifecycle state semantics owned by `C-002@1.0` |
| Method linkage | Referencing canonical method identity/version and provenance requirements | Defining plugin internals or parameter/result schema ownership from `C-004@1.0` |
| Lineage modeling | Declaring provenance groups, lineage edges, and evidence expectations | Implementing storage/query engines for lineage graphs |
| Analysis evidence | Defining analysis reference identity and reproducibility fingerprint fields | Defining panel visualization behavior for analysis displays |
| Policy metadata | Declaring planning-level retention/redaction metadata constraints | Defining infrastructure-specific retention implementations |

### Schema

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `contract_id` | `string` | Must equal `C-005@1.0`. |
| `provenance_record_id` | `string` | Stable identifier for the provenance record entity. |
| `analysis_ref_id` | `string` | Stable identifier for the analysis output or analysis bundle represented by the record. |
| `request_id` | `string` | Canonical request identifier from `C-002@1.0`. |
| `method_ref` | `string` | Canonical method identifier from `C-004@1.0`. |
| `method_version` | `string` | Method behavior/version token from `C-004@1.0`. |
| `provenance_state` | `enum` | One of `drafted`, `assembled`, `verified`, `published`, `superseded`. |
| `provenance_group_key` | `string` | Deterministic grouping key for related provenance records. |
| `grouping_version` | `string` | Version marker for grouping semantics used to compute `provenance_group_key`. |
| `source_input_refs` | `array[string]` | Canonical input references contributing to the analysis record. |
| `source_artifact_refs` | `array[string]` | Canonical artifact references consumed/derived by the analysis record. |
| `lineage_edges` | `array[object]` | Directed lineage relationships among request, method, input, artifact, and analysis entities. |
| `reproducibility_fingerprint` | `object` | Deterministic hashes/signatures needed for replayability checks. |
| `evidence_bundle` | `object` | Required evidence descriptors proving provenance completeness for the record. |
| `created_at_utc` | `datetime` | Record creation timestamp in UTC ISO-8601 format. |
| `updated_at_utc` | `datetime` | Last mutation timestamp in UTC ISO-8601 format. |

Optional fields:

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | `string` | Optional execution identifier from `C-002@1.0` when runtime assignment exists. |
| `lifecycle_state_snapshot` | `string` | Optional lifecycle state captured from `C-002@1.0` at provenance materialization time. |
| `transition_anchor` | `object` | Optional pointer into lifecycle transition evidence (for example transition index/id). |
| `quality_metrics` | `object` | Optional quality diagnostics associated with the analysis record. |
| `retention_policy` | `object` | Optional planning-level retention policy metadata for provenance material. |
| `redaction_policy` | `object` | Optional planning-level redaction/masking metadata for sensitive evidence. |
| `supersedes_provenance_record_id` | `string` | Optional pointer to prior record when supersession occurs. |
| `external_reference_map` | `object` | Optional links to external reports/datasets associated with the record. |
| `metadata` | `object` | Backward-compatible extension container for non-breaking annotations. |

### Validation Checks

1. `contract_id` equals `C-005@1.0`.
2. `provenance_record_id`, `analysis_ref_id`, `request_id`, `method_ref`, and `method_version` are non-empty and stable within active planning scope.
3. `method_ref` + `method_version` must resolve to a valid method definition governed by `C-004@1.0`.
4. `provenance_state` is a valid enum member and transitions are forward-only (`drafted -> assembled -> verified -> published -> superseded`).
5. At least one lineage source reference must be present (`source_input_refs` or `source_artifact_refs`) for `assembled` and later states.
6. `published` provenance records must include non-empty `lineage_edges`, `reproducibility_fingerprint`, and `evidence_bundle`.
7. If `run_id` is present, it must be compatible with the referenced request lifecycle entity in `C-002@1.0`.
8. `lineage_edges` must reference declared nodes/refs and must not form cyclic ancestry for a single derived analysis entity.
9. `created_at_utc` and `updated_at_utc` must be valid UTC ISO-8601 timestamps and `updated_at_utc >= created_at_utc`.
10. Unknown optional fields are allowed only under `metadata` to preserve extensibility boundaries.

### Backward Compatibility Policy

1. Minor (`@1.x`) updates may add optional fields, clarify lineage/evidence semantics, and expand backward-compatible provenance metadata.
2. Minor updates must not remove required fields or reinterpret existing required-field meanings.
3. Major (`@2.0+`) updates are required for required-field additions/removals, breaking provenance-state transition semantics, or incompatible lineage/evidence interpretations.
4. Producers and consumers must preserve unknown optional fields in `metadata` for forward compatibility across independently authored steps.
5. Deprecated provenance fields must remain readable for at least one major-version window before retirement.

## Proposed Plan

1. Compose provenance semantics from `C-002@1.0` lifecycle identifiers and `C-004@1.0` method specialization identity/requirements without changing upstream ownership.
2. Define `C-005@1.0` schema, validation checks, and compatibility constraints for deterministic lineage and reproducibility planning.
3. Publish explicit scope boundaries and non-goals so `STEP-07` can consume `C-005@1.0` without UI or implementation-detail coupling.

## Definition Of Done

- [x] Define analysis/provenance semantics and ownership boundaries for `C-005@1.0`.
- [x] Define `C-005@1.0` schema, validation checks, and compatibility constraints.
- [x] Keep forbidden dependency constraints explicit, including no UI-only assumptions.
- [x] Produce one step plan document at `docs/plans/STEP-06_analysis_provenance.md` with assumption ledger and non-goals.

## Validation Checks

1. PASS: All referenced assumptions are valid `A-*` IDs and constrained to the allowed list.
2. PASS: Required input contracts are resolved via source precedence from producer outputs (`STEP-02` and `STEP-03`), with no standalone contract override.
3. PASS: Output artifact path matches `STEP-06` naming and deliverable requirements.
4. PASS: Forbidden dependency against UI-only assumptions is explicit.
5. PASS: Non-goals, assumption ledger, and contract change log are included and explicit.

## Non-Goals

- Define panel UX layout, visualization behavior, or interaction flows for provenance/analysis displays.
- Redefine lifecycle state ownership, transition rules, or terminal semantics owned by `C-002@1.0`.
- Redefine method plugin parameter/result schema ownership or plugin registration semantics owned by `C-004@1.0`.
- Define algorithm implementation internals, worker orchestration mechanics, or infrastructure storage details.

## Failure And Rollback Behavior

- In `draft_provisional`, if a required contract is missing or ambiguous, proceed with best-available source and log challenged assumptions.
- In `finalize_gated`, if a required contract is missing or ambiguous, set status to `blocked` and request clarification.
- If a contract change is breaking, publish a new major version and retain old contract references.
- If overlap with another step is discovered, defer to producer ownership in the master step registry.

## Assumption Ledger

| ID | Statement | Status | Notes |
| --- | --- | --- | --- |
| `A-001` | All planning artifacts live under `docs/plans/` unless explicitly noted. | `accepted` | Output path and scope boundary remain within `docs/plans/`. |
| `A-002` | `docs/IMPLEMENTATION_PLAN.md` remains the implementation status source of truth. | `accepted` | This document defines planning semantics only, not implementation status tracking. |
| `A-003` | Cross-step dependencies are expressed only through `C-*` contracts, not implementation details. | `accepted` | Upstream/downstream coupling is represented only through `C-002@1.0`, `C-004@1.0`, and `C-005@1.0`. |
| `A-004` | Each section plan can be authored in a fresh chat with only the step header and master doc. | `accepted` | The plan is self-contained and executable without sibling-step implementation details. |
| `A-005` | Contract IDs are stable and append-only; breaking changes require a new major contract version. | `accepted` | Compatibility policy enforces major-version changes for breaking semantics. |
| `A-006` | Section plans must include explicit non-goals and forbidden dependencies. | `accepted` | Non-goals and forbidden dependencies are explicit and scope-limiting. |
| `A-007` | Execution mode controls blocking: `draft_provisional` proceeds with fallback semantics; `finalize_gated` blocks on unresolved contracts. | `accepted` | Failure and rollback behavior mirrors mode policy exactly. |
| `A-009` | Default mode for new step runs is `draft_provisional` unless explicitly overridden in the step header. | `accepted` | This step executes in `draft_provisional` per the STEP-06 starter input. |

## Contract Change Log

| Contract ID | Change Type | Version Impact | Summary |
| --- | --- | --- | --- |
| `C-005@1.0` | `clarify` | `none` | Promoted from STEP-01 draft placeholder to active producer-defined contract with analysis/provenance semantics, validation checks, and compatibility constraints. |
