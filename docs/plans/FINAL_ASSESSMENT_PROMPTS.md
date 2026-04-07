# Final Assessment Prompt Pack

These prompts are for evaluating the completed planning set as a single system before implementation.

## 1) Integrated Plan Quality Review

```text
Review the full planning set as one system:
- docs/plans/MASTER_META_PLAN.md
- docs/plans/README.md
- docs/plans/templates/STEP_EXECUTION_HEADER.md
- docs/plans/templates/SECTION_PLAN_TEMPLATE.md
- docs/plans/templates/STEP_STARTERS.md
- docs/plans/STEP-01_master_bootstrap.md
- docs/plans/STEP-02_run_request_lifecycle.md
- docs/plans/STEP-03_method_plugin_contract.md
- docs/plans/STEP-04_input_artifact_plan.md
- docs/plans/STEP-05_clustering_specialization.md
- docs/plans/STEP-06_analysis_provenance.md
- docs/plans/STEP-07_panel_orchestration_ux.md
- docs/plans/STEP-08_cutover_risk_policy.md

Return:
1) contradictions,
2) missing dependencies,
3) ambiguous contracts,
4) top 10 fixes (severity-ordered).
```

## 2) Contract Consistency Audit (`C-001..C-006`)

```text
Audit contract consistency across all STEP-01..STEP-08 docs.
For each C-001..C-006, report:
- where it is defined,
- where it is consumed,
- any schema drift or semantic mismatch,
- required edits to make usage consistent.
Output as a table plus prioritized fix list.
```

## 3) Execution Backlog Synthesis

```text
Convert the plan set into one implementation backlog.
Output:
- ordered phases,
- tasks with acceptance criteria,
- dependency graph,
- test/validation gates,
- rollback checkpoints,
- suggested commit boundaries.
No new architecture; synthesize only from existing plan docs.
```

## 4) Red-Team Risk Review

```text
Red-team this plan set.
Assume implementation starts tomorrow.
Find failure modes likely to break delivery (process, scope, dependency, provenance, UX/perf assumptions).
Rank by impact x likelihood and propose concrete mitigations.
```

## 5) Go/No-Go Decision

```text
Give a Go/No-Go for implementation readiness.
If No-Go, provide minimum required edits to reach Go.
If Go, provide first 2 weeks of execution plan with objective milestones.
```
