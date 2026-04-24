# Subagent 6 — Doc / Contract Parity (raw)

Date: 2026-04-24
Scope: comparison of canonical documentation versus actual implementation behavior for `DATABASE_URL`, artifact backend selection, and Jetstream targeting.
Method: readonly explore subagent.

---

## A. Documents reviewed

- `docs/runbooks/README.md` — declares Jetstream Postgres as the canonical source of truth and outlines `DATABASE_URL` as the active connection string.
- `docs/runbooks/local_postgres_clone_for_dev.md` — describes the clone procedure and warns about destructive operations.
- `docs/design/clustering_pipeline_provenance.md` — defines artifact provenance contracts.
- `docs/design/database_safety_guardrails.md` — destructive DDL guardrail spec.
- `.env.example` — documents `JETSTREAM_DATABASE_URL`, `LOCAL_DOCKER_POSTGRES_URL`, `DATABASE_URL`, `ARTIFACT_*` variables.
- `AGENTS.md` — repository agent guidance, including database lane defaults.

---

## B. Contracts vs reality

| Documented contract | Reality | Discrepancy |
|---------------------|---------|-------------|
| `DATABASE_URL` is the active session URL; defaults to Jetstream when running production tools (`runbooks/README.md`). | Not enforced; any value is accepted and silently used (`db/_base_connection.py:140`). | **Drift**: docs imply Jetstream-by-default, code accepts arbitrary URLs without preflight. |
| `ARTIFACT_STORAGE_BACKEND=azure_blob` for any "real" run; local backend only for dev (`runbooks/README.md`). | `_resolve_default_backend` defaults to `local` when unset and silently falls back to local on azure errors when not strict (`artifact_service.py:93-138`). | **Drift**: docs treat azure as default; code defaults to local. |
| Destructive DDL on Jetstream is blocked (`docs/design/database_safety_guardrails.md`). | Implemented (`db/_base_connection.py:147-171`). | Matches. |
| Snapshots and analyze runs persist to canonical Jetstream when launched via runbooks. | Implementation persists to **whatever** `DATABASE_URL` resolves to; URI may be local. | **Drift**: provenance contracts assume blob URIs. |
| Provenance contracts (`docs/design/clustering_pipeline_provenance.md`) state that `result_ref` and `uri` always reference durable storage. | No code-level guarantee; columns accept any string (`models_v2.py:227, 433, 755`). | **Drift**: contract not enforced at schema level. |
| `.env.example` warns to keep `LOCAL_DOCKER_POSTGRES_URL` separate from Jetstream. | Honored by sync scripts; **ignored** by pipeline runners. | **Partial**: ops scripts comply, pipeline runners do not. |
| `AGENTS.md` instructs agents to use Jetstream by default and to avoid local writes that won't propagate. | No enforcement; agent must self-police. | **Conventional only**. |

---

## C. Specific docs that need updating

1. **`docs/runbooks/README.md`** — add an "Operational Lane" matrix that mirrors the proposed `WriteIntent` enum and references the preflight banner.
2. **`docs/design/clustering_pipeline_provenance.md`** — add explicit constraint: artifact URIs in canonical tables MUST be HTTPS blob URLs.
3. **`docs/design/database_safety_guardrails.md`** — extend to cover artifact-backend ↔ DB-target consistency, not just destructive DDL.
4. **`.env.example`** — promote `JETSTREAM_DATABASE_URL`, `LOCAL_DATABASE_URL`, `ARTIFACT_STORAGE_BACKEND` to required entries with comments referencing the new chokepoint.
5. **`AGENTS.md`** — codify the rule "no agent action may write canonical data unless preflight banner is observed."

---

## D. Cross-doc inconsistencies

- `runbooks/README.md` references "Jetstream is the source of truth" while `local_postgres_clone_for_dev.md` describes scenarios where engineers explicitly point pipelines at the local clone for QA. The **how-to-prevent-leaking** guidance is missing in both.
- The CLI `connect_jetstream` documentation suggests using the SSH tunnel for Jetstream queries but does not warn that other pipeline tools may not use the same target.
- `docs/design/clustering_pipeline_provenance.md` references `result_ref` lifecycle but never discusses local-path artifact URIs as a failure mode.

---

## E. Summary

Documentation broadly assumes that pipeline tools always converge on Jetstream + Azure Blob. Implementation does not enforce this; defaults frequently send writers to local storage with no warning. The audit's recommended fix (chokepoint + `WriteIntent` + preflight banner) closes the contract gap and gives docs a concrete mechanism to reference instead of relying on convention.
