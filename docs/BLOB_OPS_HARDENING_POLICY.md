# Blob Ops Hardening Policy (Phase 2)

This document defines Phase 2 operational policy for blob-backed artifacts.

## 1) Environment and container routing policy

- Runtime lane must be explicit via `ARTIFACT_RUNTIME_ENV` (`dev`, `stage`, `prod`).
- Container is derived from `AZURE_STORAGE_CONTAINER`: for lanes `dev`/`stage`/`prod` the name is `{AZURE_STORAGE_CONTAINER}-{lane}` (e.g. `artifacts` → `artifacts-dev`, `artifacts-stage`, `artifacts-prod`). Optional per-lane overrides: `AZURE_STORAGE_CONTAINER_DEV`, `_STAGE`, `_PROD`.
- Non-prod lanes must not target prod-like containers unless
  `ARTIFACT_ALLOW_CROSS_ENV_CONTAINER=true` is intentionally set.

## 2) Authentication policy

- `ARTIFACT_AUTH_MODE=connection_string`
  - Allowed for local development and controlled ad-hoc runs.
  - Requires `AZURE_STORAGE_CONNECTION_STRING`.
- `ARTIFACT_AUTH_MODE=managed_identity`
  - Preferred for hosted lanes and required for hardened production posture.
  - Requires `AZURE_STORAGE_ACCOUNT_URL`.
  - Uses `AZURE_STORAGE_MANAGED_IDENTITY_CLIENT_ID` when set (user-assigned MI).

## 3) Strict mode and fallback policy

- `ARTIFACT_STORAGE_STRICT_MODE=true` enables hard-fail behavior:
  - local backend is disallowed
  - backend init errors do not silently fall back to local
- Strict mode is automatically expected for `stage` and `prod` lanes.

## 4) Reliability and integrity policy

- Retry/backoff is enabled for transient blob failures:
  - `AZURE_STORAGE_MAX_RETRIES`
  - `AZURE_STORAGE_RETRY_BACKOFF_SECONDS`
- Auth/permission failures are non-retryable and must fail fast.
- Upload verification is enabled by default (`AZURE_STORAGE_VERIFY_UPLOADS=true`).
- Artifact metadata must include integrity fields:
  - `sha256`
  - `byte_size`

## 5) Metadata governance contract

Every persisted artifact must include governance tags in `metadata_json`:

- `schema_version` (default `artifact.v1`)
- `governance_version` (default `blob_ops_phase2`)
- `storage_backend`
- `created_at`

## 6) Retention and versioning directives

- Data retention and lifecycle transitions are enforced at storage-account policy level.
- Application metadata must remain versioned for migration safety.
- Artifact format/schema changes require:
  - schema/version increment
  - migration note
  - backfill strategy for existing readers

## 7) Restore drill playbook (minimum)

Run at least once per lane before production sign-off:

1. Select sample artifacts across types (`sweep_results`, `embedding_matrix`, `metrics`).
2. Record artifact IDs, URIs, checksums, and expected byte sizes.
3. Simulate recovery by re-reading artifacts and validating integrity.
4. Verify downstream ingestion/readers consume restored artifacts successfully.
5. Capture evidence and sign-off from owner + reviewer.

## 8) Ownership and escalation

- Engineering owner: artifact pipeline maintainers.
- Platform owner: storage account policy, identity, and RBAC.
- Escalate immediately on:
  - lane/container routing ambiguity
  - auth bypass or fallback in hosted lanes
  - checksum/size mismatch with unknown cause
