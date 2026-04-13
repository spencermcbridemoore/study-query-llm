# Runbook: record dataset download (layer 0)

Status: living  
Owner: data-maintainers  
Last reviewed: 2026-04-10

## Canonical References

- Contract and schema: [`docs/DATASET_ACQUISITION_LAYER0.md`](../DATASET_ACQUISITION_LAYER0.md)
- Snapshot/provenance relationship: [`docs/DATASET_SNAPSHOT_PROVENANCE.md`](../DATASET_SNAPSHOT_PROVENANCE.md)
- DB URL policy and source-of-truth rules: [`docs/runbooks/README.md`](README.md)

## Command Patterns

### 1) Local files only (no DB writes)

```bash
python scripts/record_dataset_download.py --dataset ausem --output-dir ./data/acquisitions/ausem
```

Produces `acquisition.json` plus `files/...` under the output directory.

### 2) Dry-run (manifest only)

```bash
python scripts/record_dataset_download.py --dataset ausem --dry-run
```

### 3) Persist Layer-0 artifacts to DB + blob storage

Prerequisites:

- `ARTIFACT_STORAGE_BACKEND=azure_blob`
- Azure variables from `.env.example`
- `DATABASE_URL` points to the intended write target (Jetstream by default policy)
- Optional safety check: `python scripts/check_azure_blob_storage.py`
- Guardrails: `--persist-db` refuses `LOCAL_DATABASE_URL` and non-`JETSTREAM_DATABASE_URL` targets unless explicit override flags are passed.

```bash
python scripts/record_dataset_download.py --dataset ausem --output-dir ./data/acquisitions/ausem --persist-db --dataset-group-name acquire_ausem --database-url "$JETSTREAM_DATABASE_URL"
```

For DB-only artifact persistence, omit `--output-dir`.
