# Runbook: record dataset download (layer 0)

Status: living  
Owner: data-maintainers  
Last reviewed: 2026-04-10

## References

- Contract and schema: [docs/DATASET_ACQUISITION_LAYER0.md](../DATASET_ACQUISITION_LAYER0.md)

## Local filesystem bundle

```bash
python scripts/record_dataset_download.py --dataset ausem --output-dir ./data/acquisitions/ausem
```

Produces `acquisition.json` and `files/...` under the output directory.

## Dry run (print manifest)

```bash
python scripts/record_dataset_download.py --dataset ausem --dry-run
```

## Persist to Postgres + Azure Blob

Prerequisites:

- `DATABASE_URL` set
- `ARTIFACT_STORAGE_BACKEND=azure_blob` and Azure variables per `.env.example`
- Verify connectivity: `python scripts/check_azure_blob_storage.py`

```bash
python scripts/record_dataset_download.py --dataset ausem --output-dir ./data/acquisitions/ausem --persist-db --dataset-group-name acquire_ausem
```

You may omit `--output-dir` if you only want database artifacts (blobs + group metadata).
