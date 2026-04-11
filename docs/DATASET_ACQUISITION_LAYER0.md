# Dataset acquisition (layer 0)

Status: living  
Owner: data-maintainers  
Last reviewed: 2026-04-10

## Purpose

**Layer 0** records **download provenance** before any parsing or normalized snapshots:

- Exact **URLs** used (pinned where possible, e.g. `raw.githubusercontent.com/.../{git_ref}/...`)
- **UTC timestamp** (`acquired_at`)
- Per-file **SHA-256** and **byte size**
- Optional **runner** metadata (script path, optional `git rev-parse` for this repo)

Layer 1 (separate work) is a **`dataset_snapshot`** + `dataset_snapshot_manifest` as described in [DATASET_SNAPSHOT_PROVENANCE.md](DATASET_SNAPSHOT_PROVENANCE.md).

## Manifest schema (`acquisition.json`)

Top-level keys:

| Key | Description |
|-----|-------------|
| `schema_version` | String, currently `1.0` |
| `dataset_slug` | Short id (e.g. `ausem`) |
| `acquired_at` | ISO-8601 UTC |
| `source` | Dataset-specific dict (e.g. `kind: github_raw`, `organization`, `repository`, `git_ref`) |
| `runner` | `script`, optional `git_commit`, optional extras |
| `files` | Sorted by `relative_path`; each item has `relative_path`, `url`, `sha256`, `byte_size` |

Canonical JSON for hashing uses `json.dumps(..., sort_keys=True, ensure_ascii=False)` via `acquisition_manifest_sha256()` in `study_query_llm.datasets.acquisition`.

## On-disk bundle layout

When using [scripts/record_dataset_download.py](../scripts/record_dataset_download.py) with `--output-dir`:

```text
<output-dir>/
  acquisition.json
  files/
    <relative_path>/...   # mirrors logical paths (e.g. Student_Explanations/problem1.csv)
```

## Database + Azure (optional)

With `--persist-db`, the script:

1. Requires `DATABASE_URL` and `ARTIFACT_STORAGE_BACKEND=azure_blob`.
2. Creates a **`dataset`** group (`ProvenanceService.create_dataset_group`) with summary metadata (`manifest_sha256`, `file_count`, etc.).
3. Stores each file as `CallArtifact` with `artifact_type=dataset_acquisition_file` via `ArtifactService.store_group_blob_artifact`.
4. Stores the manifest JSON as `artifact_type=dataset_acquisition_manifest`.

See [docs/runbooks/record_dataset_download.md](runbooks/record_dataset_download.md) for commands.

## Code locations

- Library: `src/study_query_llm/datasets/acquisition.py`
- Pinned sources: `src/study_query_llm/datasets/source_specs/` (registry: `ACQUIRE_REGISTRY`)
- CLI: `scripts/record_dataset_download.py`

## Supported dataset slugs

| Slug | Source | Notes |
|------|--------|--------|
| `ausem` | [tufts-ml/AuSeM](https://github.com/tufts-ml/AuSeM) | Four `Student_Explanations/problem*.csv` at a pinned git ref ([`source_specs/ausem.py`](../src/study_query_llm/datasets/source_specs/ausem.py)). |
| `sources_uncertainty_qc` | [Zenodo 16912394](https://zenodo.org/records/16912394) (DOI [10.5281/zenodo.16912394](https://doi.org/10.5281/zenodo.16912394)) | Single `sources_v2.xlsx` — “Sources of Uncertainty in Quantum and Classical Measurement” public dataset ([`source_specs/sources_uncertainty_zenodo.py`](../src/study_query_llm/datasets/source_specs/sources_uncertainty_zenodo.py)). |
| `semeval2013_sra_5way` | [ashudeep/Student-Response-Analysis](https://github.com/ashudeep/Student-Response-Analysis) | **Community mirror** of SemEval-2013 Task 7 five-way **gold** text files only (pinned commit); not the official LDC release ([`source_specs/semeval2013_sra_5way.py`](../src/study_query_llm/datasets/source_specs/semeval2013_sra_5way.py)). |

Additional slugs can be registered in `source_specs/registry.py` using the same pattern.
