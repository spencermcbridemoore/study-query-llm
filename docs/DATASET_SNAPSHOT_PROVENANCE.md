# Dataset Snapshot Provenance

Status: living  
Owner: data-maintainers  
Last reviewed: 2026-04-06

This document defines conventions for storing frozen sampled datasets using existing
`Group` + `CallArtifact` entities (no new DB table).

## Group Type

- `group_type = "dataset_snapshot"`

## Required Snapshot Metadata

Store the following keys in `groups.metadata_json`:

- `snapshot_name`: unique human-readable identifier
- `source_dataset`: source dataset name (e.g. `dbpedia`, `estela`)
- `sample_size`: integer sample size (e.g. `286`)
- `label_mode`: `labeled` or `unlabeled`
- `sampling_method`: method description (e.g. `seeded_random_filtered`)
- `sampling_seed`: integer seed (optional)
- `created_at`: ISO timestamp

Suggested optional keys:

- `has_ground_truth`: boolean
- `label_stats`: summary counts for labeled datasets
- `manifest_hash`: SHA-256 of JSON sample manifest for idempotency

## Manifest Artifact Convention

Attach one JSON artifact per snapshot group:

- `artifact_type = "dataset_snapshot_manifest"`
- `metadata_json.group_id = <snapshot_group_id>`
- `metadata_json.manifest_hash = <sha256>`
- URI path under `artifacts/<snapshot_group_id>/...`

Expected manifest payload shape:

```json
{
  "snapshot_name": "dbpedia_286_seed42_labeled",
  "entries": [
    {
      "position": 0,
      "source_id": 12345,
      "text": "example text",
      "label": 7
    }
  ]
}
```

For unlabeled snapshots (`estela`), omit `label` or set `label` to `null`.

## Run Linkage Convention

Each `clustering_run` should reference snapshot provenance:

- `metadata_json.dataset_snapshot_ids = [<snapshot_group_id>, ...]`

Optional explicit relationship:

- `GroupLink(parent=run_id, child=snapshot_group_id, link_type="depends_on")`

This enables future hierarchical grouping without schema changes.
