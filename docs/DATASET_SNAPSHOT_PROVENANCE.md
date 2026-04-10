# Dataset Snapshot Provenance

Status: living  
Owner: data-maintainers  
Last reviewed: 2026-04-07

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
- Logical path pattern: `<snapshot_group_id>/snapshot_manifest/dataset_snapshot_manifest.json`

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

## BANK77 Bootstrap Convention

The BANK77 bootstrap flow is implemented in:

- `scripts/create_bank77_snapshot_and_embeddings.py`

Canonical source and contract:

- Source dataset: `mteb/banking77` (`train` + `test` concatenated in deterministic order)
- Snapshot label mode: `labeled`
- Stable source row IDs: `train:<index>` and `test:<index>`
- Expected label cardinality: `77` (halt if source schema changes)

Example manifest entry shape for BANK77:

```json
{
  "position": 12,
  "source_id": "train:12",
  "split": "train",
  "text": "Why was my transfer reversed?",
  "label": 43,
  "intent": "beneficiary_not_allowed"
}
```

Embedding artifacts produced by the bootstrap:

- Full matrix (cold reuse path):
  - `artifact_type = "embedding_matrix"`
  - Shape: `(N, d)` where `N` is full BANK77 row count.
- Intent means (hot immediate-use path):
  - `artifact_type = "embedding_matrix"`
  - Shape: `(77, d)` in deterministic row order by `label_id` ascending.
- Deterministic row mapping + counts:
  - `artifact_type = "metrics"`
  - `metadata_json.step_name = "bank77_intent_label_mapping"`
  - `metadata_json.mapping_type = "bank77_intent_means_label_mapping"`
  - Payload includes `ordered_label_ids`, `ordered_intents`, and per-row `count`.

Lineage/linkage conventions for BANK77 embeddings:

- `GroupLink(parent=<embedding_batch>, child=<dataset_snapshot>, link_type="depends_on")`
  - relation metadata: `embedding_source_snapshot`
- `GroupLink(parent=<means_embedding_batch>, child=<full_embedding_batch>, link_type="depends_on")`
  - relation metadata: `derived_from_full_embedding_matrix`

Verification/readback path:

- `--verify-only` performs snapshot/full/means/link integrity checks.
- `read_means_payload(...)` reads means + mapping artifacts without loading the full matrix.

## Run Linkage Convention

Each `clustering_run` should reference snapshot provenance:

- `metadata_json.dataset_snapshot_ids = [<snapshot_group_id>, ...]`
- IDs should be normalized as sorted unique integers before persistence.

Primary snapshot pointer for unified execution provenance:

- `provenanced_runs.input_snapshot_group_id = <primary_snapshot_group_id>`
- Primary selection rule: first value from normalized `dataset_snapshot_ids`.

Optional explicit relationship:

- `GroupLink(parent=run_id, child=snapshot_group_id, link_type="depends_on")`

Read-time compatibility fallback:

- If legacy run metadata does not contain `dataset_snapshot_ids`, unified execution
  compatibility mapping may infer a primary snapshot from `depends_on` links.

Consistency expectations:

- If `dataset_snapshot_ids` is present, `depends_on` links should contain the same
  snapshot IDs.
- Validation/backfill script (`scripts/validate_and_backfill_run_snapshots.py`) reports
  mismatch classes (`metadata_without_link`, `link_without_metadata`, and ID disagreement)
  in dry-run mode.

This enables future hierarchical grouping without schema changes.
