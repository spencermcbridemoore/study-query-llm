# Root Scripts Triage Matrix (Preflight)

Status: draft  
Owner: scripts-cleanup pass  
Last reviewed: 2026-05-03

## Scope

- Inventory target: root `scripts/*.py` only (78 files).
- Classification options: `keep-root` | `move-to-scripts-living` | `move-to-scripts-history` | `move-to-scripts-deprecated` | `delete` | `unclear-needs-investigation`.
- Evidence sources used: `.github/workflows/`, `src/`, `tests/`, living docs/runbooks, and sibling scripts.
- Restricted historical docs were not used as decision authority.

## Summary Counts

| action | count |
|---|---:|
| keep-root | 71 |
| move-to-scripts-living | 4 |
| move-to-scripts-history | 3 |
| move-to-scripts-deprecated | 0 |
| delete | 0 |
| unclear-needs-investigation | 0 |
| total | 78 |

## Matrix

| script | recommended_action | confidence | ci_refs | src_refs | tests_refs | docs_refs | other_script_refs | rationale |
|---|---|---|---|---|---|---|---|---|
| `analyze_dataset_lengths.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/analyze_dataset_lengths.py` | Path-stable deprecation wrapper to `scripts/deprecated`/`scripts/history` per lane policy. |
| `analyze_dbpedia_character_length_grid.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/analyze_dbpedia_character_length_grid.py` | Same deprecation wrapper contract as other moved no-PCA analysis scripts. |
| `analyze_estela_lengths.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/analyze_estela_lengths.py` | Same deprecation wrapper contract. |
| `archive_defective_data.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/sync_from_online.py`; `scripts/init_local_db.py`; `scripts/check_db_lane_policy.py` | Data-ops utility cross-linked from peer DB scripts and lane-policy allowlist. |
| `archive_mcq_artifact_blobs.py` | `keep-root` | high | - | - | `tests/test_scripts/test_archive_mcq_artifact_blobs.py` | `scripts/README.md` | `scripts/backup_mcq_db_to_json.py` | High-signal entrypoint in README with dedicated tests and backup coupling. |
| `archive_pre_fix_runs.py` | `keep-root` | high | - | - | - | `scripts/README.md`; `docs/living/CURRENT_STATE.md` | `scripts/history/sweep_recovery/archive_pre_fix_runs.py` | Root wrapper preserved for sweep recovery; living docs and README call this out. |
| `audit_last_partial_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | - | Listed in README high-signal sweep triage entrypoints. |
| `audit_mcq_method_definitions.py` | `keep-root` | medium | - | - | - | - | - | Read-only MCQ method-schema drift audit tied to canonical config builders; keep as a root operator diagnostic. |
| `backfill_run_fingerprints.py` | `keep-root` | high | - | - | - | `docs/living/CURRENT_STATE.md` | - | Canonical fingerprint backfill entrypoint cited in living current-state. |
| `backup_jetstream_full_state.py` | `keep-root` | high | - | - | `tests/test_scripts/test_backup_jetstream_full_state.py` | `docs/living/CURRENT_STATE.md`; `scripts/README.md` | - | Living doc + README high-signal + tested backup orchestration. |
| `backup_mcq_db_to_json.py` | `keep-root` | high | - | - | `tests/test_scripts/test_backup_mcq_db_to_json.py` | `scripts/README.md` | `scripts/archive_mcq_artifact_blobs.py` | README high-signal and tested; archive script consumes its export. |
| `build_embedding_cache_30k.py` | `move-to-scripts-living` | medium | - | - | - | - | - | Standalone cache builder with no CI/src/tests anchors; good living-lane candidate with wrapper. |
| `check_active_workers.py` | `keep-root` | high | - | - | `tests/test_scripts/test_check_active_workers.py` | `scripts/README.md` | `scripts/check_db_lane_policy.py` | README high-signal + tests; named in lane-policy allowlist. |
| `check_azure_blob_storage.py` | `keep-root` | high | - | - | - | `scripts/README.md` | - | README high-signal entrypoint. |
| `check_call_artifacts_uri_constraint.py` | `keep-root` | high | - | - | - | `docs/runbooks/README.md`; `scripts/README.md` | `scripts/check_db_lane_policy.py` | Runbook command + README DB matrix + lane-policy allowlist. |
| `check_db_lane_policy.py` | `keep-root` | high | `.github/workflows/persistence-contract.yml` | - | `tests/test_scripts/test_check_db_lane_policy.py` | `docs/living/CURRENT_STATE.md`; `scripts/README.md` | - | CI-gated static lane check with tests and living-doc citation. |
| `check_living_docs_drift.py` | `keep-root` | high | `.github/workflows/living-docs-drift.yml` | - | `tests/test_governance/test_living_docs_governance.py` | `docs/STANDING_ORDERS.md`; `scripts/README.md` | `scripts/internal/living_docs_governance.py` | CI hard gate with governance coupling and tests. |
| `check_orchestration_jobs.py` | `keep-root` | high | - | - | `tests/test_scripts/test_check_orchestration_jobs_cli.py` | `scripts/README.md` | - | README high-signal with CLI-focused tests. |
| `check_persistence_contract.py` | `keep-root` | high | `.github/workflows/persistence-contract.yml` | - | `tests/pipeline/test_lint.py`; `tests/pipeline/test_acquire_snapshot_chain.py` | `docs/living/ARCHITECTURE_CURRENT.md`; `docs/living/CURRENT_STATE.md`; `docs/DATA_PIPELINE.md`; `scripts/README.md` | - | CI + pipeline tests + architecture/data-pipeline living citations. |
| `check_raw_calls_uri_sentinel.py` | `keep-root` | high | - | - | `tests/test_scripts/test_check_raw_calls_uri_sentinel.py` | `docs/living/CURRENT_STATE.md`; `scripts/README.md` | `scripts/check_db_lane_policy.py` | Living docs + README DB matrix + tests + lane-policy allowlist. |
| `check_run_groups.py` | `keep-root` | high | - | - | - | `scripts/README.md` | - | README high-signal sweep triage companion. |
| `check_sweep_requests.py` | `keep-root` | high | - | - | - | `scripts/README.md` | - | README high-signal sweep triage entrypoint. |
| `create_bank77_contrast_snapshots.py` | `keep-root` | medium | - | - | - | - | - | Active BANK77 contrast-snapshot materialization utility with recent canonical naming/validation work; keep at root as an operator entrypoint. |
| `db_target_guardrails.py` | `keep-root` | high | - | - | `tests/test_scripts/test_db_target_guardrails.py` | `docs/runbooks/README.md` | `scripts/check_db_lane_policy.py`; `scripts/restore_pg_dump_to_local_docker.py`; `scripts/sync_from_online.py`; `scripts/purge_dataset_acquisition.py` | Shared guardrail module with tests and multiple DB-script imports. |
| `docker_smoke.py` | `keep-root` | high | `.github/workflows/docker-smoke.yml` | - | - | `scripts/README.md` | - | CI workflow invokes it; README high-signal. |
| `download_embedding_models.py` | `move-to-scripts-living` | medium | - | - | - | - | - | Model-prep utility with no CI/tests anchors; living-lane candidate with wrapper. |
| `download_summarizer_models.py` | `move-to-scripts-living` | medium | - | - | - | - | - | Model-prep utility; same living-lane rationale as embedding download helper. |
| `dump_postgres_for_jetstream_migration.py` | `keep-root` | high | - | - | `tests/test_scripts/test_dump_postgres_migration_cli.py` | `docs/runbooks/README.md`; `scripts/README.md` | `scripts/restore_pg_dump_to_local_docker.py`; `scripts/upload_jetstream_pg_dump_to_blob.py` | Runbook + README matrix + tests + peer dump/restore/upload coupling. |
| `export_mcq_sweep_option_counts_db.py` | `keep-root` | medium | - | - | - | - | `scripts/run_openrouter_mcq_sweep_until_done.py` | Chained from OpenRouter automation script; no living-doc anchor in current scan. |
| `export_mcq_sweep_stats.py` | `move-to-scripts-history` | medium | - | - | - | - | - | Experimental-results CSV exporter with no living-doc/CI/test anchors; better aligned with historical analysis lane. |
| `fill_snapshot_embeddings_from_baseline.py` | `move-to-scripts-living` | medium | - | - | - | - | `scripts/probe_snapshot_long_text_openrouter.py` | Only sibling-script coupling found; relocate-with-wrapper candidate. |
| `ingest_mcq_probe_json_to_sweep_db.py` | `keep-root` | medium | - | - | - | - | - | Write-path MCQ ingest utility without living-doc/test anchors; keep Tier B pending coverage/docs. |
| `ingest_sweep_to_db.py` | `keep-root` | high | - | - | `tests/test_scripts/test_ingest_sweep_to_db.py` | `scripts/README.md` | `scripts/reconcile_last_partial_sweep.py` | README high-signal + strong tests + reconcile-script invocation. |
| `init_local_db.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/sync_from_online.py`; `scripts/archive_defective_data.py`; `scripts/history/sweep_recovery/archive_pre_fix_runs.py` | Operator bootstrap referenced by DB/sync scripts and README matrix. |
| `label_pre_fix_runs.py` | `keep-root` | high | - | - | - | `scripts/README.md`; `docs/living/CURRENT_STATE.md` | `scripts/history/sweep_recovery/label_pre_fix_runs.py` | Preserved sweep-recovery wrapper per README and current-state docs. |
| `migrate_v1_to_v2.py` | `keep-root` | high | - | - | `tests/test_governance/test_living_docs_governance.py` | `scripts/README.md` | `scripts/deprecated/migrate_v1_to_v2.py` | README documents wrapper to deprecated implementation. |
| `pca_kllmeans_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/pca_kllmeans_sweep.py` | Historical-name wrapper explicitly documented in lane policy. |
| `plot_no_pca_50runs.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/plot_no_pca_50runs.py` | Deprecation wrapper consistent with move-set policy. |
| `plot_no_pca_multi_embedding.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/plot_no_pca_multi_embedding.py` | Deprecation wrapper consistent with move-set policy. |
| `probe_embedding_limits_live.py` | `keep-root` | high | - | - | `tests/test_scripts/test_probe_embedding_limits_live.py` | - | - | Dedicated tests give confidence despite sparse living-doc mentions. |
| `probe_postgres_inventory.py` | `keep-root` | high | - | - | - | `docs/runbooks/README.md`; `scripts/README.md` | `scripts/check_db_lane_policy.py` | Runbook quick probe + README DB matrix + lane-policy allowlist. |
| `probe_snapshot_long_text_openrouter.py` | `move-to-scripts-history` | high | - | - | - | - | - | One-off OpenRouter long-text probe (snapshot-ID/defaults oriented) with no active integration anchors; classify as historical diagnostic tooling. |
| `purge_dataset_acquisition.py` | `keep-root` | high | - | - | `tests/test_scripts/test_purge_dataset_acquisition_guardrails.py` | `docs/runbooks/README.md`; `scripts/README.md` | `scripts/check_db_lane_policy.py` | Destructive op with guardrail tests and runbook/README matrix placement. |
| `reconcile_last_partial_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/ingest_sweep_to_db.py` | README high-signal and invokes ingest helper as subprocess. |
| `regenerate_p0_baseline.py` | `keep-root` | high | `.github/workflows/persistence-contract.yml` | - | - | `docs/living/CURRENT_STATE.md`; `scripts/README.md` | - | CI `--check` step plus living-doc PR0 fixture narrative. |
| `register_clustering_methods.py` | `keep-root` | high | - | - | - | `docs/living/CURRENT_STATE.md`; `docs/living/METHOD_RECIPES.md`; `docs/DESIGN_FLAWS.md`; `scripts/README.md` | - | Living docs bind clustering registration to this operator entrypoint. |
| `register_text_classification_methods.py` | `keep-root` | high | - | - | - | `docs/STANDING_ORDERS.md`; `scripts/README.md` | - | Standing orders explicitly require script-family method registration. |
| `remediate_call_artifacts_to_blob.py` | `keep-root` | high | - | - | `tests/test_scripts/test_remediate_call_artifacts_to_blob.py` | `docs/living/CURRENT_STATE.md`; `docs/runbooks/README.md`; `scripts/README.md` | - | Living + runbook + README + tests. |
| `report_layer0_dataset_stats.py` | `keep-root` | high | - | - | `tests/test_scripts/test_report_layer0_dataset_stats.py` | - | - | Tested reporting utility; no active lane-move pressure. |
| `restore_pg_dump_to_local_docker.py` | `keep-root` | high | - | - | `tests/test_scripts/test_restore_pg_dump_guardrails.py` | `docs/runbooks/README.md`; `scripts/README.md` | `scripts/dump_postgres_for_jetstream_migration.py`; `scripts/sanity_check_database_url.py` | Runbook + README + tests + dump/sanity references. |
| `run_300_bigrun_sweep.py` | `keep-root` | high | - | - | `tests/test_scripts/test_run_300_bigrun_sweep_request_mode.py` | `scripts/README.md` | - | README high-signal + dedicated sweep-mode tests. |
| `run_bank77_pipeline.py` | `keep-root` | high | - | - | `tests/pipeline/test_bank77_pipeline_script_mapping.py` | `docs/living/ARCHITECTURE_CURRENT.md`; `docs/living/CURRENT_STATE.md`; `docs/DATA_PIPELINE.md`; `docs/runbooks/README.md`; `scripts/README.md` | `scripts/purge_dataset_acquisition.py` | Canonical five-stage operator entrypoint with living-doc binding and tests. |
| `run_cached_job_supervisor.py` | `keep-root` | high | - | `src/study_query_llm/services/jobs/runtime_supervisors.py` | - | `scripts/README.md` | - | Compatibility supervisor path tied to runtime supervisor module. |
| `run_custom_full_categories_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/run_custom_full_categories_sweep.py` | Deprecation wrapper per move-set policy. |
| `run_experimental_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/run_experimental_sweep.py` | Deprecation wrapper per move-set policy. |
| `run_langgraph_job_worker.py` | `keep-root` | high | - | - | `tests/test_scripts/test_run_langgraph_job_worker.py` | `scripts/README.md` | - | README high-signal + subprocess-style script tests. |
| `run_local_300_2datasets_engine_supervisor.py` | `keep-root` | high | - | - | `tests/test_scripts/test_local_300_2datasets_one_container_mode.py` | `scripts/README.md` | - | README high-signal + one-container mode tests reference this path. |
| `run_local_300_2datasets_worker.py` | `keep-root` | high | - | `src/study_query_llm/services/jobs/runtime_supervisors.py` | `tests/test_scripts/test_local_300_2datasets_one_container_mode.py` | `docs/living/ARCHITECTURE_CURRENT.md`; `scripts/README.md` | - | Subprocess-spawned worker path is explicitly locked by runtime and docs. |
| `run_mcq_answer_position_probe.py` | `keep-root` | medium | - | - | - | - | `scripts/run_mcq_sweep.py`; `scripts/run_openrouter_mcq_sweep_until_done.py` | MCQ probe driver is cross-invoked by MCQ scripts; lacking living-doc anchor. |
| `run_mcq_sweep.py` | `keep-root` | medium | - | - | - | - | `scripts/run_openrouter_mcq_sweep_until_done.py`; `scripts/ingest_mcq_probe_json_to_sweep_db.py` | Core MCQ driver for automation chain; no living-doc mirror found in this pass. |
| `run_no_pca_50runs_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/run_no_pca_50runs_sweep.py` | Deprecation wrapper per move-set policy. |
| `run_no_pca_multi_embedding_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/run_no_pca_multi_embedding_sweep.py` | Deprecation wrapper per move-set policy. |
| `run_openrouter_mcq_sweep_until_done.py` | `keep-root` | medium | - | - | - | - | `scripts/run_mcq_sweep.py` | Automation wrapper over MCQ sweep; needs stronger docs/tests ownership. |
| `run_pca_kllmeans_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md`; `docs/living/ARCHITECTURE_CURRENT.md` | `scripts/deprecated/pca_kllmeans_sweep.py` | README and architecture references preserve this compatibility surface. |
| `run_pca_kllmeans_sweep_full.py` | `keep-root` | medium | - | - | - | `scripts/README.md` | - | Documented in README narrative; fewer explicit tests found for this script path. |
| `run_prompt_snapshot_llama_openrouter.py` | `move-to-scripts-history` | high | - | - | - | - | - | Experimental prompt-sweep runner with local CSV-default assumptions and no active docs/tests/CI references; classify as historical experiment tooling. |
| `run_snapshot_embedding_model_sweep.py` | `keep-root` | high | - | - | - | `docs/living/API_CURRENT.md`; `docs/living/CURRENT_STATE.md` | - | Living docs explicitly describe this driver behavior and role. |
| `sanity_check_database_url.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/restore_pg_dump_to_local_docker.py`; `scripts/check_db_lane_policy.py` | README lane-policy allowlist + referenced from restore flow messaging. |
| `snapshot_inventory.py` | `keep-root` | high | - | - | - | - | - | Still used as operator inventory probe; additional references exist outside strict living-doc columns. |
| `start_jetstream_postgres_tunnel.py` | `keep-root` | high | - | - | - | `scripts/README.md` | - | README DB matrix includes this tunnel entrypoint. |
| `sync_from_online.py` | `keep-root` | high | - | - | `tests/test_scripts/test_sync_from_online_guardrails.py` | `docs/runbooks/README.md`; `scripts/README.md` | `scripts/init_local_db.py`; `scripts/archive_defective_data.py`; `scripts/check_db_lane_policy.py` | Runbook + README matrix + guardrail tests + peer references. |
| `test_no_pca_sweep.py` | `keep-root` | high | - | - | - | `scripts/README.md` | `scripts/deprecated/test_no_pca_sweep.py` | Deprecation wrapper per move-set policy. |
| `upload_jetstream_pg_dump_to_blob.py` | `keep-root` | high | - | - | `tests/test_scripts/test_upload_jetstream_pg_dump_to_blob.py` | `docs/living/CURRENT_STATE.md`; `docs/runbooks/README.md`; `scripts/README.md` | `scripts/dump_postgres_for_jetstream_migration.py`; `scripts/verify_db_backup_inventory.py` | Living + runbook + README + tests + dump/verify coupling. |
| `validate_and_backfill_run_snapshots.py` | `keep-root` | high | - | - | `tests/test_scripts/test_validate_and_backfill_run_snapshots.py` | `docs/runbooks/README.md`; `scripts/README.md` | - | README high-signal + runbook mention + dedicated tests. |
| `verify_call_artifact_blob_lanes.py` | `keep-root` | high | - | - | `tests/test_scripts/test_verify_call_artifact_blob_lanes.py` | `docs/living/CURRENT_STATE.md`; `docs/runbooks/README.md`; `scripts/README.md` | `scripts/check_db_lane_policy.py` | Living “what to use” guidance + runbook + README + tests. |
| `verify_db_backup_inventory.py` | `keep-root` | high | - | - | `tests/test_scripts/test_verify_db_backup_inventory.py` | `docs/living/CURRENT_STATE.md`; `docs/runbooks/README.md`; `scripts/README.md` | `scripts/upload_jetstream_pg_dump_to_blob.py`; `scripts/check_db_lane_policy.py` | Triple doc surface + tests + upload coupling. |
| `verify_script_path_references.py` | `keep-root` | high | - | - | `tests/test_scripts/test_verify_script_path_references.py` | `scripts/README.md` | - | README maintenance rule + dedicated script test. |
| `warn_restricted_doc_edits.py` | `keep-root` | high | - | - | - | `docs/STANDING_ORDERS.md`; `scripts/README.md` | `scripts/internal/living_docs_governance.py`; `scripts/check_living_docs_drift.py` | Standing orders + README document hook/CI pairing and governance imports. |

## Manual Review Priority (Top 10)

1. `create_bank77_contrast_snapshots.py` (kept at root, medium confidence; add runbook/living-doc anchor if this remains active).
2. `audit_mcq_method_definitions.py` (kept at root, medium confidence; add explicit operator ownership/docs).
3. `export_mcq_sweep_stats.py` (`move-to-scripts-history`; verify no out-of-repo users before path move).
4. `probe_snapshot_long_text_openrouter.py` (`move-to-scripts-history`; verify no active ad-hoc playbooks depend on current path).
5. `run_prompt_snapshot_llama_openrouter.py` (`move-to-scripts-history`; verify no active automation references current path).
6. `fill_snapshot_embeddings_from_baseline.py` (only sibling-script coupling found).
7. `ingest_mcq_probe_json_to_sweep_db.py` (write-path ingest without script-test coverage found).
8. `export_mcq_sweep_option_counts_db.py` (automation coupling but weak living-doc anchoring).
9. `run_openrouter_mcq_sweep_until_done.py` (cost/rate-limit-sensitive automation wrapper).
10. `purge_dataset_acquisition.py` (destructive script; keep-review cadence recommended despite existing tests/guardrails).

## Suggested Cleanup Batches

1. **Batch A (completed):** resolved previously unclear scripts (2 kept at root, 3 classified for `scripts/history`).
2. **Batch B (history lane):** move the 3 `move-to-scripts-history` scripts and keep temporary root wrappers.
3. **Batch C (living lane):** move the 4 `move-to-scripts-living` scripts and keep temporary root wrappers.
4. **Batch D (wrapper consolidation):** evaluate deprecation wrappers for eventual deletion only after references are removed from living docs/runbooks/tests and compatibility window is closed.
5. **Batch E (MCQ automation hardening):** add tests/docs anchors for medium-confidence MCQ scripts before any path moves.

