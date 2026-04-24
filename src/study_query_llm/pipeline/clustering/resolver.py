"""YAML-backed resolver for clustering provenance v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import (
    CLUSTER_PIPELINE_OPERATION_TYPE,
    CLUSTER_PIPELINE_OPERATION_VERSION,
    ClusteringResolution,
    RuleSet,
    STAGE_VOCABULARY_V1,
    base_algorithm_for_method,
    is_v1_clustering_method,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RULE_SET_PATH = REPO_ROOT / "config" / "rules" / "clustering" / "rules-v1.0.0.yaml"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_rule_set(path: str | Path | None = None) -> RuleSet:
    """Load and hash a clustering rule set from YAML."""
    resolved_path = Path(path) if path is not None else DEFAULT_RULE_SET_PATH
    raw_bytes = resolved_path.read_bytes()
    parsed = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Rule set at {resolved_path} must decode to a mapping")
    canonical_payload = json.dumps(
        _canonicalize(parsed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    version = str(parsed.get("version") or "").strip()
    if not version:
        raise ValueError(f"Rule set at {resolved_path} must define 'version'")
    context_keys = tuple(str(k) for k in list(parsed.get("context_keys") or []))
    return RuleSet(
        version=version,
        canonical_hash=_sha256_hex(canonical_payload),
        raw_file_digest=_sha256_hex(raw_bytes),
        context_keys=context_keys,
        payload=dict(parsed),
    )


def _normalize_declared_pipeline(
    method_name: str,
    parameters: Mapping[str, Any],
) -> list[str]:
    declared = parameters.get("pipeline_declared")
    if isinstance(declared, (list, tuple)):
        normalized = [str(stage).strip().lower() for stage in declared if str(stage).strip()]
        if normalized:
            return normalized

    normalized_method = str(method_name).strip().lower()
    if normalized_method == "hdbscan":
        return ["embed", "hdbscan"]
    if normalized_method == "kmeans+silhouette+kneedle":
        return ["embed", "normalize", "pca", "kmeans"]
    if normalized_method == "gmm+bic+argmin":
        return ["embed", "normalize", "pca", "gmm"]
    return ["embed", base_algorithm_for_method(normalized_method)]


def _stage_has_upstream(
    stage: str,
    pipeline: list[dict[str, Any]],
    upstream_stage: str,
) -> bool:
    for entry in pipeline:
        if str(entry.get("stage") or "") == stage:
            return False
        if str(entry.get("stage") or "") == upstream_stage and not bool(entry.get("skipped")):
            return True
    return False


def _first_matching_threshold(
    rows: list[dict[str, Any]],
    *,
    input_dim: int,
) -> Any:
    for row in rows:
        threshold = row.get("input_dim_le")
        if threshold is None:
            continue
        try:
            if input_dim <= int(threshold):
                return row.get("value")
        except (TypeError, ValueError):
            continue
    return None


def _resolve_dataset_key(parameters: Mapping[str, Any]) -> str | None:
    raw_slug = str(parameters.get("dataset_slug") or "").strip().lower()
    if not raw_slug:
        return None
    if "twenty_newsgroups" in raw_slug:
        return "twenty_newsgroups"
    return raw_slug


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def resolve_clustering_resolution(
    *,
    method_name: str,
    parameters: Mapping[str, Any],
    rule_set: RuleSet,
    context: Mapping[str, Any],
) -> ClusteringResolution:
    """Resolve declared/resolved/effective pipeline for v1 clustering methods."""
    normalized_method = str(method_name or "").strip().lower()
    if not is_v1_clustering_method(normalized_method):
        raise ValueError(f"Unsupported clustering method for v1 resolver: {method_name!r}")

    for key in rule_set.context_keys:
        if key not in context:
            raise ValueError(f"Missing required rule input context key: {key!r}")
    rule_inputs = {key: context[key] for key in rule_set.context_keys}

    declared = _normalize_declared_pipeline(normalized_method, parameters)
    unknown = [stage for stage in declared if stage not in STAGE_VOCABULARY_V1]
    if unknown:
        raise ValueError(f"Unknown stage(s) in declared pipeline: {unknown}")

    payload = rule_set.payload
    global_hygiene = _copy_dict(payload.get("global_hygiene"))
    adjacency_rules = _copy_dict(payload.get("adjacency_rules"))
    selection_policies = _copy_dict(payload.get("selection_policies"))
    dataset_constants_all = _copy_dict(payload.get("dataset_constants"))

    embedding_dim = int(rule_inputs.get("embedding_dim") or 0)
    n_samples = int(rule_inputs.get("n_samples") or 0)
    current_dim = embedding_dim
    base_algorithm = base_algorithm_for_method(normalized_method)
    dataset_key = _resolve_dataset_key(parameters)
    dataset_constants = _copy_dict(dataset_constants_all.get(dataset_key))
    random_state = int(global_hygiene.get("random_state", 42))

    resolved_entries: list[dict[str, Any]] = []
    skipped_stages: list[dict[str, str]] = []

    for stage in declared:
        params_for_stage: dict[str, Any] = {}
        skip_reason: str | None = None
        stage_defaults = _copy_dict(global_hygiene.get(stage))
        params_for_stage.update(stage_defaults)

        if stage in {"pca", "umap", "kmeans", "gmm", "hdbscan"}:
            params_for_stage.setdefault("random_state", random_state)

        if stage == "pca":
            pca_rules = _copy_dict(adjacency_rules.get("pca"))
            down_key = base_algorithm if base_algorithm in {"hdbscan", "kmeans", "gmm"} else "kmeans"
            rule_rows = list(
                _copy_dict(pca_rules.get("n_components_for_downstream")).get(down_key) or []
            )
            suggested = _first_matching_threshold(rule_rows, input_dim=current_dim)
            override = parameters.get("pca_n_components")
            if override is not None:
                suggested = int(override)
            if suggested is not None:
                hard_cap = max(1, min(int(suggested), int(current_dim), int(max(1, n_samples - 1))))
                params_for_stage["n_components"] = hard_cap
            skip_rows = list(pca_rules.get("skip_when") or [])
            for row in skip_rows:
                threshold = row.get("input_dim_le")
                if threshold is None:
                    continue
                if current_dim <= int(threshold):
                    skip_reason = f"pca.skip_when input_dim_le {int(threshold)}"
                    break

        if stage == "normalize":
            params_for_stage = {}

        if stage == "kmeans":
            has_normalize = _stage_has_upstream("kmeans", resolved_entries, "normalize")
            params_for_stage.setdefault(
                "distance_metric",
                "cosine" if has_normalize else "euclidean",
            )
            if parameters.get("kmeans_distance_metric") is not None:
                params_for_stage["distance_metric"] = str(parameters["kmeans_distance_metric"])

        if stage == "gmm":
            params_for_stage.setdefault("covariance_type", "full")
            if parameters.get("gmm_covariance_type") is not None:
                params_for_stage["covariance_type"] = str(parameters["gmm_covariance_type"])

        if stage == "hdbscan":
            hdbscan_rules = _copy_dict(adjacency_rules.get("hdbscan"))
            has_normalize = _stage_has_upstream("hdbscan", resolved_entries, "normalize")
            metric_key = (
                "metric_when_normalized_upstream" if has_normalize else "metric_otherwise"
            )
            params_for_stage.setdefault(
                "metric",
                str(hdbscan_rules.get(metric_key) or "cosine"),
            )
            default_mcs = dataset_constants.get("hdbscan_min_cluster_size_default")
            if default_mcs is not None:
                params_for_stage.setdefault("min_cluster_size", int(default_mcs))
            if parameters.get("hdbscan_min_cluster_size") is not None:
                params_for_stage["min_cluster_size"] = int(parameters["hdbscan_min_cluster_size"])
            if parameters.get("hdbscan_min_samples") is not None:
                params_for_stage["min_samples"] = int(parameters["hdbscan_min_samples"])
            if parameters.get("hdbscan_metric") is not None:
                params_for_stage["metric"] = str(parameters["hdbscan_metric"])

        entry: dict[str, Any] = {"stage": stage, "params": params_for_stage}
        if skip_reason:
            entry["skipped"] = True
            entry["skip_reason"] = skip_reason
            skipped_stages.append({"stage": stage, "reason": skip_reason})
        resolved_entries.append(entry)

        if not skip_reason:
            if stage == "pca" and params_for_stage.get("n_components") is not None:
                current_dim = int(params_for_stage["n_components"])
            elif stage == "umap" and params_for_stage.get("n_components") is not None:
                current_dim = int(params_for_stage["n_components"])

    selection_policy: dict[str, Any] | None = None
    if normalized_method in selection_policies:
        selection_policy = _copy_dict(selection_policies[normalized_method])
        terminal_stage = base_algorithm
        for entry in resolved_entries:
            if str(entry.get("stage") or "") != terminal_stage:
                continue
            params_for_stage = _copy_dict(entry.get("params"))
            sweep_range = list(selection_policy.get("k_range") or [])
            params_for_stage["k_range"] = [int(v) for v in sweep_range]
            params_for_stage["selection_metric"] = str(selection_policy.get("selection_metric") or "")
            params_for_stage["selection_rule"] = str(selection_policy.get("selection_rule") or "")
            if parameters.get("k") is not None:
                params_for_stage["k"] = int(parameters["k"])
            entry["params"] = params_for_stage
            break

    pipeline_effective = [
        str(entry.get("stage") or "")
        for entry in resolved_entries
        if not bool(entry.get("skipped"))
    ]

    input_audit_metadata: dict[str, str] = {}
    for key in ("embedding_model_id", "embedding_provider", "embedding_engine"):
        if parameters.get(key) is not None:
            input_audit_metadata[key] = str(parameters[key])
    input_audit_metadata["rule_set_file_digest"] = str(rule_set.raw_file_digest)

    return ClusteringResolution(
        method_name=normalized_method,
        base_algorithm=base_algorithm,
        operation_type=CLUSTER_PIPELINE_OPERATION_TYPE,
        operation_version=CLUSTER_PIPELINE_OPERATION_VERSION,
        rule_set_version=f"rules-v{rule_set.version}",
        rule_set_hash=rule_set.canonical_hash,
        rule_inputs=rule_inputs,
        input_audit_metadata=input_audit_metadata,
        pipeline_declared=declared,
        pipeline_resolved=resolved_entries,
        pipeline_effective=pipeline_effective,
        selection_policy=selection_policy,
        skipped_stages=skipped_stages,
        aliases=[],
    )
