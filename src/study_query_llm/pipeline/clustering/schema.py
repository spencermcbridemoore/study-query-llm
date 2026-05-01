"""Shared schema constants/types for clustering provenance v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .registry import get_algorithm_spec, iter_algorithm_specs

CLUSTER_PIPELINE_OPERATION_TYPE = "cluster_pipeline"
CLUSTER_PIPELINE_OPERATION_VERSION = "v1"

METHOD_HDBSCAN = "hdbscan"
METHOD_KMEANS_SILHOUETTE_KNEEDLE = "kmeans+silhouette+kneedle"
METHOD_GMM_BIC_ARGMIN = "gmm+bic+argmin"

V1_CLUSTERING_METHODS = frozenset(
    spec.method_name
    for spec in iter_algorithm_specs()
    if spec.provenance_envelope == "clustering_v1"
)

STAGE_VOCABULARY_V1 = frozenset(
    {
        "embed",
        "normalize",
        "pca",
        "umap",
        "hdbscan",
        "kmeans",
        "gmm",
    }
)


def is_v1_clustering_method(method_name: str) -> bool:
    """Return True when ``method_name`` is in v1 clustering scope."""
    spec = get_algorithm_spec(method_name)
    return bool(spec is not None and spec.provenance_envelope == "clustering_v1")


def base_algorithm_for_method(method_name: str) -> str:
    """Return terminal/base algorithm name for a v1 clustering method."""
    normalized = str(method_name or "").strip().lower()
    spec = get_algorithm_spec(normalized)
    if spec is not None and spec.provenance_envelope == "clustering_v1":
        return str(spec.base_algorithm)
    return normalized


@dataclass(frozen=True)
class RuleSet:
    """Loaded rule-set payload and hashes."""

    version: str
    canonical_hash: str
    raw_file_digest: str
    context_keys: tuple[str, ...]
    payload: dict[str, Any]


@dataclass
class ClusteringResolution:
    """Pre-run resolution result for clustering provenance."""

    method_name: str
    base_algorithm: str
    operation_type: str
    operation_version: str
    rule_set_version: str
    rule_set_hash: str
    rule_inputs: dict[str, Any]
    input_audit_metadata: dict[str, str]
    pipeline_declared: list[str]
    pipeline_resolved: list[dict[str, Any]]
    pipeline_effective: list[str]
    selection_policy: dict[str, Any] | None = None
    skipped_stages: list[dict[str, str]] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def clone(self) -> "ClusteringResolution":
        """Return a deep-ish copy suitable for runner-time patching."""
        return ClusteringResolution(
            method_name=str(self.method_name),
            base_algorithm=str(self.base_algorithm),
            operation_type=str(self.operation_type),
            operation_version=str(self.operation_version),
            rule_set_version=str(self.rule_set_version),
            rule_set_hash=str(self.rule_set_hash),
            rule_inputs=dict(self.rule_inputs),
            input_audit_metadata=dict(self.input_audit_metadata),
            pipeline_declared=list(self.pipeline_declared),
            pipeline_resolved=[
                {
                    "stage": str(entry.get("stage") or ""),
                    "params": dict(entry.get("params") or {}),
                    **(
                        {"skipped": bool(entry.get("skipped"))}
                        if "skipped" in entry
                        else {}
                    ),
                    **(
                        {"skip_reason": str(entry.get("skip_reason") or "")}
                        if "skip_reason" in entry
                        else {}
                    ),
                }
                for entry in self.pipeline_resolved
            ],
            pipeline_effective=list(self.pipeline_effective),
            selection_policy=(
                dict(self.selection_policy) if self.selection_policy is not None else None
            ),
            skipped_stages=[dict(item) for item in self.skipped_stages],
            aliases=list(self.aliases),
        )
