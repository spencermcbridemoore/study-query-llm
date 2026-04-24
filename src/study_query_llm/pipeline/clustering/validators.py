"""Hard-constraint validators for clustering provenance v1."""

from __future__ import annotations

from typing import Any

from .schema import ClusteringResolution, STAGE_VOCABULARY_V1, V1_CLUSTERING_METHODS


def _stage_params_by_name(resolution: ClusteringResolution) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("stage") or ""): dict(entry.get("params") or {})
        for entry in resolution.pipeline_resolved
    }


def validate_pre_run(
    resolution: ClusteringResolution,
    *,
    allowed_context_keys: tuple[str, ...] | None = None,
) -> None:
    """Validate pre-run hard constraints."""
    if resolution.method_name not in V1_CLUSTERING_METHODS:
        raise ValueError(f"Unknown method/composite name in v1 allowlist: {resolution.method_name}")
    for stage in resolution.pipeline_declared:
        if stage not in STAGE_VOCABULARY_V1:
            raise ValueError(f"Unknown declared stage name for v1 stage vocabulary: {stage}")

    if allowed_context_keys is not None:
        unknown_keys = sorted(set(resolution.rule_inputs) - set(allowed_context_keys))
        if unknown_keys:
            raise ValueError(
                "Rule references undeclared context key(s): " + ", ".join(unknown_keys)
            )

    if "embedding_dim" in resolution.rule_inputs and "n_samples" in resolution.rule_inputs:
        current_dim = int(resolution.rule_inputs["embedding_dim"])
        n_samples = int(resolution.rule_inputs["n_samples"])
        for entry in resolution.pipeline_resolved:
            stage = str(entry.get("stage") or "")
            params = dict(entry.get("params") or {})
            skipped = bool(entry.get("skipped"))
            if stage == "pca" and not skipped and params.get("n_components") is not None:
                n_components = int(params["n_components"])
                max_allowed = max(1, min(current_dim, n_samples - 1))
                if n_components > max_allowed:
                    raise ValueError(
                        "pca.n_components exceeds hard cap "
                        f"(n_components={n_components}, cap={max_allowed})"
                    )
                current_dim = n_components
            elif stage == "umap" and not skipped and params.get("n_components") is not None:
                n_components = int(params["n_components"])
                if n_components >= current_dim:
                    raise ValueError(
                        "umap.n_components must be < input_dim "
                        f"(n_components={n_components}, input_dim={current_dim})"
                    )
                current_dim = n_components

    stage_params = _stage_params_by_name(resolution)
    if "hdbscan" in stage_params:
        mcs = stage_params["hdbscan"].get("min_cluster_size")
        if mcs is not None and int(mcs) < 2:
            raise ValueError("hdbscan.min_cluster_size must be >= 2")


def validate_post_selection(
    resolution: ClusteringResolution,
    *,
    selection_evidence: dict[str, Any] | None,
) -> None:
    """Validate post-selection constraints."""
    if selection_evidence is None:
        return
    sweep_range = [int(v) for v in list(selection_evidence.get("sweep_range") or [])]
    if not sweep_range:
        raise ValueError("selection_evidence.sweep_range must be non-empty")
    chosen_key = "chosen_k" if "chosen_k" in selection_evidence else "chosen_value"
    if chosen_key not in selection_evidence:
        raise ValueError("selection_evidence must include chosen value")
    chosen = int(selection_evidence[chosen_key])
    if chosen not in sweep_range:
        raise ValueError(
            "selection chosen value must be in sweep range "
            f"(chosen={chosen}, sweep_range={sweep_range})"
        )
    artifact_ref = str(selection_evidence.get("selection_curve_artifact_ref") or "").strip()
    if not artifact_ref:
        raise ValueError("selection_evidence must include selection_curve_artifact_ref")


def validate_identity_contract(
    *,
    pipeline_effective_hash: str,
    recipe_hash: str | None,
) -> None:
    """Validate pipeline-hash/recipe-hash identity contract."""
    if recipe_hash is None:
        return
    if str(pipeline_effective_hash) != str(recipe_hash):
        raise ValueError(
            "Identity contract violation: pipeline_effective_hash must match recipe_hash "
            f"(pipeline_effective_hash={pipeline_effective_hash}, recipe_hash={recipe_hash})"
        )
