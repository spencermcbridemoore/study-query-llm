"""All-variant clustering analysis backfill: plan, coverage, and execution helpers.

Used by :file:`scripts/living/backfill_all_variant_clustering_analysis.py` to run
registry bundled methods across snapshot/embedding_batch pairs with resumable
state and deterministic ``analysis_run_key`` values.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from study_query_llm.db.models_v2 import Group, ProvenancedRun
from study_query_llm.pipeline.clustering.registry import (
    AlgorithmSpec,
    iter_algorithm_specs,
)

# Prefix for provenanced run_key rows created by this backfill (query-friendly).
BACKFILL_ANALYSIS_RUN_KEY_PREFIX = "backfill_exec__"

# Default sweep grid (k=1 is requested but runners filter to k>=2).
DEFAULT_SWEEP_K_RANGE: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]

# Fixed-k expansion (k=1 is invalid for bundled fixed-k runners).
DEFAULT_FIXED_K_MIN = 2
DEFAULT_FIXED_K_MAX = 10


@dataclass
class SnapshotLineage:
    """Lineage for a ``dataset_snapshot`` group."""

    snapshot_group_id: int
    source_dataframe_group_id: int
    snapshot_row_count: int
    source_dataframe_row_count: int

    @property
    def lineage_key(self) -> tuple[int, int]:
        return (int(self.source_dataframe_group_id), int(self.source_dataframe_row_count))


@dataclass
class EmbeddingBatchMatch:
    group_id: int
    provider: str
    embedding_engine: str
    entry_max: int
    source_dataframe_group_id: int


@dataclass
class BackfillTarget:
    """One analyze invocation: snapshot + batch + method + parameters."""

    snapshot_group_id: int
    embedding_batch_group_id: int
    method_name: str
    method_version: str
    parameters: dict[str, Any]
    run_key: str
    analysis_run_key: str
    param_fingerprint: str
    pca_n_components: int

    def coverage_key(self) -> str:
        return self.analysis_run_key


@dataclass
class RunStateEntry:
    status: str  # planned | started | completed | failed | skipped
    error: str | None = None
    attempts: int = 0
    updated_at: str | None = None


@dataclass
class RunState:
    """Resume state persisted as JSON (UTF-8)."""

    version: int = 1
    targets: dict[str, RunStateEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunState:
        entries = {}
        for k, v in (raw.get("targets") or {}).items():
            if not isinstance(v, dict):
                continue
            entries[str(k)] = RunStateEntry(
                status=str(v.get("status") or "planned"),
                error=(str(v["error"]) if v.get("error") is not None else None),
                attempts=int(v.get("attempts") or 0),
                updated_at=(
                    str(v["updated_at"]) if v.get("updated_at") is not None else None
                ),
            )
        return cls(version=int(raw.get("version") or 1), targets=entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "targets": {
                k: {
                    "status": v.status,
                    "error": v.error,
                    "attempts": v.attempts,
                    "updated_at": v.updated_at,
                }
                for k, v in self.targets.items()
            },
        }


def load_run_state(path: str) -> RunState:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return RunState()
    raw = json.loads(p.read_text(encoding="utf-8"))
    return RunState.from_dict(raw if isinstance(raw, dict) else {})


def save_run_state(path: str, state: RunState) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )


def snapshot_lineage(session, snapshot_group_id: int) -> SnapshotLineage:
    g = (
        session.query(Group)
        .filter(
            Group.id == int(snapshot_group_id),
            Group.group_type == "dataset_snapshot",
        )
        .first()
    )
    if g is None:
        raise ValueError(f"dataset_snapshot id={snapshot_group_id} not found")
    md = dict(g.metadata_json or {})
    sdf = int(md.get("source_dataframe_group_id") or 0)
    rows = int(md.get("row_count") or 0)
    if sdf <= 0 or rows <= 0:
        raise ValueError(
            f"snapshot {snapshot_group_id} missing source_dataframe_group_id/row_count"
        )
    dfg = (
        session.query(Group)
        .filter(Group.id == sdf, Group.group_type == "dataset_dataframe")
        .first()
    )
    df_md = dict((dfg.metadata_json if dfg else {}) or {})
    df_rows = int(df_md.get("row_count") or 0)
    return SnapshotLineage(
        snapshot_group_id=int(snapshot_group_id),
        source_dataframe_group_id=sdf,
        snapshot_row_count=rows,
        source_dataframe_row_count=df_rows,
    )


def safe_pca_n_components(*, n_samples: int, cap: int = 10) -> int:
    """PCA dims for backfill: min(cap, max(1, n_samples-1))."""
    n = int(n_samples)
    return max(1, min(int(cap), max(1, n - 1)))


def safe_pca_for_sweep(*, n_samples: int, embedding_dim: int) -> int:
    """Default PCA size for sweep methods (matches analyze clamp semantics)."""
    n = int(n_samples)
    d = int(embedding_dim)
    return max(1, min(100, d, max(1, n - 1)))


def list_snapshots(session) -> list[int]:
    rows = (
        session.query(Group.id)
        .filter(Group.group_type == "dataset_snapshot")
        .order_by(Group.id.asc())
        .all()
    )
    return [int(r[0]) for r in rows]


def embedding_batches_for_lineage_and_engine(
    session,
    lineage_key: tuple[int, int],
    embedding_engine: str,
    *,
    provider: str | None = None,
) -> list[EmbeddingBatchMatch]:
    sdf_id, entry_max = int(lineage_key[0]), int(lineage_key[1])
    eng_norm = str(embedding_engine).strip()
    out: list[EmbeddingBatchMatch] = []
    for g in session.query(Group).filter(Group.group_type == "embedding_batch").all():
        md = dict(g.metadata_json or {})
        try:
            ksdf = int(md.get("source_dataframe_group_id") or 0)
            kem = int(md.get("entry_max") or 0)
        except (TypeError, ValueError):
            continue
        if ksdf != sdf_id or kem != entry_max:
            continue
        prov = str(md.get("provider") or "").strip()
        eng = str(md.get("embedding_engine") or md.get("deployment") or "").strip()
        if eng != eng_norm:
            continue
        if provider is not None and prov.lower() != str(provider).strip().lower():
            continue
        out.append(
            EmbeddingBatchMatch(
                group_id=int(g.id),
                provider=prov,
                embedding_engine=eng,
                entry_max=kem,
                source_dataframe_group_id=ksdf,
            )
        )
    out.sort(key=lambda x: x.group_id)
    return out


def _method_token_for_key(method_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(method_name).strip())[:120]


def param_fingerprint(parameters: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        parameters,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def line_run_key(*, snapshot_group_id: int, embedding_batch_group_id: int) -> str:
    return f"backfill_line__snap{int(snapshot_group_id)}__emb{int(embedding_batch_group_id)}"


def analysis_run_key_for_target(
    *,
    snapshot_group_id: int,
    embedding_batch_group_id: int,
    method_name: str,
    parameters: Mapping[str, Any],
) -> str:
    fp = param_fingerprint(parameters)
    tok = _method_token_for_key(method_name)
    return (
        f"{BACKFILL_ANALYSIS_RUN_KEY_PREFIX}snap{int(snapshot_group_id)}__"
        f"emb{int(embedding_batch_group_id)}__{tok}__{fp}"
    )


def validate_parameters_for_method(
    method_name: str,
    parameters: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
) -> None:
    props = dict(schema.get("properties") or {})
    allowed = set(props.keys())
    params = dict(parameters)
    unknown = sorted(k for k in params.keys() if k not in allowed)
    if unknown:
        raise ValueError(
            f"unknown_clustering_method_parameters:{method_name}:"
            f"{','.join(unknown)}"
        )
    required = [
        str(x).strip()
        for x in list(schema.get("required") or [])
        if str(x).strip()
    ]
    missing = sorted(k for k in required if params.get(k) is None)
    if missing:
        raise ValueError(
            f"missing_required_clustering_method_parameters:{method_name}:"
            f"{','.join(missing)}"
        )


def dbscan_defaults(method_name: str) -> dict[str, Any]:
    """First-pass DBSCAN template from backfill plan."""
    mn = str(method_name).strip().lower()
    if mn == "dbscan+fixed-eps":
        return {"eps": 0.20, "min_samples": 5, "metric": "cosine"}
    if mn == "dbscan+normalize+fixed-eps":
        return {"eps": 0.15, "min_samples": 5, "metric": "cosine"}
    if mn == "dbscan+pca+fixed-eps":
        return {"eps": 0.30, "min_samples": 5, "metric": "euclidean"}
    if mn == "dbscan+normalize+pca+fixed-eps":
        return {"eps": 0.25, "min_samples": 5, "metric": "euclidean"}
    raise ValueError(f"not a dbscan fixed-eps method: {method_name!r}")


def hdbscan_defaults() -> dict[str, Any]:
    """Minimal explicit params for schema-visible knobs."""
    return {
        "min_cluster_size": 5,
        "min_samples": 5,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "cluster_selection_epsilon": 0.0,
        "alpha": 1.0,
        "allow_single_cluster": False,
        "random_state": 42,
        "core_dist_n_jobs": 1,
        "approx_min_span_tree": False,
    }


def expand_parameter_variants(
    spec: AlgorithmSpec,
    *,
    n_samples: int,
    embedding_dim: int,
    sweep_k_range: Sequence[int] | None = None,
    fixed_k_range: range | None = None,
) -> list[dict[str, Any]]:
    """Return concrete parameter dicts for one registry spec."""
    sweep_k_range = list(sweep_k_range or DEFAULT_SWEEP_K_RANGE)
    fk = fixed_k_range or range(DEFAULT_FIXED_K_MIN, DEFAULT_FIXED_K_MAX + 1)
    schema = dict(spec.parameters_schema or {})
    mn = str(spec.method_name).strip().lower()

    if spec.fit_mode == "sweep_select":
        pca_nc = safe_pca_for_sweep(n_samples=n_samples, embedding_dim=embedding_dim)
        k_range = [int(x) for x in sweep_k_range]
        if not k_range:
            raise ValueError(f"empty k_range for sweep method {mn}")
        if any(i < 1 for i in k_range):
            raise ValueError(f"invalid k_range (k<1) for {mn}: {k_range!r}")
        params = {
            "k_range": k_range,
            "pca_n_components": int(pca_nc),
        }
        validate_parameters_for_method(mn, params, schema=schema)
        return [params]

    if "dbscan" in mn and "fixed-eps" in mn:
        params = dict(dbscan_defaults(mn))
        if "pca" in mn.split("+"):
            params["pca_n_components"] = safe_pca_n_components(n_samples=n_samples)
        validate_parameters_for_method(mn, params, schema=schema)
        return [params]

    if mn.startswith("hdbscan"):
        params = dict(hdbscan_defaults())
        if "pca" in mn.split("+"):
            params["pca_n_components"] = safe_pca_n_components(n_samples=n_samples)
        validate_parameters_for_method(mn, params, schema=schema)
        return [params]

    # fixed-k families (kmeans, gmm, agglomerative, spherical-kmeans)
    required = list(schema.get("required") or [])
    if "k" not in required:
        raise ValueError(f"unexpected schema for {mn}: expected fixed-k with required k")
    variants: list[dict[str, Any]] = []
    max_k = max(DEFAULT_FIXED_K_MIN, min(DEFAULT_FIXED_K_MAX, n_samples - 1))
    for k in fk:
        if int(k) == 1:
            raise ValueError(f"k=1 is forbidden for bundled fixed-k runners ({mn})")
        if int(k) < DEFAULT_FIXED_K_MIN or int(k) > max_k:
            continue
        params = {"k": int(k)}
        if "pca_n_components" in required:
            params["pca_n_components"] = safe_pca_n_components(n_samples=n_samples)
        if mn.startswith("agglomerative"):
            params.setdefault("linkage", "ward")
            params.setdefault("metric", "euclidean")
        if mn.startswith("gmm"):
            params.setdefault("covariance_type", "full")
            params.setdefault("n_init", 10)
            params.setdefault("random_state", 42)
        if "kmeans" in mn or mn.startswith("spherical-kmeans"):
            params.setdefault("random_state", 42)
            params.setdefault("n_init", 10)
            params.setdefault("init", "k-means++")
        validate_parameters_for_method(mn, params, schema=schema)
        variants.append(params)
    return variants


def build_all_targets(
    *,
    snapshot_group_id: int,
    embedding_batch_group_id: int,
    n_samples: int,
    embedding_dim: int,
    method_version: str = "1.0",
    specs: Sequence[AlgorithmSpec] | None = None,
) -> list[BackfillTarget]:
    specs = list(specs or iter_algorithm_specs())
    rk = line_run_key(
        snapshot_group_id=snapshot_group_id,
        embedding_batch_group_id=embedding_batch_group_id,
    )
    pca_cap = safe_pca_n_components(n_samples=n_samples)
    out: list[BackfillTarget] = []
    for spec in specs:
        for params in expand_parameter_variants(
            spec,
            n_samples=n_samples,
            embedding_dim=embedding_dim,
        ):
            fp = param_fingerprint(params)
            ark = analysis_run_key_for_target(
                snapshot_group_id=snapshot_group_id,
                embedding_batch_group_id=embedding_batch_group_id,
                method_name=spec.method_name,
                parameters=params,
            )
            out.append(
                BackfillTarget(
                    snapshot_group_id=int(snapshot_group_id),
                    embedding_batch_group_id=int(embedding_batch_group_id),
                    method_name=str(spec.method_name),
                    method_version=str(method_version),
                    parameters=dict(params),
                    run_key=rk,
                    analysis_run_key=ark,
                    param_fingerprint=fp,
                    pca_n_components=int(pca_cap),
                )
            )
    return out


def fetch_completed_backfill_run_keys(session) -> set[str]:
    """Run_keys for completed backfill analyses (prefix-filtered)."""
    rows = (
        session.query(ProvenancedRun.run_key)
        .filter(
            ProvenancedRun.run_status == "completed",
            ProvenancedRun.run_key.like(f"{BACKFILL_ANALYSIS_RUN_KEY_PREFIX}%"),
        )
        .all()
    )
    return {str(r[0]) for r in rows if r[0]}


def build_manifest(
    session,
    *,
    embedding_engine: str,
    provider: str | None = None,
    snapshot_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Dry-run manifest: pairs, targets, completed keys, collisions."""
    snaps = list(snapshot_ids) if snapshot_ids is not None else list_snapshots(session)
    completed = fetch_completed_backfill_run_keys(session)
    pairs: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    all_targets: list[BackfillTarget] = []

    for sid in snaps:
        try:
            lin = snapshot_lineage(session, int(sid))
        except ValueError as exc:
            pairs.append(
                {
                    "snapshot_group_id": int(sid),
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        batches = embedding_batches_for_lineage_and_engine(
            session,
            lin.lineage_key,
            embedding_engine,
            provider=provider,
        )
        if len(batches) == 0:
            pairs.append(
                {
                    "snapshot_group_id": int(sid),
                    "lineage_key": list(lin.lineage_key),
                    "status": "no_embedding_batch",
                    "embedding_engine": embedding_engine,
                }
            )
            continue
        if len(batches) > 1:
            collisions.append(
                {
                    "snapshot_group_id": int(sid),
                    "lineage_key": list(lin.lineage_key),
                    "embedding_engine": embedding_engine,
                    "batch_ids": [b.group_id for b in batches],
                }
            )
            pairs.append(
                {
                    "snapshot_group_id": int(sid),
                    "status": "collision_multiple_batches",
                    "batch_ids": [b.group_id for b in batches],
                }
            )
            continue
        b = batches[0]
        # Embedding matrix dimension: from batch metadata when present.
        bg = session.query(Group).filter(Group.id == int(b.group_id)).first()
        bmd = dict((bg.metadata_json if bg else {}) or {})
        dim_raw = bmd.get("dimension")
        try:
            embedding_dim = int(dim_raw) if dim_raw is not None else 0
        except (TypeError, ValueError):
            embedding_dim = 0
        n_samples = int(lin.snapshot_row_count)
        targets = build_all_targets(
            snapshot_group_id=int(sid),
            embedding_batch_group_id=int(b.group_id),
            n_samples=n_samples,
            embedding_dim=max(1, embedding_dim),
        )
        missing = [t.analysis_run_key for t in targets if t.analysis_run_key not in completed]
        pairs.append(
            {
                "snapshot_group_id": int(sid),
                "embedding_batch_group_id": int(b.group_id),
                "lineage_key": list(lin.lineage_key),
                "provider": b.provider,
                "embedding_engine": b.embedding_engine,
                "n_samples": n_samples,
                "embedding_dimension": embedding_dim,
                "target_count": len(targets),
                "completed_count": len([t for t in targets if t.analysis_run_key in completed]),
                "missing_count": len(missing),
                "status": "ok",
            }
        )
        all_targets.extend(targets)

    missing_all = [t.analysis_run_key for t in all_targets if t.analysis_run_key not in completed]

    keys_seen: set[str] = set()
    dup_keys: list[str] = []
    for t in all_targets:
        k = t.analysis_run_key
        if k in keys_seen:
            dup_keys.append(k)
        keys_seen.add(k)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_engine": embedding_engine,
        "provider_filter": provider,
        "pairs": pairs,
        "collisions": collisions,
        "total_targets": len(all_targets),
        "completed_distinct": len([t for t in all_targets if t.analysis_run_key in completed]),
        "missing_count": len(set(missing_all)),
        "targets": [
            {
                "snapshot_group_id": t.snapshot_group_id,
                "embedding_batch_group_id": t.embedding_batch_group_id,
                "method_name": t.method_name,
                "method_version": t.method_version,
                "parameters": t.parameters,
                "run_key": t.run_key,
                "analysis_run_key": t.analysis_run_key,
                "param_fingerprint": t.param_fingerprint,
            }
            for t in all_targets
        ],
        "missing_analysis_run_keys": sorted(set(missing_all)),
        "duplicate_analysis_run_keys": sorted(set(dup_keys)),
    }


def validate_registry_expansion_or_raise(
    *,
    n_samples: int = 300,
    embedding_dim: int = 1536,
) -> None:
    """Dry-run expand every bundled registry method; raises on schema/param errors."""
    for spec in iter_algorithm_specs():
        expand_parameter_variants(
            spec,
            n_samples=int(n_samples),
            embedding_dim=int(embedding_dim),
        )


def ok_snapshot_ids_from_manifest(manifest: Mapping[str, Any]) -> list[int]:
    """Return sorted unique ``dataset_snapshot`` group ids with manifest pair ``status=="ok"``.

    Used to shard backfill work across worker processes without overlapping snapshots.
    """
    out: list[int] = []
    for p in manifest.get("pairs") or []:
        if not isinstance(p, dict):
            continue
        if p.get("status") != "ok":
            continue
        sid = p.get("snapshot_group_id")
        if sid is None:
            continue
        try:
            out.append(int(sid))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def round_robin_shard_snapshot_ids(
    snapshot_ids: Sequence[int],
    shard_count: int,
) -> list[list[int]]:
    """Partition snapshot ids into ``shard_count`` disjoint round-robin shards.

    Shards are deterministic given sorted input ids: shard ``i`` receives ids at
    indices ``i, i+N, i+2N, ...`` after sorting deduplicated ids ascending.
    """
    n = int(shard_count)
    if n < 1:
        raise ValueError("shard_count must be >= 1")
    ids = sorted({int(x) for x in snapshot_ids})
    return [ids[i::n] for i in range(n)]


def validate_shards_partition_exact(
    snapshot_ids: Sequence[int],
    shards: Sequence[Sequence[int]],
) -> None:
    """Raise ``ValueError`` if shards overlap or their union != ``snapshot_ids``."""
    universe = set(int(x) for x in snapshot_ids)
    seen: set[int] = set()
    for shard in shards:
        for x in shard:
            xi = int(x)
            if xi in seen:
                raise ValueError(f"overlapping snapshot id across shards: {xi}")
            seen.add(xi)
    if seen != universe:
        missing = sorted(universe - seen)
        extra = sorted(seen - universe)
        raise ValueError(
            "shard partition mismatch: "
            f"missing={missing!r} extra={extra!r}"
        )


def preflight_manifest_blocking_issues(manifest: Mapping[str, Any]) -> list[str]:
    """Human-readable blockers for CLI exit codes (collisions, duplicates, empty scope)."""
    issues: list[str] = []
    if manifest.get("collisions"):
        issues.append(
            "embedding_batch_collisions: resolve lineage so each snapshot maps to "
            "exactly one batch for the target engine"
        )
    dup = list(manifest.get("duplicate_analysis_run_keys") or [])
    if dup:
        issues.append(f"duplicate_analysis_run_keys:{len(dup)}")
    ok_pairs = [
        p
        for p in (manifest.get("pairs") or [])
        if isinstance(p, dict) and p.get("status") == "ok"
    ]
    if not ok_pairs:
        issues.append("no_eligible_snapshot_batch_pairs: check snapshots and embedding coverage")
    return issues


def refresh_manifest_completion(session, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recompute ``missing_analysis_run_keys`` from DB (mutates manifest copy-safe)."""
    completed = fetch_completed_backfill_run_keys(session)
    targets = list(manifest.get("targets") or [])
    missing = sorted(
        {
            str(t.get("analysis_run_key"))
            for t in targets
            if t.get("analysis_run_key") and str(t.get("analysis_run_key")) not in completed
        }
    )
    out = dict(manifest)
    out["missing_analysis_run_keys"] = missing
    out["missing_count"] = len(missing)
    all_keys = {str(t.get("analysis_run_key")) for t in targets if t.get("analysis_run_key")}
    out["completed_distinct"] = len(all_keys) - len(set(missing))
    return out


def verify_manifest_coverage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Post-run verification summary."""
    targets = list(manifest.get("targets") or [])
    keys = {str(t.get("analysis_run_key") or "") for t in targets if t.get("analysis_run_key")}
    missing = {str(x) for x in (manifest.get("missing_analysis_run_keys") or [])}
    dup_manifest = list(manifest.get("duplicate_analysis_run_keys") or [])
    unexpected_missing = missing - keys
    return {
        "expected_target_keys": len(keys),
        "missing_keys_remaining": len(missing & keys),
        "collision_count": len(manifest.get("collisions") or []),
        "duplicate_analysis_run_keys": len(dup_manifest),
        "unexpected_missing_keys": sorted(unexpected_missing),
        "coverage_complete": len(keys) > 0 and len(missing & keys) == 0 and not dup_manifest,
    }


def run_backfill_execution(
    *,
    db: Any,
    manifest: Mapping[str, Any],
    run_state_path: str | None,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    max_attempts_per_target: int = 3,
    failure_halt_fraction: float = 0.05,
    sleep_seconds_on_transient: float = 2.0,
) -> dict[str, Any]:
    """Execute analyze() for missing targets; update optional run-state JSON."""
    from study_query_llm.pipeline.analyze import analyze

    targets_raw = list(manifest.get("targets") or [])
    completed_global = set()
    with db.session_scope() as session:
        completed_global = fetch_completed_backfill_run_keys(session)

    state = load_run_state(run_state_path) if run_state_path else RunState()
    if not resume and run_state_path:
        state = RunState()

    # Initialize state entries
    for row in targets_raw:
        key = str(row.get("analysis_run_key") or "")
        if key and key not in state.targets:
            st = "completed" if (key in completed_global and not force) else "planned"
            state.targets[key] = RunStateEntry(status=st)

    stats: dict[str, Any] = {
        "dry_run": dry_run,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "started": 0,
        "dry_run_would_execute": 0,
    }
    min_failures_before_rate_halt = 5

    for row in targets_raw:
        key = str(row.get("analysis_run_key") or "")
        if not key:
            continue
        entry = state.targets[key]
        if entry.status == "completed" and not force:
            stats["skipped"] += 1
            continue
        if key in completed_global and not force:
            entry.status = "completed"
            stats["skipped"] += 1
            continue

        snap = int(row["snapshot_group_id"])
        emb = int(row["embedding_batch_group_id"])
        method = str(row["method_name"])
        params = dict(row.get("parameters") or {})
        run_key = str(row.get("run_key") or line_run_key(snapshot_group_id=snap, embedding_batch_group_id=emb))

        if dry_run:
            stats["dry_run_would_execute"] += 1
            entry.status = "planned"
            continue

        entry.status = "started"
        entry.attempts += 1
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        if run_state_path:
            save_run_state(run_state_path, state)

        stats["started"] += 1
        last_err: str | None = None
        for attempt in range(max_attempts_per_target):
            try:
                analyze(
                    snap,
                    emb,
                    method_name=method,
                    run_key=run_key,
                    analysis_run_key=key,
                    request_group_id=None,
                    method_version=str(row.get("method_version") or "1.0"),
                    parameters=params,
                    force=force,
                    db=db,
                )
                entry.status = "completed"
                entry.error = None
                stats["completed"] += 1
                completed_global.add(key)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < max_attempts_per_target:
                    time.sleep(float(sleep_seconds_on_transient))
        if last_err is not None:
            entry.status = "failed"
            entry.error = last_err
            stats["failed"] += 1
            finished = stats["completed"] + stats["failed"]
            if finished >= min_failures_before_rate_halt and (
                stats["failed"] / float(finished)
            ) > float(failure_halt_fraction):
                stats["halted_reason"] = "failure_rate_exceeded"
                if run_state_path:
                    save_run_state(run_state_path, state)
                return stats

        if run_state_path:
            save_run_state(run_state_path, state)

    if run_state_path:
        save_run_state(run_state_path, state)
    return stats
