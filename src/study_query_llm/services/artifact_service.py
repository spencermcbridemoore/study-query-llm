"""
Artifact Service - Store algorithm outputs as artifacts with provenance tracking.

This service handles storing algorithm outputs (cluster labels, metrics, PCA components,
sweep results) as files (JSON/NPY/CSV) with URIs in CallArtifact, linked to runs.

Usage:
    from study_query_llm.services.artifact_service import ArtifactService
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository

    db = DatabaseConnectionV2("postgresql://...")
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir="artifacts")

        # Store sweep results
        artifact_id = service.store_sweep_results(
            run_id=run_id,
            sweep_results={"by_k": {...}, "pca_meta": {...}},
            step_name="sweep_complete"
        )
"""

import csv
import io
import json
import hashlib
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import numpy as np

from ..db.write_intent import WriteIntent, parse_write_intent
from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository
    from ..storage.protocol import StorageBackend

logger = get_logger(__name__)

# Default artifact directory
DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_AZURE_BLOB_QUOTA_BYTES = 100 * 1024 * 1024 * 1024  # 100 GiB


class ArtifactService:
    """
    Service for storing algorithm outputs as artifacts with provenance tracking.

    Features:
    - File-based storage (JSON/NPY/CSV)
    - URI generation following conventions
    - CallArtifact creation and linking to groups
    - Metadata storage for provenance
    """

    def __init__(
        self,
        repository: Optional["RawCallRepository"] = None,
        artifact_dir: str = DEFAULT_ARTIFACT_DIR,
        storage_backend: Optional["StorageBackend"] = None,
        write_intent: WriteIntent | str | None = None,
    ):
        """
        Initialize the artifact service.

        Args:
            repository: Optional RawCallRepository for DB persistence
            artifact_dir: Base directory for storing artifacts (default: "artifacts")
            storage_backend: Optional StorageBackend. When None, uses LocalStorageBackend
                with base_dir=artifact_dir for backward compatibility.
        """
        self.repository = repository
        self.artifact_dir = Path(artifact_dir)
        self._operation_counts: dict[str, int] = defaultdict(int)
        self.write_intent = self._normalize_write_intent(write_intent)
        if storage_backend is not None:
            self.storage = storage_backend
        else:
            self.storage = self._resolve_default_backend(
                artifact_dir,
                enforce_azure=self.write_intent is WriteIntent.CANONICAL,
            )
        self._enforce_backend_intent_contract()

    @staticmethod
    def _normalize_write_intent(
        write_intent: WriteIntent | str | None,
    ) -> WriteIntent | None:
        if write_intent is None:
            return None
        if isinstance(write_intent, WriteIntent):
            return write_intent
        return parse_write_intent(write_intent)

    def _enforce_backend_intent_contract(self) -> None:
        backend_type = str(getattr(self.storage, "backend_type", "unknown")).lower()
        if self.write_intent is WriteIntent.CANONICAL and backend_type != "azure_blob":
            raise RuntimeError(
                "Canonical write intent requires azure_blob artifact storage backend. "
                f"Resolved backend={backend_type!r}."
            )

    def _resolve_default_backend(self, artifact_dir: str, *, enforce_azure: bool):
        """
        Resolve storage backend from ARTIFACT_STORAGE_BACKEND env.
        Safe fallback to local when unset or when azure_blob cannot be created.
        """
        from ..storage.factory import StorageBackendFactory

        backend_type = (os.environ.get("ARTIFACT_STORAGE_BACKEND") or "local").strip().lower()
        runtime_env = (os.environ.get("ARTIFACT_RUNTIME_ENV") or "dev").strip().lower()
        strict_mode = self._is_truthy(os.environ.get("ARTIFACT_STORAGE_STRICT_MODE"))
        if runtime_env in {"stage", "prod"}:
            strict_mode = True

        if backend_type == "local" and (strict_mode or enforce_azure):
            if enforce_azure:
                raise RuntimeError(
                    "Canonical write intent requires azure_blob artifact storage backend. "
                    "Set ARTIFACT_STORAGE_BACKEND=azure_blob."
                )
            raise ValueError(
                "Local artifact backend is disallowed in strict mode "
                f"(runtime={runtime_env}). Set ARTIFACT_STORAGE_BACKEND=azure_blob."
            )

        if backend_type == "azure_blob":
            container_name = self._resolve_blob_container(runtime_env)
            auth_mode = (os.environ.get("ARTIFACT_AUTH_MODE") or "connection_string").strip().lower()
            try:
                return StorageBackendFactory.create(
                    "azure_blob",
                    container_name=container_name,
                    auth_mode=auth_mode,
                    account_url=os.environ.get("AZURE_STORAGE_ACCOUNT_URL"),
                    managed_identity_client_id=os.environ.get(
                        "AZURE_STORAGE_MANAGED_IDENTITY_CLIENT_ID"
                    ),
                    blob_prefix=(os.environ.get("AZURE_STORAGE_PREFIX") or runtime_env),
                    max_retries=int(os.environ.get("AZURE_STORAGE_MAX_RETRIES") or "3"),
                    retry_backoff_seconds=float(
                        os.environ.get("AZURE_STORAGE_RETRY_BACKOFF_SECONDS") or "0.5"
                    ),
                    verify_uploads=self._is_truthy(
                        os.environ.get("AZURE_STORAGE_VERIFY_UPLOADS"),
                        default=True,
                    ),
                    runtime_env=runtime_env,
                )
            except (ValueError, ImportError) as e:
                if strict_mode or enforce_azure:
                    raise RuntimeError(
                        "Failed to configure azure_blob artifact backend in strict mode "
                        "or canonical intent mode."
                    ) from e
                logger.warning(
                    "ARTIFACT_STORAGE_BACKEND=azure_blob requested but backend unavailable: %s. "
                    "Falling back to local.",
                    e,
                )
                backend_type = "local"
        if backend_type == "local" or not backend_type:
            return StorageBackendFactory.create("local", base_dir=artifact_dir)
        logger.warning(
            "Unknown ARTIFACT_STORAGE_BACKEND=%r, falling back to local.",
            backend_type,
        )
        return StorageBackendFactory.create("local", base_dir=artifact_dir)

    @staticmethod
    def _is_truthy(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _integrity_metadata(data: bytes) -> Dict[str, Any]:
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        }

    def _resolve_blob_container(self, runtime_env: str) -> str:
        base = (os.environ.get("AZURE_STORAGE_CONTAINER") or "artifacts").strip()
        # Optional per-lane override; otherwise derive from base: {base}-{dev|stage|prod}
        explicit = os.environ.get(f"AZURE_STORAGE_CONTAINER_{runtime_env.upper()}")
        if explicit and explicit.strip():
            container = explicit.strip()
        elif runtime_env in ("dev", "stage", "prod"):
            container = f"{base}-{runtime_env}"
        else:
            container = base
        if not container:
            raise ValueError("Resolved empty Azure blob container name.")

        allow_cross_env = self._is_truthy(
            os.environ.get("ARTIFACT_ALLOW_CROSS_ENV_CONTAINER"), default=False
        )
        lowered = container.lower()
        if runtime_env != "prod" and "prod" in lowered and not allow_cross_env:
            raise ValueError(
                f"Refusing to use prod-like container {container!r} in runtime {runtime_env!r}. "
                "Set ARTIFACT_ALLOW_CROSS_ENV_CONTAINER=true to override."
            )
        return container

    def _record_storage_event(
        self,
        *,
        operation: str,
        status: str,
        started_at: float,
        logical_path: Optional[str] = None,
        uri: Optional[str] = None,
        artifact_type: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        key = f"{operation}.{status}"
        self._operation_counts[key] = int(self._operation_counts.get(key, 0)) + 1
        if status == "success":
            logger.info(
                "artifact_storage_event op=%s status=%s elapsed_ms=%.2f artifact_type=%s logical_path=%s uri=%s count=%s",
                operation,
                status,
                elapsed_ms,
                artifact_type or "",
                logical_path or "",
                uri or "",
                self._operation_counts[key],
            )
        else:
            logger.warning(
                "artifact_storage_event op=%s status=%s elapsed_ms=%.2f artifact_type=%s logical_path=%s uri=%s error=%s count=%s",
                operation,
                status,
                elapsed_ms,
                artifact_type or "",
                logical_path or "",
                uri or "",
                error,
                self._operation_counts[key],
            )

    @staticmethod
    def _resolve_azure_quota_bytes() -> int:
        """
        Resolve Azure blob hard quota from env, defaulting to 100 GiB.

        Supported env vars:
        - ARTIFACT_BLOB_MAX_BYTES
        - ARTIFACT_BLOB_MAX_GB
        """
        raw_bytes = os.environ.get("ARTIFACT_BLOB_MAX_BYTES")
        if raw_bytes and raw_bytes.strip():
            try:
                return max(0, int(raw_bytes))
            except ValueError:
                logger.warning(
                    "Invalid ARTIFACT_BLOB_MAX_BYTES=%r; using default %s",
                    raw_bytes,
                    DEFAULT_AZURE_BLOB_QUOTA_BYTES,
                )
        raw_gb = os.environ.get("ARTIFACT_BLOB_MAX_GB")
        if raw_gb and raw_gb.strip():
            try:
                return max(0, int(float(raw_gb) * 1024 * 1024 * 1024))
            except ValueError:
                logger.warning(
                    "Invalid ARTIFACT_BLOB_MAX_GB=%r; using default %s",
                    raw_gb,
                    DEFAULT_AZURE_BLOB_QUOTA_BYTES,
                )
        return DEFAULT_AZURE_BLOB_QUOTA_BYTES

    def _current_storage_usage_bytes(self) -> Optional[int]:
        """Best-effort current storage usage bytes for quota enforcement."""
        getter = getattr(self.storage, "get_total_bytes", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception as e:
                logger.warning("Failed to compute storage usage for quota checks: %s", e)
                return None
        return None

    def _enforce_quota_before_write(
        self,
        *,
        incoming_bytes: int,
        artifact_type: str,
        logical_path: str,
    ) -> None:
        """
        Apply hard fail guardrail for Azure blob artifact writes.

        The guard is intentionally fail-closed once usage can be computed and the
        write would exceed configured quota.
        """
        backend_type = str(getattr(self.storage, "backend_type", "")).lower()
        if backend_type != "azure_blob":
            return
        quota_bytes = self._resolve_azure_quota_bytes()
        if quota_bytes <= 0:
            raise RuntimeError("Azure artifact quota is configured to zero bytes.")
        current_bytes = self._current_storage_usage_bytes()
        if current_bytes is None:
            # Fail-open if usage cannot be computed. We still log above so ops can fix.
            return
        projected = int(current_bytes) + int(incoming_bytes)
        if projected > quota_bytes:
            raise RuntimeError(
                "Artifact quota exceeded for Azure blob storage: "
                f"current_bytes={current_bytes}, incoming_bytes={incoming_bytes}, "
                f"projected_bytes={projected}, quota_bytes={quota_bytes}, "
                f"artifact_type={artifact_type}, logical_path={logical_path}"
            )

    def _write_artifact_bytes(
        self,
        *,
        logical_path: str,
        data: bytes,
        artifact_type: str,
        content_type: Optional[str] = None,
    ) -> str:
        started_at = time.perf_counter()
        try:
            self._enforce_quota_before_write(
                incoming_bytes=len(data),
                artifact_type=artifact_type,
                logical_path=logical_path,
            )
            uri = self.storage.write(logical_path, data, content_type=content_type)
            self._record_storage_event(
                operation="write",
                status="success",
                started_at=started_at,
                logical_path=logical_path,
                uri=uri,
                artifact_type=artifact_type,
            )
            return uri
        except Exception as e:
            self._record_storage_event(
                operation="write",
                status="failure",
                started_at=started_at,
                logical_path=logical_path,
                artifact_type=artifact_type,
                error=e,
            )
            raise

    def _read_artifact_bytes(
        self,
        *,
        uri: str,
        artifact_type: str,
        expected_sha256: Optional[str] = None,
        expected_byte_size: Optional[int] = None,
    ) -> bytes:
        started_at = time.perf_counter()
        try:
            if not self.storage.exists_from_uri(uri):
                raise FileNotFoundError(f"Artifact not found: {uri}")
            data = self.storage.read_from_uri(uri)
            if expected_byte_size is not None and int(expected_byte_size) != len(data):
                raise ValueError(
                    f"Artifact byte size mismatch for {uri}: "
                    f"expected={expected_byte_size}, actual={len(data)}"
                )
            if expected_sha256:
                actual_sha256 = hashlib.sha256(data).hexdigest()
                if actual_sha256 != str(expected_sha256):
                    raise ValueError(
                        f"Artifact checksum mismatch for {uri}: "
                        f"expected={expected_sha256}, actual={actual_sha256}"
                    )
            self._record_storage_event(
                operation="read",
                status="success",
                started_at=started_at,
                uri=uri,
                artifact_type=artifact_type,
            )
            return data
        except Exception as e:
            self._record_storage_event(
                operation="read",
                status="failure",
                started_at=started_at,
                uri=uri,
                artifact_type=artifact_type,
                error=e,
            )
            raise

    def get_operation_counts(self) -> Dict[str, int]:
        """Expose local artifact read/write counters for diagnostics/tests."""
        return dict(self._operation_counts)

    def _generate_logical_path(
        self, run_id: int, step_name: str, artifact_type: str, extension: str
    ) -> str:
        """
        Generate logical path for artifact (relative to storage base_dir).

        Pattern: {run_id}/{step_name}/{artifact_type}.{ext}

        Args:
            run_id: Run group ID
            step_name: Step name (e.g., "pca_projection", "clustering_k=5")
            artifact_type: Type of artifact (e.g., "sweep_results", "cluster_labels")
            extension: File extension (e.g., "json", "npy", "csv")

        Returns:
            Logical path for storage backend
        """
        safe_step_name = step_name.replace("/", "_").replace("\\", "_")
        return f"{run_id}/{safe_step_name}/{artifact_type}.{extension}"

    def _link_artifact_to_group(
        self,
        group_id: int,
        artifact_type: str,
        uri: str,
        content_type: Optional[str] = None,
        byte_size: Optional[int] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        call_id: Optional[int] = None,
    ) -> int:
        """
        Create CallArtifact entry and link to Group via metadata.

        Args:
            group_id: Group ID to link artifact to
            artifact_type: Type of artifact
            uri: URI/path to the artifact
            content_type: Optional MIME type
            byte_size: Optional size in bytes
            metadata_json: Optional additional metadata
            call_id: Optional RawCall ID to link artifact to.
                    If None, creates a placeholder RawCall.

        Returns:
            CallArtifact ID
        """
        if not self.repository:
            logger.warning("No repository provided, artifact not linked to database")
            return 0

        from ..db.models_v2 import CallArtifact

        # If no call_id provided, create a placeholder RawCall
        if call_id is None:
            call_id = self.repository.insert_raw_call(
                provider="artifact_service",
                request_json={"artifact_type": artifact_type, "group_id": group_id},
                modality="artifact",
                status="success",
                response_json={"uri": uri},
            )

        metadata = {
            "group_id": group_id,
            "schema_version": "artifact.v1",
            "governance_version": "blob_ops_phase2",
            "storage_backend": getattr(self.storage, "backend_type", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata_json:
            metadata.update(metadata_json)

        artifact = CallArtifact(
            call_id=call_id,
            artifact_type=artifact_type,
            uri=uri,
            content_type=content_type,
            byte_size=byte_size,
            metadata_json=metadata,
        )

        session = self.repository.session
        session.add(artifact)
        session.flush()
        session.refresh(artifact)

        logger.debug(
            f"Created artifact: id={artifact.id}, type={artifact_type}, "
            f"uri={uri}, group_id={group_id}"
        )

        return artifact.id

    def store_group_blob_artifact(
        self,
        group_id: int,
        step_name: str,
        logical_filename: str,
        data: bytes,
        artifact_type: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Persist arbitrary bytes for a Group (e.g. layer-0 dataset acquisition files).

        logical_filename must be unique within (group_id, step_name); path separators
        are sanitized to avoid traversal issues.

        Args:
            group_id: Target Group ID (e.g. dataset group)
            step_name: Folder segment under group_id (e.g. "acquisition")
            logical_filename: Filename or flattened relative path (e.g. "problem1.csv")
            data: Raw bytes to store
            artifact_type: CallArtifact.artifact_type (e.g. dataset_acquisition_file)
            content_type: Optional MIME type
            metadata: Extra metadata_json merged into artifact record

        Returns:
            CallArtifact ID
        """
        safe_step = step_name.replace("/", "_").replace("\\", "_")
        safe_file = str(logical_filename).replace("..", "_").replace("/", "_").replace("\\", "_")
        if not safe_file.strip():
            raise ValueError("logical_filename must be non-empty after sanitization")
        logical_path = f"{group_id}/{safe_step}/{safe_file}"
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type=artifact_type,
            content_type=content_type,
        )
        byte_size = len(data)
        artifact_metadata: Dict[str, Any] = {
            "step_name": step_name,
            "logical_filename": safe_file,
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=group_id,
            artifact_type=artifact_type,
            uri=str(uri),
            content_type=content_type,
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )
        logger.info(
            "Stored group blob artifact: artifact_id=%s group_id=%s type=%s path=%s",
            artifact_id,
            group_id,
            artifact_type,
            logical_path,
        )
        return artifact_id

    def store_sweep_results(
        self,
        run_id: int,
        sweep_results: Dict[str, Any],
        step_name: str = "sweep_complete",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save sweep results (by_k dict, metrics) as JSON artifact.

        Args:
            run_id: Run group ID
            sweep_results: Sweep results dict (by_k, pca_meta, etc.)
            step_name: Step name (default: "sweep_complete")
            metadata: Optional additional metadata

        Returns:
            CallArtifact ID
        """
        logical_path = self._generate_logical_path(
            run_id, step_name, "sweep_results", "json"
        )
        data = json.dumps(sweep_results, indent=2, default=str).encode("utf-8")
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="sweep_results",
            content_type="application/json",
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "json",
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=run_id,
            artifact_type="sweep_results",
            uri=uri,
            content_type="application/json",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            f"Stored sweep results: artifact_id={artifact_id}, "
            f"run_id={run_id}, uri={uri}"
        )
        return artifact_id

    def store_dataset_snapshot_manifest(
        self,
        snapshot_group_id: int,
        snapshot_name: str,
        entries: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save a dataset snapshot manifest (JSON) and link it to a snapshot group.

        Args:
            snapshot_group_id: Group ID of dataset snapshot
            snapshot_name: Snapshot identifier (e.g., "dbpedia_286_seed42_labeled")
            entries: Manifest entries. Convention:
                [{"position": int, "source_id": str|int, "text": str, "label": int|None}, ...]
            metadata: Optional additional metadata

        Returns:
            CallArtifact ID
        """
        payload = {
            "snapshot_name": snapshot_name,
            "entries": entries,
        }
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        manifest_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        logical_path = self._generate_logical_path(
            snapshot_group_id,
            "snapshot_manifest",
            "dataset_snapshot_manifest",
            "json",
        )
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="dataset_snapshot_manifest",
            content_type="application/json",
        )
        byte_size = len(data)
        artifact_metadata = {
            "snapshot_name": snapshot_name,
            "artifact_format": "json",
            "manifest_hash": manifest_hash,
            "entry_count": len(entries),
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=snapshot_group_id,
            artifact_type="dataset_snapshot_manifest",
            uri=str(uri),
            content_type="application/json",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            "Stored dataset snapshot manifest: artifact_id=%s, snapshot_group_id=%s, entries=%s",
            artifact_id,
            snapshot_group_id,
            len(entries),
        )
        return artifact_id

    def store_embedding_matrix(
        self,
        embedding_batch_group_id: int,
        matrix: np.ndarray,
        *,
        dataset_key: str,
        embedding_engine: str,
        provider: str,
        entry_max: int,
        key_version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist an embedding matrix artifact and link it to embedding_batch group."""
        logical_path = self._generate_logical_path(
            embedding_batch_group_id,
            "embedding_matrix",
            "embedding_matrix",
            "npy",
        )
        buf = io.BytesIO()
        np.save(buf, matrix)
        data = buf.getvalue()
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="embedding_matrix",
            content_type="application/octet-stream",
        )
        byte_size = len(data)
        artifact_metadata = {
            "dataset_key": dataset_key,
            "embedding_engine": embedding_engine,
            "provider": provider,
            "entry_max": int(entry_max),
            "key_version": key_version,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if metadata:
            artifact_metadata.update(metadata)
        return self._link_artifact_to_group(
            group_id=embedding_batch_group_id,
            artifact_type="embedding_matrix",
            uri=str(uri),
            content_type="application/octet-stream",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

    def find_embedding_matrix_artifact(
        self,
        *,
        dataset_key: str,
        embedding_engine: str,
        provider: str,
        entry_max: int,
        key_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Find matching embedding_matrix artifact metadata/URI in DB."""
        if not self.repository:
            return None
        from ..db.models_v2 import CallArtifact

        session = self.repository.session
        artifacts = (
            session.query(CallArtifact)
            .filter(CallArtifact.artifact_type == "embedding_matrix")
            .order_by(CallArtifact.id.desc())
            .all()
        )
        for art in artifacts:
            md = art.metadata_json or {}
            if not isinstance(md, dict):
                continue
            if (
                md.get("dataset_key") == dataset_key
                and md.get("embedding_engine") == embedding_engine
                and md.get("provider") == provider
                and int(md.get("entry_max", -1)) == int(entry_max)
                and md.get("key_version") == key_version
            ):
                return {
                    "artifact_id": int(art.id),
                    "uri": str(art.uri),
                    "metadata": md,
                    "group_id": md.get("group_id"),
                }
        return None

    def store_cluster_labels(
        self,
        run_id: int,
        labels: np.ndarray,
        step_name: str,
        k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save cluster labels array as NPY artifact.

        Args:
            run_id: Run group ID
            labels: Cluster labels array (1D numpy array)
            step_name: Step name (e.g., "clustering_k=5")
            k: Optional cluster count (for metadata)
            metadata: Optional additional metadata

        Returns:
            CallArtifact ID
        """
        logical_path = self._generate_logical_path(
            run_id, step_name, "cluster_labels", "npy"
        )
        buf = io.BytesIO()
        np.save(buf, labels)
        data = buf.getvalue()
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="cluster_labels",
            content_type="application/octet-stream",
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "npy",
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if k is not None:
            artifact_metadata["k"] = k
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=run_id,
            artifact_type="cluster_labels",
            uri=str(uri),
            content_type="application/octet-stream",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            f"Stored cluster labels: artifact_id={artifact_id}, "
            f"run_id={run_id}, step={step_name}, shape={labels.shape}"
        )

        return artifact_id

    def store_pca_components(
        self,
        run_id: int,
        components: np.ndarray,
        step_name: str = "pca_projection",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save PCA components/vectors as NPY artifact.

        Args:
            run_id: Run group ID
            components: PCA components array (2D numpy array)
            step_name: Step name (default: "pca_projection")
            metadata: Optional additional metadata

        Returns:
            CallArtifact ID
        """
        logical_path = self._generate_logical_path(
            run_id, step_name, "pca_components", "npy"
        )
        buf = io.BytesIO()
        np.save(buf, components)
        data = buf.getvalue()
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="pca_components",
            content_type="application/octet-stream",
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "npy",
            "shape": list(components.shape),
            "dtype": str(components.dtype),
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=run_id,
            artifact_type="pca_components",
            uri=str(uri),
            content_type="application/octet-stream",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            f"Stored PCA components: artifact_id={artifact_id}, "
            f"run_id={run_id}, step={step_name}, shape={components.shape}"
        )

        return artifact_id

    def store_metrics(
        self,
        run_id: int,
        metrics: Dict[str, Any],
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save computed metrics (silhouette, ARI, coverage) as JSON artifact.

        Args:
            run_id: Run group ID
            metrics: Metrics dict (e.g., {"silhouette": 0.5, "ari": 0.8})
            step_name: Step name (e.g., "metrics_k=5")
            metadata: Optional additional metadata

        Returns:
            CallArtifact ID
        """
        logical_path = self._generate_logical_path(
            run_id, step_name, "metrics", "json"
        )
        data = json.dumps(metrics, indent=2, default=str).encode("utf-8")
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="metrics",
            content_type="application/json",
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "json",
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=run_id,
            artifact_type="metrics",
            uri=str(uri),
            content_type="application/json",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            f"Stored metrics: artifact_id={artifact_id}, "
            f"run_id={run_id}, step={step_name}"
        )

        return artifact_id

    def store_representatives(
        self,
        run_id: int,
        representatives: List[str],
        step_name: str,
        k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save cluster representatives as CSV artifact.

        Args:
            run_id: Run group ID
            representatives: List of representative texts
            step_name: Step name (e.g., "representatives_k=5")
            k: Optional cluster count (for metadata)
            metadata: Optional additional metadata

        Returns:
            CallArtifact ID
        """
        logical_path = self._generate_logical_path(
            run_id, step_name, "representatives", "csv"
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["index", "representative"])
        for idx, rep in enumerate(representatives):
            writer.writerow([idx, rep])
        data = buf.getvalue().encode("utf-8")
        uri = self._write_artifact_bytes(
            logical_path=logical_path,
            data=data,
            artifact_type="representatives",
            content_type="text/csv",
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "csv",
            "count": len(representatives),
        }
        artifact_metadata.update(self._integrity_metadata(data))
        if k is not None:
            artifact_metadata["k"] = k
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=run_id,
            artifact_type="representatives",
            uri=str(uri),
            content_type="text/csv",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            f"Stored representatives: artifact_id={artifact_id}, "
            f"run_id={run_id}, step={step_name}, count={len(representatives)}"
        )

        return artifact_id

    def load_artifact(
        self,
        uri: str,
        artifact_type: str,
        *,
        expected_sha256: Optional[str] = None,
        expected_byte_size: Optional[int] = None,
    ) -> Union[Dict[str, Any], np.ndarray, list]:
        """
        Load an artifact from URI.

        Args:
            uri: Artifact URI (from storage.get_uri)
            artifact_type: Type of artifact (determines loading method)
            expected_sha256: Optional checksum expectation for integrity verification
            expected_byte_size: Optional byte-size expectation for integrity verification

        Returns:
            Loaded artifact data (dict for JSON, np.ndarray for NPY, list for CSV)
        """
        data = self._read_artifact_bytes(
            uri=uri,
            artifact_type=artifact_type,
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
        )

        if artifact_type in (
            "sweep_results",
            "metrics",
            "dataset_snapshot_manifest",
            "dataset_acquisition_manifest",
        ):
            return json.loads(data.decode("utf-8"))

        if artifact_type in ("cluster_labels", "pca_components", "embedding_matrix"):
            return np.load(io.BytesIO(data))

        if artifact_type == "representatives":
            representatives = []
            reader = csv.reader(io.StringIO(data.decode("utf-8")))
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    representatives.append(row[1])
            return representatives

        raise ValueError(f"Unknown artifact type: {artifact_type}")
