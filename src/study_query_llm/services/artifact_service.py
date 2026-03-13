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
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import numpy as np

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository
    from ..storage.protocol import StorageBackend

logger = get_logger(__name__)

# Default artifact directory
DEFAULT_ARTIFACT_DIR = "artifacts"


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
        if storage_backend is not None:
            self.storage = storage_backend
        else:
            from ..storage.local import LocalStorageBackend

            self.storage = LocalStorageBackend(base_dir=artifact_dir)

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
        uri = self.storage.write(
            logical_path, data, content_type="application/json"
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "json",
        }
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
        uri = self.storage.write(
            logical_path, data, content_type="application/json"
        )
        byte_size = len(data)
        artifact_metadata = {
            "snapshot_name": snapshot_name,
            "artifact_format": "json",
            "manifest_hash": manifest_hash,
            "entry_count": len(entries),
        }
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
        uri = self.storage.write(
            logical_path, data, content_type="application/octet-stream"
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
        uri = self.storage.write(
            logical_path, data, content_type="application/octet-stream"
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "npy",
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
        }
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
        uri = self.storage.write(
            logical_path, data, content_type="application/octet-stream"
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "npy",
            "shape": list(components.shape),
            "dtype": str(components.dtype),
        }
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
        uri = self.storage.write(
            logical_path, data, content_type="application/json"
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "json",
        }
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
        uri = self.storage.write(
            logical_path, data, content_type="text/csv"
        )
        byte_size = len(data)

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "csv",
            "count": len(representatives),
        }
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

    def load_artifact(self, uri: str, artifact_type: str) -> Union[Dict[str, Any], np.ndarray, list]:
        """
        Load an artifact from URI.

        Args:
            uri: Artifact URI (from storage.get_uri)
            artifact_type: Type of artifact (determines loading method)

        Returns:
            Loaded artifact data (dict for JSON, np.ndarray for NPY, list for CSV)
        """
        if not self.storage.exists_from_uri(uri):
            raise FileNotFoundError(f"Artifact not found: {uri}")

        data = self.storage.read_from_uri(uri)

        if artifact_type in ("sweep_results", "metrics", "dataset_snapshot_manifest"):
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
