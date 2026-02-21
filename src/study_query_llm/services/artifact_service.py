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

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import numpy as np

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


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
        artifact_dir: str = "artifacts",
    ):
        """
        Initialize the artifact service.

        Args:
            repository: Optional RawCallRepository for DB persistence
            artifact_dir: Base directory for storing artifacts (default: "artifacts")
        """
        self.repository = repository
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _generate_uri(
        self, run_id: int, step_name: str, artifact_type: str, extension: str
    ) -> str:
        """
        Generate artifact URI following conventions.

        Pattern: artifacts/{run_id}/{step_name}/{artifact_type}.{ext}

        Args:
            run_id: Run group ID
            step_name: Step name (e.g., "pca_projection", "clustering_k=5")
            artifact_type: Type of artifact (e.g., "sweep_results", "cluster_labels")
            extension: File extension (e.g., "json", "npy", "csv")

        Returns:
            Relative URI path
        """
        # Sanitize step_name for filesystem
        safe_step_name = step_name.replace("/", "_").replace("\\", "_")
        uri = f"{self.artifact_dir}/{run_id}/{safe_step_name}/{artifact_type}.{extension}"
        return uri

    def _ensure_directory(self, uri: str) -> Path:
        """
        Ensure directory exists for artifact URI.

        Args:
            uri: Artifact URI (relative or absolute)

        Returns:
            Path to the artifact file
        """
        artifact_path = Path(uri)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        return artifact_path

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
        uri = self._generate_uri(run_id, step_name, "sweep_results", "json")
        artifact_path = self._ensure_directory(uri)

        # Save as JSON
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2, default=str)

        byte_size = artifact_path.stat().st_size

        artifact_metadata = {
            "step_name": step_name,
            "artifact_format": "json",
        }
        if metadata:
            artifact_metadata.update(metadata)

        artifact_id = self._link_artifact_to_group(
            group_id=run_id,
            artifact_type="sweep_results",
            uri=str(uri),
            content_type="application/json",
            byte_size=byte_size,
            metadata_json=artifact_metadata,
        )

        logger.info(
            f"Stored sweep results: artifact_id={artifact_id}, "
            f"run_id={run_id}, uri={uri}"
        )

        return artifact_id

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
        uri = self._generate_uri(run_id, step_name, "cluster_labels", "npy")
        artifact_path = self._ensure_directory(uri)

        # Save as NPY
        np.save(artifact_path, labels)

        byte_size = artifact_path.stat().st_size

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
        uri = self._generate_uri(run_id, step_name, "pca_components", "npy")
        artifact_path = self._ensure_directory(uri)

        # Save as NPY
        np.save(artifact_path, components)

        byte_size = artifact_path.stat().st_size

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
        uri = self._generate_uri(run_id, step_name, "metrics", "json")
        artifact_path = self._ensure_directory(uri)

        # Save as JSON
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        byte_size = artifact_path.stat().st_size

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
        import csv

        uri = self._generate_uri(run_id, step_name, "representatives", "csv")
        artifact_path = self._ensure_directory(uri)

        # Save as CSV
        with open(artifact_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "representative"])
            for idx, rep in enumerate(representatives):
                writer.writerow([idx, rep])

        byte_size = artifact_path.stat().st_size

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
            uri: Artifact URI
            artifact_type: Type of artifact (determines loading method)

        Returns:
            Loaded artifact data (dict for JSON, np.ndarray for NPY, list for CSV)
        """
        artifact_path = Path(uri)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {uri}")

        if artifact_type in ("sweep_results", "metrics"):
            # Load JSON
            with open(artifact_path, "r", encoding="utf-8") as f:
                return json.load(f)

        elif artifact_type in ("cluster_labels", "pca_components"):
            # Load NPY
            return np.load(artifact_path)

        elif artifact_type == "representatives":
            # Load CSV
            import csv

            representatives = []
            with open(artifact_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        representatives.append(row[1])
            return representatives

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
