"""
Provenance Service - Track algorithm runs, experiments, and data provenance.

This service establishes standard conventions for using Group and GroupMember
to track algorithm runs, experiments, and data provenance.

Usage:
    from study_query_llm.services.provenance_service import ProvenanceService
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository

    db = DatabaseConnectionV2("postgresql://...")
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        # Create a run group
        run_id = provenance.create_run_group(
            algorithm="pca_kllmeans_sweep",
            config={"k_min": 2, "k_max": 10, "pca_dim": 64}
        )

        # Link embedding calls to the run
        provenance.link_raw_calls_to_group(run_id, [call_id1, call_id2])
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime, timezone

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


# Standard group types
GROUP_TYPE_DATASET = "dataset"
GROUP_TYPE_EMBEDDING_BATCH = "embedding_batch"
GROUP_TYPE_RUN = "run"
GROUP_TYPE_STEP = "step"
GROUP_TYPE_METRICS = "metrics"
GROUP_TYPE_SUMMARIZATION_BATCH = "summarization_batch"


class ProvenanceService:
    """
    Service for tracking algorithm runs, experiments, and data provenance.

    This service provides methods to create and manage groups for tracking
    algorithm executions, linking RawCalls and artifacts to runs, and
    querying provenance information.

    Standard Group Types:
    - `dataset`: Input data collection (links to embedding RawCalls)
    - `embedding_batch`: Batch of embeddings created together
    - `run`: Complete algorithm execution (e.g., PCA+KLLMeans sweep)
    - `step`: Individual step within a run (e.g., "pca_projection", "clustering_k=5")
    - `metrics`: Computed metrics/analysis results
    - `summarization_batch`: Batch of LLM summarization calls
    """

    def __init__(self, repository: "RawCallRepository"):
        """
        Initialize the provenance service.

        Args:
            repository: RawCallRepository instance for database access
        """
        self.repository = repository

    def create_run_group(
        self,
        algorithm: str,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Create a run group for tracking a complete algorithm execution.

        Args:
            algorithm: Algorithm name (e.g., "pca_kllmeans_sweep", "clustering_analysis")
            config: Algorithm configuration dict (e.g., {"k_min": 2, "k_max": 10, "pca_dim": 64})
            name: Optional custom name for the run (default: auto-generated)
            description: Optional description

        Returns:
            Group ID of the created run group
        """
        if name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = f"{algorithm}_{timestamp}"

        metadata_json = {
            "algorithm": algorithm,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if config:
            metadata_json["config"] = config

        group_id = self.repository.create_group(
            group_type=GROUP_TYPE_RUN,
            name=name,
            description=description or f"Algorithm run: {algorithm}",
            metadata_json=metadata_json,
        )

        logger.info(f"Created run group: id={group_id}, algorithm={algorithm}, name={name}")
        return group_id

    def create_step_group(
        self,
        parent_run_id: int,
        step_name: str,
        step_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a step group within a run.

        Args:
            parent_run_id: ID of the parent run group
            step_name: Name of the step (e.g., "pca_projection", "clustering_k=5")
            step_type: Optional step type (e.g., "pca", "clustering", "metrics")
            metadata: Optional additional metadata

        Returns:
            Group ID of the created step group
        """
        metadata_json = {
            "parent_run_id": parent_run_id,
            "step_name": step_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if step_type:
            metadata_json["step_type"] = step_type
        if metadata:
            metadata_json.update(metadata)

        group_id = self.repository.create_group(
            group_type=GROUP_TYPE_STEP,
            name=f"step_{step_name}",
            description=f"Step: {step_name}",
            metadata_json=metadata_json,
        )

        logger.info(
            f"Created step group: id={group_id}, parent_run={parent_run_id}, step={step_name}"
        )
        return group_id

    def link_raw_calls_to_group(
        self,
        group_id: int,
        call_ids: List[int],
        role: Optional[str] = None,
        position: Optional[int] = None,
    ) -> List[int]:
        """
        Link RawCalls to a group via GroupMember.

        Args:
            group_id: ID of the group
            call_ids: List of RawCall IDs to link
            role: Optional role for the calls (e.g., "input", "output", "intermediate")
            position: Optional position for ordering

        Returns:
            List of GroupMember IDs created
        """
        member_ids = []
        for idx, call_id in enumerate(call_ids):
            member_id = self.repository.add_call_to_group(
                group_id=group_id,
                call_id=call_id,
                role=role,
                position=position if position is not None else idx,
            )
            member_ids.append(member_id)

        logger.debug(
            f"Linked {len(call_ids)} RawCalls to group {group_id} "
            f"(role={role}, position={position})"
        )
        return member_ids

    def link_artifacts_to_group(
        self,
        group_id: int,
        artifacts: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Create CallArtifact entries and link them to a group.

        Args:
            group_id: ID of the group
            artifacts: List of artifact dicts, each with:
                - artifact_type: Type of artifact ('image', 'audio', 'video', 'json', 'npy', 'csv', etc.)
                - uri: URI/path to the artifact
                - content_type: Optional MIME type
                - byte_size: Optional size in bytes
                - metadata_json: Optional additional metadata
                - call_id: Optional RawCall ID to link artifact to

        Returns:
            List of CallArtifact IDs created
        """
        from ..db.models_v2 import CallArtifact

        artifact_ids = []
        session = self.repository.session

        for artifact_data in artifacts:
            artifact = CallArtifact(
                call_id=artifact_data.get("call_id"),  # Optional
                artifact_type=artifact_data["artifact_type"],
                uri=artifact_data["uri"],
                content_type=artifact_data.get("content_type"),
                byte_size=artifact_data.get("byte_size"),
                metadata_json={
                    **(artifact_data.get("metadata_json") or {}),
                    "group_id": group_id,
                },
            )

            session.add(artifact)
            session.flush()
            session.refresh(artifact)
            artifact_ids.append(artifact.id)

        logger.debug(f"Created {len(artifacts)} artifacts linked to group {group_id}")
        return artifact_ids

    def get_run_provenance(self, run_id: int) -> Dict[str, Any]:
        """
        Query all RawCalls, artifacts, and sub-groups for a run.

        Args:
            run_id: ID of the run group

        Returns:
            Dict with:
                - run_group: Group object for the run
                - raw_calls: List of RawCall objects linked to the run
                - step_groups: List of step groups within the run
                - artifacts: List of CallArtifact objects linked to the run
                - metadata: Run metadata from group.metadata_json
        """
        from ..db.models_v2 import Group, GroupMember, CallArtifact

        session = self.repository.session

        # Get run group
        run_group = self.repository.get_group_by_id(run_id)
        if not run_group:
            raise ValueError(f"Run group {run_id} not found")

        # Get all RawCalls linked to the run
        raw_calls = self.repository.get_calls_in_group(run_id)

        # Get all step groups (groups with parent_run_id in metadata)
        all_groups = session.query(Group).filter(Group.group_type == GROUP_TYPE_STEP).all()
        step_groups = [
            g
            for g in all_groups
            if (
                g.metadata_json
                and isinstance(g.metadata_json, dict)
                and g.metadata_json.get("parent_run_id") == run_id
            )
        ]

        # Get all artifacts linked to the run (via metadata_json.group_id)
        all_artifacts = session.query(CallArtifact).all()
        artifacts = [
            a
            for a in all_artifacts
            if (
                a.metadata_json
                and isinstance(a.metadata_json, dict)
                and a.metadata_json.get("group_id") == run_id
            )
        ]

        # Also get artifacts linked to RawCalls in this run
        call_ids = [call.id for call in raw_calls]
        if call_ids:
            call_artifacts = (
                session.query(CallArtifact).filter(CallArtifact.call_id.in_(call_ids)).all()
            )
            # Deduplicate
            artifact_ids = {a.id for a in artifacts}
            for a in call_artifacts:
                if a.id not in artifact_ids:
                    artifacts.append(a)

        return {
            "run_group": run_group,
            "raw_calls": raw_calls,
            "step_groups": step_groups,
            "artifacts": artifacts,
            "metadata": run_group.metadata_json or {},
        }

    def create_dataset_group(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a dataset group for input data collection.

        Args:
            name: Dataset name
            description: Optional description
            metadata: Optional metadata (e.g., source, size, format)

        Returns:
            Group ID of the created dataset group
        """
        metadata_json = {
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            metadata_json.update(metadata)

        group_id = self.repository.create_group(
            group_type=GROUP_TYPE_DATASET,
            name=name,
            description=description or f"Dataset: {name}",
            metadata_json=metadata_json,
        )

        logger.info(f"Created dataset group: id={group_id}, name={name}")
        return group_id

    def create_embedding_batch_group(
        self,
        name: Optional[str] = None,
        deployment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a group for a batch of embeddings created together.

        Args:
            name: Optional batch name (default: auto-generated)
            deployment: Optional embedding deployment/model name
            metadata: Optional metadata

        Returns:
            Group ID of the created embedding batch group
        """
        if name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = f"embedding_batch_{timestamp}"

        metadata_json = {
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if deployment:
            metadata_json["deployment"] = deployment
        if metadata:
            metadata_json.update(metadata)

        group_id = self.repository.create_group(
            group_type=GROUP_TYPE_EMBEDDING_BATCH,
            name=name,
            description=f"Embedding batch: {name}",
            metadata_json=metadata_json,
        )

        logger.info(f"Created embedding batch group: id={group_id}, name={name}")
        return group_id

    def create_summarization_batch_group(
        self,
        name: Optional[str] = None,
        llm_deployment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a group for a batch of LLM summarization calls.

        Args:
            name: Optional batch name (default: auto-generated)
            llm_deployment: Optional LLM deployment name
            metadata: Optional metadata

        Returns:
            Group ID of the created summarization batch group
        """
        if name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = f"summarization_batch_{timestamp}"

        metadata_json = {
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if llm_deployment:
            metadata_json["llm_deployment"] = llm_deployment
        if metadata:
            metadata_json.update(metadata)

        group_id = self.repository.create_group(
            group_type=GROUP_TYPE_SUMMARIZATION_BATCH,
            name=name,
            description=f"Summarization batch: {name}",
            metadata_json=metadata_json,
        )

        logger.info(f"Created summarization batch group: id={group_id}, name={name}")
        return group_id
