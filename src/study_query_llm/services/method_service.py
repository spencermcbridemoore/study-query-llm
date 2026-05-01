"""
Method Service - Versioned analysis method definitions and result tracking.

Provides structured provenance for analysis methods (extraction, metrics, etc.)
and the results they produce. Supports default-to-latest version lookup.

Parameters convention (soft): When a MethodDefinition has parameters_schema set,
callers may include a "parameters" key in result_json whose shape matches that
schema. This links the parameters used for a run to the method's definition.
Validation is optional; the convention supports queryability and documentation.

Usage:
    from study_query_llm.services.method_service import MethodService
    from study_query_llm.db.raw_call_repository import RawCallRepository

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        method_id = method_svc.register_method(
            name="extract_correct_answers",
            version="2.1",
            code_ref="scripts/parse_quiz.py",
            code_commit="abc123",
        )
        method_svc.record_result(
            method_definition_id=method_id,
            source_group_id=42,
            result_key="chi_square",
            result_value=12.3,
        )

    With parameters convention (when method has parameters_schema):
        method_svc.record_result(
            method_definition_id=method_id,
            source_group_id=run_id,
            result_key="graph_output",
            result_json={
                "parameters": {"prompt": "hello", "k": 5},  # matches parameters_schema
                "state": {"output": "..."},
            },
        )
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Mapping

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


@dataclass(frozen=True)
class MethodInputRequirements:
    """Normalized required-input contract for an analysis method."""

    snapshot: bool = True
    embedding_batch: bool = True


class MethodService:
    """
    Service for registering versioned analysis methods and recording results.

    Methods are versioned; only one version per name is active at a time.
    get_method(name) returns the active version by default.
    """

    def __init__(self, repository: "RawCallRepository"):
        """
        Initialize the method service.

        Args:
            repository: RawCallRepository instance for database access
        """
        self.repository = repository

    @staticmethod
    def _coerce_required_input_flag(value: Any, *, default: bool) -> bool:
        """Coerce required-input flags from persisted JSON-like payloads."""
        if isinstance(value, bool):
            return value
        return default

    @classmethod
    def resolve_input_requirements_from_schema(
        cls,
        input_schema: Mapping[str, Any] | None,
    ) -> MethodInputRequirements:
        """Resolve input requirements from a method definition input schema.

        Read behavior is intentionally backward-compatible:
        - absent/invalid schema -> default requirements
        - absent/invalid required_inputs -> default requirements
        - non-boolean flags -> default per field
        """
        schema = dict(input_schema) if isinstance(input_schema, Mapping) else {}
        required_inputs = schema.get("required_inputs")
        if not isinstance(required_inputs, Mapping):
            return MethodInputRequirements()
        return MethodInputRequirements(
            snapshot=cls._coerce_required_input_flag(
                required_inputs.get("snapshot"),
                default=True,
            ),
            embedding_batch=cls._coerce_required_input_flag(
                required_inputs.get("embedding_batch"),
                default=True,
            ),
        )

    def resolve_method_input_requirements(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> MethodInputRequirements:
        """Resolve normalized input requirements for a registered method."""
        method = self.get_method(name, version=version)
        if method is None and version is None:
            method = self.get_method(name)
        input_schema = (
            method.input_schema
            if method is not None and isinstance(method.input_schema, Mapping)
            else None
        )
        return self.resolve_input_requirements_from_schema(input_schema)

    @staticmethod
    def _validate_input_schema_for_registration(
        input_schema: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Validate required-input contract shape on new/updated registrations."""
        if input_schema is None:
            return None
        if not isinstance(input_schema, Mapping):
            raise ValueError("input_schema must be a JSON object when provided")
        normalized_input_schema = dict(input_schema)
        required_inputs = normalized_input_schema.get("required_inputs")
        if required_inputs is None:
            return normalized_input_schema
        if not isinstance(required_inputs, Mapping):
            raise ValueError("input_schema.required_inputs must be a JSON object")
        required_inputs_dict = dict(required_inputs)
        for key in ("snapshot", "embedding_batch"):
            if key in required_inputs_dict and not isinstance(required_inputs_dict[key], bool):
                raise ValueError(
                    f"input_schema.required_inputs.{key} must be a boolean when provided"
                )
        normalized_input_schema["required_inputs"] = required_inputs_dict
        return normalized_input_schema

    def register_method(
        self,
        name: str,
        version: str,
        code_ref: Optional[str] = None,
        code_commit: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
        recipe_json: Optional[Dict[str, Any]] = None,
        parent_version_id: Optional[int] = None,
    ) -> int:
        """
        Register a new method definition and set it as the active version.

        Deactivates any previously active version with the same name.

        Args:
            name: Method name (e.g., "extract_correct_answers")
            version: Version string (e.g., "2.1")
            code_ref: Path to code (e.g., "scripts/parse_quiz.py")
            code_commit: Git SHA
            description: Optional description
            input_schema: JSON describing expected input
            output_schema: JSON describing output shape
            parameters_schema: JSON describing configurable knobs. When set,
                results may include result_json["parameters"] matching this shape.
            recipe_json: Optional structured recipe for composite/pipeline
                methods describing ordered component stages. See
                ``study_query_llm.algorithms.recipes`` and
                ``docs/living/METHOD_RECIPES.md`` for the v0 shape.
            parent_version_id: FK to previous version (nullable for v1)

        Returns:
            The ID of the new MethodDefinition
        """
        from ..db.models_v2 import MethodDefinition

        session = self.repository.session
        validated_input_schema = self._validate_input_schema_for_registration(input_schema)

        # Deactivate previous active version with same name
        existing = (
            session.query(MethodDefinition)
            .filter(
                MethodDefinition.name == name,
                MethodDefinition.is_active.is_(True),
            )
            .all()
        )
        for m in existing:
            m.is_active = False

        method = MethodDefinition(
            name=name,
            version=version,
            is_active=True,
            description=description,
            code_ref=code_ref,
            code_commit=code_commit,
            input_schema=validated_input_schema,
            output_schema=output_schema,
            parameters_schema=parameters_schema,
            recipe_json=recipe_json,
            parent_version_id=parent_version_id,
        )
        session.add(method)
        session.flush()
        session.refresh(method)

        logger.info(
            f"Registered method: id={method.id}, name={name}, version={version}"
        )
        return method.id

    def update_recipe(
        self,
        method_definition_id: int,
        recipe_json: Dict[str, Any],
    ) -> bool:
        """
        Attach or replace the ``recipe_json`` for an existing method row
        in-place (no version bump).

        Use this when a composite method was registered before its recipe was
        available (e.g. lazy auto-registration) and the recipe is being
        back-filled. Adding a recipe to an existing row is not a semantic
        change warranting a new version; callers who do want a semantic change
        should call :meth:`register_method` with a new version string instead.

        Args:
            method_definition_id: ID of the MethodDefinition to update.
            recipe_json: Recipe dict conforming to the v0 shape.

        Returns:
            True if the row was updated, False if not found.
        """
        from ..db.models_v2 import MethodDefinition

        method = (
            self.repository.session.query(MethodDefinition)
            .filter(MethodDefinition.id == method_definition_id)
            .first()
        )
        if method is None:
            return False
        method.recipe_json = recipe_json
        self.repository.session.flush()
        logger.info(
            "Updated recipe_json for method id=%s name=%s version=%s",
            method.id,
            method.name,
            method.version,
        )
        return True

    def get_method(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get a method definition by name and optional version.

        If version is None, returns the active version for that name.

        Args:
            name: Method name
            version: Optional version string; if None, returns active version

        Returns:
            MethodDefinition instance or None if not found
        """
        from ..db.models_v2 import MethodDefinition

        session = self.repository.session

        if version is not None:
            return (
                session.query(MethodDefinition)
                .filter(
                    MethodDefinition.name == name,
                    MethodDefinition.version == version,
                )
                .first()
            )
        return (
            session.query(MethodDefinition)
            .filter(
                MethodDefinition.name == name,
                MethodDefinition.is_active.is_(True),
            )
            .first()
        )

    def record_result(
        self,
        method_definition_id: int,
        source_group_id: int,
        result_key: str,
        result_value: Optional[float] = None,
        result_json: Optional[Dict[str, Any]] = None,
        analysis_group_id: Optional[int] = None,
    ) -> int:
        """
        Record an analysis result.

        Args:
            method_definition_id: ID of the MethodDefinition that produced this
            source_group_id: ID of the Group (sweep/run) that was analyzed
            result_key: Metric name (e.g., "chi_square", "ari")
            result_value: Numeric scalar (optional)
            result_json: Structured data (optional). When the method has
                parameters_schema, prefer including a "parameters" key with
                the run parameters used (e.g., payload from the job).
            analysis_group_id: Optional Group ID for full provenance chain

        Returns:
            The ID of the new AnalysisResult
        """
        from ..db.models_v2 import AnalysisResult

        result = AnalysisResult(
            method_definition_id=method_definition_id,
            source_group_id=source_group_id,
            analysis_group_id=analysis_group_id,
            result_key=result_key,
            result_value=result_value,
            result_json=result_json,
        )
        self.repository.session.add(result)
        self.repository.session.flush()
        self.repository.session.refresh(result)

        logger.debug(
            f"Recorded result: id={result.id}, key={result_key}, "
            f"source_group={source_group_id}"
        )
        return result.id

    def query_results(
        self,
        method_name: Optional[str] = None,
        method_version: Optional[str] = None,
        source_group_id: Optional[int] = None,
        result_key: Optional[str] = None,
    ) -> List[Any]:
        """
        Query analysis results with optional filters.

        Args:
            method_name: Filter by method name
            method_version: Filter by method version (use with method_name)
            source_group_id: Filter by source group
            result_key: Filter by result key

        Returns:
            List of AnalysisResult instances
        """
        from ..db.models_v2 import AnalysisResult, MethodDefinition

        session = self.repository.session
        query = session.query(AnalysisResult).join(MethodDefinition)

        if method_name is not None:
            query = query.filter(MethodDefinition.name == method_name)
        if method_version is not None:
            query = query.filter(MethodDefinition.version == method_version)
        if source_group_id is not None:
            query = query.filter(
                AnalysisResult.source_group_id == source_group_id
            )
        if result_key is not None:
            query = query.filter(AnalysisResult.result_key == result_key)

        return query.order_by(AnalysisResult.created_at.desc()).all()
