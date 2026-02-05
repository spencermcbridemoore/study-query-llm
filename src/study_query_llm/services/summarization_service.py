"""
Summarization Service - LLM-based summarization/paraphrasing with provenance tracking.

This service handles batch summarization of texts using LLM deployments,
with full logging to RawCall and integration with grouping for experiment tracking.

Usage:
    from study_query_llm.services.summarization_service import SummarizationService
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository

    db = DatabaseConnectionV2("postgresql://...")
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        # Summarize a batch
        request = SummarizationRequest(
            texts=["Text 1", "Text 2", "Text 3"],
            llm_deployment="gpt-4",
            group_id=run_id
        )
        result = await service.summarize_batch(request)
        print(result.summaries)
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING

from ..config import Config
from ..providers.factory import ProviderFactory
from ..services.inference_service import InferenceService
from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


@dataclass
class SummarizationRequest:
    """Request parameters for summarization."""

    texts: List[str]
    llm_deployment: str
    temperature: float = 0.2
    max_tokens: int = 128
    group_id: Optional[int] = None
    provider: str = "azure"
    validate_deployment: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummarizationResponse:
    """Response from batch summarization."""

    summaries: List[str]
    raw_call_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SummarizationService:
    """
    Service for LLM-based summarization/paraphrasing with provenance tracking.

    Features:
    - Batch processing with asyncio
    - RawCall logging for all calls (success/failure)
    - Group integration for experiment tracking
    - Deployment validation
    - Error handling with failed call logging
    """

    def __init__(
        self,
        repository: Optional["RawCallRepository"] = None,
        require_db_persistence: bool = True,
    ):
        """
        Initialize the summarization service.

        Args:
            repository: Optional RawCallRepository for DB persistence
            require_db_persistence: If True (default), raise exception if DB save fails when repository is provided.
                                   If False, log warning and continue (graceful degradation).
                                   Ignored if repository is None.
        """
        self.repository = repository
        self.require_db_persistence = require_db_persistence

    async def _validate_deployment(
        self, deployment: str, provider: str = "azure"
    ) -> bool:
        """
        Validate that a deployment exists and supports text completion.

        Args:
            deployment: Deployment name to validate
            provider: Provider name (default: "azure")

        Returns:
            True if deployment is valid, False otherwise
        """
        try:
            # Create fresh Config to pick up environment changes
            fresh_config = Config()
            provider_config = fresh_config.get_provider_config(provider)

            # Temporarily override deployment
            original_deployment = None
            if provider == "azure":
                original_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment

            try:
                # Create provider and service
                factory = ProviderFactory(fresh_config)
                provider_instance = factory.create_from_config(provider)
                service = InferenceService(provider_instance, repository=None)

                # Try a minimal completion call
                await service.run_inference(
                    "ping", temperature=0.0, max_tokens=1
                )

                return True

            finally:
                if provider == "azure":
                    if original_deployment:
                        os.environ["AZURE_OPENAI_DEPLOYMENT"] = original_deployment
                    elif "AZURE_OPENAI_DEPLOYMENT" in os.environ:
                        del os.environ["AZURE_OPENAI_DEPLOYMENT"]

        except Exception as e:
            logger.warning(
                f"Deployment validation failed: {deployment}, error: {str(e)}"
            )
            return False

    def _log_summarization_call(
        self,
        text: str,
        summary: Optional[str],
        deployment: str,
        provider: str,
        request: SummarizationRequest,
        raw_call_id: Optional[int] = None,
        error: Optional[Exception] = None,
        latency_ms: Optional[float] = None,
    ) -> Optional[int]:
        """
        Persist a summarization call to RawCall.

        Args:
            text: Original text that was summarized
            summary: Summarized text (None if failed)
            deployment: LLM deployment name
            provider: Provider name
            request: Original request
            raw_call_id: Optional existing RawCall ID (if already logged)
            error: Optional exception if call failed
            latency_ms: Optional latency in milliseconds

        Returns:
            RawCall ID if logged, None otherwise
        """
        if not self.repository:
            return None

        try:
            if error:
                # Log failure
                error_json = {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }

                metadata_json = {
                    "original_text": text[:500],  # Truncate for storage
                }
                if request.group_id:
                    metadata_json["group_id"] = request.group_id
                metadata_json.update(request.metadata)

                call_id = self.repository.insert_raw_call(
                    provider=f"{provider}_openai_{deployment}",
                    request_json={
                        "input": text,
                        "model": deployment,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                    },
                    model=deployment,
                    modality="text",
                    status="failed",
                    response_json=None,
                    error_json=error_json,
                    latency_ms=latency_ms,
                    tokens_json=None,
                    metadata_json=metadata_json,
                )

                logger.warning(
                    f"Logged failed summarization: id={call_id}, "
                    f"deployment={deployment}, error={str(error)}"
                )

                return call_id

            else:
                # Log success
                metadata_json = {
                    "original_text": text[:500],  # Truncate for storage
                }
                if request.group_id:
                    metadata_json["group_id"] = request.group_id
                metadata_json.update(request.metadata)

                call_id = self.repository.insert_raw_call(
                    provider=f"{provider}_openai_{deployment}",
                    request_json={
                        "input": text,
                        "model": deployment,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                    },
                    model=deployment,
                    modality="text",
                    status="success",
                    response_json={"text": summary},
                    error_json=None,
                    latency_ms=latency_ms,
                    tokens_json=None,
                    metadata_json=metadata_json,
                )

                logger.debug(
                    f"Logged successful summarization: id={call_id}, "
                    f"deployment={deployment}"
                )

                return call_id

        except Exception as db_error:
            if error is None and self.require_db_persistence:
                # Fail-fast for successful saves when persistence is required
                logger.error(
                    f"Failed to log successful summarization (require_db_persistence=True): {str(db_error)}",
                    exc_info=True
                )
                raise RuntimeError(
                    f"Database persistence failed. This is required for experimental data tracking. "
                    f"Original error: {str(db_error)}"
                ) from db_error
            else:
                # Graceful degradation for failures or when persistence is not required
                logger.warning(
                    f"Failed to log summarization call (require_db_persistence=False or failure case): {str(db_error)}",
                    exc_info=True
                )
                return None

    async def summarize_batch(
        self, request: SummarizationRequest
    ) -> SummarizationResponse:
        """
        Summarize a batch of texts using specified LLM deployment.

        Args:
            request: SummarizationRequest with texts, deployment, etc.

        Returns:
            SummarizationResponse with summaries and raw_call_ids

        Raises:
            ValueError: If deployment validation fails
        """
        # Validate deployment if requested
        if request.validate_deployment:
            is_valid = await self._validate_deployment(
                request.llm_deployment, request.provider
            )
            if not is_valid:
                error = ValueError(
                    f"Invalid deployment: {request.llm_deployment} "
                    f"(does not exist or does not support text completion)"
                )
                # Log all texts as failed
                raw_call_ids = []
                for text in request.texts:
                    call_id = self._log_summarization_call(
                        text=text,
                        summary=None,
                        deployment=request.llm_deployment,
                        provider=request.provider,
                        request=request,
                        error=error,
                    )
                    if call_id:
                        raw_call_ids.append(call_id)
                raise error

        # Create fresh Config and provider for this deployment
        fresh_config = Config()
        provider_config = fresh_config.get_provider_config(request.provider)

        # Temporarily override deployment
        original_deployment = None
        if request.provider == "azure":
            original_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            os.environ["AZURE_OPENAI_DEPLOYMENT"] = request.llm_deployment

        try:
            # Create provider and service
            factory = ProviderFactory(fresh_config)
            provider_instance = factory.create_from_config(request.provider)
            service = InferenceService(
                provider_instance, repository=None
            )  # We'll log manually

            # Process all texts concurrently
            async def summarize_one(text: str) -> tuple[str, int, Optional[Exception]]:
                """Summarize a single text and return (summary, call_id, error)."""
                import time

                start_time = time.time()
                call_id = None
                error = None
                summary = None

                try:
                    # Create prompt for summarization
                    prompt = (
                        "Write a single question that represents the ones in this list concisely:\n"
                        f"- {text}"
                    )

                    result = await service.run_inference(
                        prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    )

                    summary = result["response"].strip()
                    latency_ms = (time.time() - start_time) * 1000

                    # Log success
                    call_id = self._log_summarization_call(
                        text=text,
                        summary=summary,
                        deployment=request.llm_deployment,
                        provider=request.provider,
                        request=request,
                        latency_ms=latency_ms,
                    )

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    error = e

                    # Log failure
                    call_id = self._log_summarization_call(
                        text=text,
                        summary=None,
                        deployment=request.llm_deployment,
                        provider=request.provider,
                        request=request,
                        error=error,
                        latency_ms=latency_ms,
                    )

                    # Re-raise for batch processing (we'll collect errors)
                    raise

                return summary, call_id or 0, error

            # Process all texts concurrently
            tasks = [summarize_one(text) for text in request.texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect summaries and call IDs
            summaries = []
            raw_call_ids = []
            errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Error occurred
                    errors.append((i, result))
                    summaries.append("")  # Placeholder
                    raw_call_ids.append(0)  # Placeholder (already logged)
                else:
                    summary, call_id, error = result
                    summaries.append(summary)
                    raw_call_ids.append(call_id)
                    if error:
                        errors.append((i, error))

            # If all failed, raise an error
            if len(errors) == len(request.texts):
                error_msg = f"All {len(request.texts)} summarizations failed"
                raise RuntimeError(error_msg) from errors[0][1]

            # Log any partial failures
            if errors:
                logger.warning(
                    f"{len(errors)} out of {len(request.texts)} summarizations failed"
                )

            return SummarizationResponse(
                summaries=summaries,
                raw_call_ids=raw_call_ids,
                metadata={
                    "deployment": request.llm_deployment,
                    "provider": request.provider,
                    "total_texts": len(request.texts),
                    "successful": len(summaries) - len(errors),
                    "failed": len(errors),
                },
            )

        finally:
            # Restore original deployment
            if request.provider == "azure":
                if original_deployment:
                    os.environ["AZURE_OPENAI_DEPLOYMENT"] = original_deployment
                elif "AZURE_OPENAI_DEPLOYMENT" in os.environ:
                    del os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def create_paraphraser_for_llm(
        self, llm_deployment: str, provider: str = "azure"
    ) -> Optional[Callable[[List[str]], List[str]]]:
        """
        Factory function to create a paraphraser callable for a specific deployment.

        This creates a synchronous function that can be used as a paraphraser
        in algorithms (e.g., for cluster representative summarization).

        Args:
            llm_deployment: LLM deployment name
            provider: Provider name (default: "azure")

        Returns:
            Callable that takes a list of texts and returns a list of summaries,
            or None if deployment validation fails
        """
        # Validate deployment
        # Try to validate deployment if not in async context
        try:
            loop = asyncio.get_running_loop()
            # Loop is running, can't validate synchronously
            # Just create the paraphraser and let it fail on first use
            pass
        except RuntimeError:
            # No running loop, can validate
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _validate():
                return await self._validate_deployment(llm_deployment, provider)

            is_valid = loop.run_until_complete(_validate())
            if not is_valid:
                logger.warning(
                    f"Deployment validation failed: {llm_deployment}, "
                    f"paraphraser may not work"
                )
                return None

        def paraphrase_batch(texts: List[str]) -> List[str]:
            """Synchronous wrapper for batch summarization."""
            request = SummarizationRequest(
                texts=texts,
                llm_deployment=llm_deployment,
                provider=provider,
                validate_deployment=False,  # Already validated or will fail on use
            )

            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # If we get here, we're in an async context
                raise RuntimeError(
                    "Cannot run paraphraser in async context. "
                    "Use summarize_batch() directly instead."
                )
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.summarize_batch(request))
                    return result.summaries
                finally:
                    loop.close()

        return paraphrase_batch
