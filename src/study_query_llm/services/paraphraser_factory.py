"""Factory for creating synchronous paraphraser callables from LLM services."""

import asyncio
from typing import Callable, List, Optional

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.summarization_service import (
    SummarizationService,
    SummarizationRequest,
)


def create_paraphraser_for_llm(
    llm_deployment: Optional[str],
    db: DatabaseConnectionV2,
    *,
    provider: str = "azure",
    combine_texts: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 256,
    timeout: float = 300.0,
) -> Optional[Callable[[List[str]], str]]:
    """Create a synchronous paraphraser callable for use inside ``run_sweep``.

    Args:
        llm_deployment: Model/deployment name (``None`` -> return ``None``).
        db: Database connection for the summarization service.
        provider: Provider name, e.g. ``"azure"``, ``"local_llm"``,
            ``"ollama"``.  Defaults to ``"azure"`` for backward compatibility.
        combine_texts: When ``True`` (default), concatenate all input texts
            into a single prompt and return one summary string.  When
            ``False``, pass texts directly to ``SummarizationService`` and
            return ``result.summaries`` (a list).
        temperature: Sampling temperature for the LLM call.
        max_tokens: Maximum tokens for the response.
        timeout: Per-call wall-clock timeout in seconds.
    """
    if llm_deployment is None:
        return None

    async def _paraphrase_async(texts: List[str]):
        async def _summarize():
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                service = SummarizationService(repository=repo)

                if combine_texts:
                    combined = ""
                    for i, text in enumerate(texts, 1):
                        combined += f"Text {i}:\n{text}\n\n"
                    req_texts = [combined.strip()]
                else:
                    req_texts = texts

                request = SummarizationRequest(
                    texts=req_texts,
                    llm_deployment=llm_deployment,
                    provider=provider,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                try:
                    result = await service.summarize_batch(request)
                    if combine_texts:
                        return (
                            result.summaries[0]
                            if result.summaries
                            else (texts[0] if texts else "")
                        )
                    return result.summaries
                except Exception as exc:
                    msg = str(exc)
                    if "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg:
                        print("          [WARN] Content filtered, using original text")
                    else:
                        print(f"          [WARN] Summarization error: {msg[:100]}")
                    return texts[0] if texts else ""

        try:
            return await asyncio.wait_for(_summarize(), timeout=timeout)
        except asyncio.TimeoutError:
            print("      [WARN] Summarization timed out, using first text")
            return texts[0] if texts else ""

    def paraphrase_sync(texts: List[str]):
        """Synchronous wrapper suitable for ``run_sweep``."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
            asyncio.set_event_loop(None)
        except RuntimeError:
            pass
        return asyncio.run(_paraphrase_async(texts))

    return paraphrase_sync
