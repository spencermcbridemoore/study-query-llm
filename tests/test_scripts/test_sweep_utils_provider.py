"""Tests for the provider parameter in create_paraphraser_for_llm."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from study_query_llm.db.connection_v2 import DatabaseConnectionV2


@pytest.fixture
def db():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_paraphraser_none_returns_none(db):
    """Passing None for llm_deployment returns None (backward compat)."""
    from scripts.common.sweep_utils import create_paraphraser_for_llm

    result = create_paraphraser_for_llm(None, db)
    assert result is None


def test_create_paraphraser_none_returns_none_with_provider(db):
    """None deployment returns None regardless of provider."""
    from scripts.common.sweep_utils import create_paraphraser_for_llm

    result = create_paraphraser_for_llm(None, db, provider="local_llm")
    assert result is None


def test_create_paraphraser_returns_callable(db):
    """Non-None deployment returns a callable paraphraser."""
    from scripts.common.sweep_utils import create_paraphraser_for_llm

    result = create_paraphraser_for_llm("llama3.1:8b", db, provider="local_llm")
    assert result is not None
    assert callable(result)


def test_create_paraphraser_provider_flows_to_request(db):
    """The provider kwarg is forwarded into the SummarizationRequest."""
    from scripts.common.sweep_utils import create_paraphraser_for_llm

    with patch(
        "scripts.common.sweep_utils.SummarizationService"
    ) as MockSvc:
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.summaries = ["summary"]

        async def _fake_summarize(req):
            assert req.provider == "local_llm"
            assert req.llm_deployment == "llama3.1:8b"
            return mock_result

        mock_service.summarize_batch = AsyncMock(side_effect=_fake_summarize)
        MockSvc.return_value = mock_service

        paraphraser = create_paraphraser_for_llm(
            "llama3.1:8b", db, provider="local_llm"
        )
        assert paraphraser is not None
