"""Panel smoke tests for view imports."""

from __future__ import annotations


def test_panel_views_import_without_embedding_vector() -> None:
    from panel_app.views import analytics, embeddings

    assert callable(analytics.create_analytics_ui)
    assert callable(embeddings.create_embeddings_ui)
