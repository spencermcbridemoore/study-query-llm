"""Analytics dashboard view for the Panel application."""

import panel as pn

from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.study_service import StudyService
from study_query_llm.utils.logging_config import get_logger

from panel_app.helpers import get_db_connection

logger = get_logger(__name__)


def create_analytics_ui() -> pn.viewable.Viewable:
    """Create the analytics dashboard UI."""

    refresh_button = pn.widgets.Button(
        name="Refresh",
        button_type="primary",
        width=100
    )

    summary_output = pn.pane.Str("", sizing_mode='stretch_width')
    provider_comparison_table = pn.pane.DataFrame(None, sizing_mode='stretch_width', height=200)
    recent_table = pn.pane.DataFrame(None, sizing_mode='stretch_width', height=400)

    def update_analytics():
        """Update analytics display."""
        try:
            logger.debug("Updating analytics display...")
            db = get_db_connection()
            with db.session_scope() as session:
                repository = RawCallRepository(session)
                study_service = StudyService(repository)

                stats = study_service.get_summary_stats()
                logger.debug(f"Analytics stats: {stats}")
                summary_text = f"""
**Total Inferences:** {stats['total_inferences']}  
**Total Tokens:** {stats['total_tokens']:,}  
**Unique Providers:** {stats['unique_providers']}
"""
                summary_output.object = summary_text

                comparison_df = study_service.get_provider_comparison()
                if not comparison_df.empty:
                    display_cols = ['provider', 'count', 'avg_tokens', 'avg_latency_ms', 'total_tokens']
                    if 'avg_cost_estimate' in comparison_df.columns:
                        display_cols.append('avg_cost_estimate')
                    provider_comparison_table.object = comparison_df[display_cols]
                else:
                    provider_comparison_table.object = None

                recent_df = study_service.get_recent_inferences(limit=20)
                if not recent_df.empty:
                    display_df = recent_df[['id', 'prompt', 'provider', 'tokens', 'latency_ms', 'created_at']].copy()
                    display_df['prompt'] = display_df['prompt'].apply(lambda x: x[:50] + '...' if len(str(x)) > 50 else x)
                    recent_table.object = display_df
                else:
                    recent_table.object = None

        except Exception as e:
            error_msg = f"Error loading analytics: {str(e)}"
            logger.error(f"Failed to update analytics: {str(e)}", exc_info=True)
            summary_output.object = error_msg
            provider_comparison_table.object = None
            recent_table.object = None

    refresh_button.on_click(lambda e: update_analytics())

    update_analytics()

    return pn.Column(
        pn.pane.Markdown("## Analytics Dashboard"),
        pn.Row(refresh_button, pn.Spacer()),
        pn.pane.Markdown("### Summary Statistics"),
        summary_output,
        pn.pane.Markdown("### Provider Comparison"),
        provider_comparison_table,
        pn.pane.Markdown("### Recent Inferences"),
        recent_table,
        sizing_mode='stretch_width',
        margin=(10, 20)
    )
