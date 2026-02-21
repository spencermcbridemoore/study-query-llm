"""Embeddings management view for the Panel application."""

import panel as pn

from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.utils.logging_config import get_logger

from panel_app.helpers import get_db_connection

logger = get_logger(__name__)


def create_embeddings_ui() -> pn.viewable.Viewable:
    """Create the embeddings management UI."""

    refresh_button = pn.widgets.Button(
        name="Refresh",
        button_type="primary",
        width=100
    )

    deployment_filter = pn.widgets.Select(
        name="Deployment",
        options=["All"],
        value="All",
        width=200
    )

    status_filter = pn.widgets.Select(
        name="Status",
        options=["All", "success", "failed"],
        value="All",
        width=150
    )

    summary_output = pn.pane.Str("", sizing_mode='stretch_width')
    embeddings_table = pn.pane.DataFrame(None, sizing_mode='stretch_width', height=400)
    embedding_vectors_table = pn.pane.DataFrame(None, sizing_mode='stretch_width', height=300)

    def update_embeddings():
        """Update embeddings display."""
        try:
            db = get_db_connection()
            with db.session_scope() as session:
                from study_query_llm.db.models_v2 import RawCall, EmbeddingVector
                import pandas as pd
                repository = RawCallRepository(session)

                filters = {'modality': 'embedding'}
                if status_filter.value != "All":
                    filters['status'] = status_filter.value

                embedding_calls = repository.query_raw_calls(**filters, limit=500)

                deployments = set()
                for call in embedding_calls:
                    if call.model:
                        deployments.add(call.model)
                deployment_options = ["All"] + sorted(list(deployments))
                if deployment_filter.options != deployment_options:
                    deployment_filter.options = deployment_options

                if deployment_filter.value != "All":
                    embedding_calls = [c for c in embedding_calls if c.model == deployment_filter.value]

                total_embeddings = len(embedding_calls)
                successful = len([c for c in embedding_calls if c.status == 'success'])
                failed = len([c for c in embedding_calls if c.status == 'failed'])

                cache_hits = 0
                cache_misses = 0
                total_latency = 0
                latency_count = 0
                dimensions = set()

                for call in embedding_calls:
                    if call.latency_ms:
                        total_latency += call.latency_ms
                        latency_count += 1

                    if call.metadata_json and isinstance(call.metadata_json, dict):
                        if call.metadata_json.get('cached'):
                            cache_hits += 1
                        else:
                            cache_misses += 1

                call_ids = [c.id for c in embedding_calls if c.status == 'success']
                if call_ids:
                    vectors = session.query(EmbeddingVector).filter(
                        EmbeddingVector.call_id.in_(call_ids)
                    ).all()
                    dimensions = {v.dimension for v in vectors}
                    avg_dimension = sum(dimensions) / len(dimensions) if dimensions else 0
                else:
                    avg_dimension = 0

                avg_latency = total_latency / latency_count if latency_count > 0 else 0
                cache_hit_rate = cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0

                summary_text = f"""
**Total Embeddings:** {total_embeddings}  
**Successful:** {successful}  
**Failed:** {failed}  
**Cache Hit Rate:** {cache_hit_rate:.1f}% ({cache_hits} hits, {cache_misses} misses)  
**Average Latency:** {avg_latency:.2f} ms  
**Average Dimension:** {avg_dimension:.0f}  
**Unique Deployments:** {len(deployments)}  
**Unique Dimensions:** {len(dimensions)} ({', '.join(map(str, sorted(dimensions))) if dimensions else 'N/A'})
"""
                summary_output.object = summary_text

                if embedding_calls:
                    embeddings_data = []
                    for call in embedding_calls[:100]:
                        request = call.request_json or {}
                        input_text = ''
                        if isinstance(request, dict):
                            input_text = request.get('input', '') or request.get('text', '') or ''

                        vector = session.query(EmbeddingVector).filter_by(call_id=call.id).first()

                        cached = False
                        if call.metadata_json and isinstance(call.metadata_json, dict):
                            cached = call.metadata_json.get('cached', False)

                        embeddings_data.append({
                            'id': call.id,
                            'deployment': call.model or 'N/A',
                            'provider': call.provider,
                            'input': input_text[:50] + '...' if len(input_text) > 50 else input_text,
                            'dimension': vector.dimension if vector else 'N/A',
                            'norm': f"{vector.norm:.4f}" if vector and vector.norm else 'N/A',
                            'cached': 'Yes' if cached else 'No',
                            'status': call.status,
                            'latency_ms': f"{call.latency_ms:.2f}" if call.latency_ms else 'N/A',
                            'created_at': call.created_at.isoformat() if call.created_at else '',
                        })

                    embeddings_df = pd.DataFrame(embeddings_data)
                    embeddings_table.object = embeddings_df
                else:
                    embeddings_table.object = None

                if call_ids:
                    vectors = session.query(EmbeddingVector).filter(
                        EmbeddingVector.call_id.in_(call_ids[:50])
                    ).all()

                    if vectors:
                        vectors_data = []
                        for vector in vectors:
                            call = session.query(RawCall).filter_by(id=vector.call_id).first()
                            vectors_data.append({
                                'call_id': vector.call_id,
                                'deployment': call.model if call else 'N/A',
                                'dimension': vector.dimension,
                                'norm': f"{vector.norm:.4f}" if vector.norm else 'N/A',
                                'has_vector': 'Yes' if vector.vector else 'No',
                            })

                        vectors_df = pd.DataFrame(vectors_data)
                        embedding_vectors_table.object = vectors_df
                    else:
                        embedding_vectors_table.object = None
                else:
                    embedding_vectors_table.object = None

        except Exception as e:
            error_msg = f"Error loading embeddings: {str(e)}"
            logger.error(f"Failed to update embeddings: {str(e)}", exc_info=True)
            summary_output.object = error_msg
            embeddings_table.object = None
            embedding_vectors_table.object = None

    def on_filter_change(event):
        update_embeddings()

    deployment_filter.param.watch(on_filter_change, 'value')
    status_filter.param.watch(on_filter_change, 'value')
    refresh_button.on_click(lambda e: update_embeddings())

    update_embeddings()

    return pn.Column(
        pn.pane.Markdown("## Embeddings Management"),
        pn.Row(
            refresh_button,
            deployment_filter,
            status_filter,
            pn.Spacer()
        ),
        pn.pane.Markdown("### Summary Statistics"),
        summary_output,
        pn.pane.Markdown("### Embedding Calls"),
        embeddings_table,
        pn.pane.Markdown("### Embedding Vectors"),
        embedding_vectors_table,
        sizing_mode='stretch_width',
        margin=(10, 20)
    )
