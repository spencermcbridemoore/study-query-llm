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

    _cache: dict = {"calls": [], "vectors": {}}

    def _load_from_db():
        """Fetch all embedding data from DB into _cache (plain dicts survive session close)."""
        try:
            db = get_db_connection()
            with db.session_scope() as session:
                from study_query_llm.db.models_v2 import EmbeddingVector
                repository = RawCallRepository(session)

                all_calls = repository.query_raw_calls(modality='embedding', limit=500)

                calls_data = []
                for c in all_calls:
                    calls_data.append({
                        'id': c.id, 'model': c.model, 'provider': c.provider,
                        'status': c.status, 'latency_ms': c.latency_ms,
                        'request_json': c.request_json,
                        'metadata_json': c.metadata_json,
                        'created_at': c.created_at,
                    })

                success_ids = [cd['id'] for cd in calls_data if cd['status'] == 'success']
                vectors_data: dict = {}
                if success_ids:
                    vectors = session.query(EmbeddingVector).filter(
                        EmbeddingVector.call_id.in_(success_ids)
                    ).all()
                    vectors_data = {
                        v.call_id: {
                            'dimension': v.dimension, 'norm': v.norm,
                            'has_vector': bool(v.vector),
                        }
                        for v in vectors
                    }

            _cache['calls'] = calls_data
            _cache['vectors'] = vectors_data

            deployments = sorted({cd['model'] for cd in calls_data if cd['model']})
            deployment_options = ["All"] + deployments
            if deployment_filter.options != deployment_options:
                deployment_filter.options = deployment_options

        except Exception as e:
            logger.error(f"Failed to load embeddings from DB: {str(e)}", exc_info=True)
            _cache['calls'] = []
            _cache['vectors'] = {}

    def _refresh_display():
        """Apply current filter values to cached data and update display widgets."""
        import pandas as pd
        try:
            calls = list(_cache['calls'])
            vector_by_call_id = _cache['vectors']

            if status_filter.value != "All":
                calls = [c for c in calls if c['status'] == status_filter.value]
            if deployment_filter.value != "All":
                calls = [c for c in calls if c['model'] == deployment_filter.value]

            total_embeddings = len(calls)
            successful = len([c for c in calls if c['status'] == 'success'])
            failed = len([c for c in calls if c['status'] == 'failed'])

            cache_hits = 0
            cache_misses = 0
            total_latency = 0
            latency_count = 0

            for cd in calls:
                if cd['latency_ms']:
                    total_latency += cd['latency_ms']
                    latency_count += 1
                meta = cd['metadata_json']
                if meta and isinstance(meta, dict):
                    if meta.get('cached'):
                        cache_hits += 1
                    else:
                        cache_misses += 1

            success_ids = {c['id'] for c in calls if c['status'] == 'success'}
            filtered_vectors = {
                cid: v for cid, v in vector_by_call_id.items()
                if cid in success_ids
            }
            dimensions = {v['dimension'] for v in filtered_vectors.values()}
            avg_dimension = sum(dimensions) / len(dimensions) if dimensions else 0

            avg_latency = total_latency / latency_count if latency_count > 0 else 0
            cache_hit_rate = (
                cache_hits / (cache_hits + cache_misses) * 100
                if (cache_hits + cache_misses) > 0 else 0
            )
            unique_deployments = len({c['model'] for c in calls if c['model']})

            summary_text = f"""
**Total Embeddings:** {total_embeddings}  
**Successful:** {successful}  
**Failed:** {failed}  
**Cache Hit Rate:** {cache_hit_rate:.1f}% ({cache_hits} hits, {cache_misses} misses)  
**Average Latency:** {avg_latency:.2f} ms  
**Average Dimension:** {avg_dimension:.0f}  
**Unique Deployments:** {unique_deployments}  
**Unique Dimensions:** {len(dimensions)} ({', '.join(map(str, sorted(dimensions))) if dimensions else 'N/A'})
"""
            summary_output.object = summary_text

            if calls:
                embeddings_data = []
                for cd in calls[:100]:
                    request = cd['request_json'] or {}
                    input_text = ''
                    if isinstance(request, dict):
                        input_text = request.get('input', '') or request.get('text', '') or ''

                    vec = vector_by_call_id.get(cd['id'])

                    cached = False
                    meta = cd['metadata_json']
                    if meta and isinstance(meta, dict):
                        cached = meta.get('cached', False)

                    embeddings_data.append({
                        'id': cd['id'],
                        'deployment': cd['model'] or 'N/A',
                        'provider': cd['provider'],
                        'input': input_text[:50] + '...' if len(input_text) > 50 else input_text,
                        'dimension': vec['dimension'] if vec else 'N/A',
                        'norm': f"{vec['norm']:.4f}" if vec and vec['norm'] else 'N/A',
                        'cached': 'Yes' if cached else 'No',
                        'status': cd['status'],
                        'latency_ms': f"{cd['latency_ms']:.2f}" if cd['latency_ms'] else 'N/A',
                        'created_at': cd['created_at'].isoformat() if cd['created_at'] else '',
                    })
                embeddings_table.object = pd.DataFrame(embeddings_data)
            else:
                embeddings_table.object = None

            if filtered_vectors:
                call_by_id = {c['id']: c for c in calls}
                vectors_rows = []
                for call_id, vec in list(filtered_vectors.items())[:50]:
                    cd = call_by_id.get(call_id)
                    vectors_rows.append({
                        'call_id': call_id,
                        'deployment': cd['model'] if cd else 'N/A',
                        'dimension': vec['dimension'],
                        'norm': f"{vec['norm']:.4f}" if vec['norm'] else 'N/A',
                        'has_vector': 'Yes' if vec['has_vector'] else 'No',
                    })
                embedding_vectors_table.object = pd.DataFrame(vectors_rows)
            else:
                embedding_vectors_table.object = None

        except Exception as e:
            logger.error(f"Failed to refresh embeddings display: {str(e)}", exc_info=True)
            summary_output.object = f"Error loading embeddings: {str(e)}"
            embeddings_table.object = None
            embedding_vectors_table.object = None

    def update_embeddings():
        """Full refresh: reload from DB then update display."""
        _load_from_db()
        _refresh_display()

    def on_filter_change(event):
        _refresh_display()

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
