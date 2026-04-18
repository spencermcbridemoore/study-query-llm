"""Embeddings management view for the Panel application."""

import panel as pn

from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.utils.logging_config import get_logger

from panel_app.helpers import get_db_connection

logger = get_logger(__name__)


def _response_dimension(response_json) -> int | None:
    if not isinstance(response_json, dict):
        return None
    raw = response_json.get("embedding_dim")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def create_embeddings_ui() -> pn.viewable.Viewable:
    """Create the embeddings management UI."""

    refresh_button = pn.widgets.Button(
        name="Refresh",
        button_type="primary",
        width=100,
    )

    deployment_filter = pn.widgets.Select(
        name="Deployment",
        options=["All"],
        value="All",
        width=200,
    )

    status_filter = pn.widgets.Select(
        name="Status",
        options=["All", "success", "failed"],
        value="All",
        width=150,
    )

    summary_output = pn.pane.Str("", sizing_mode="stretch_width")
    embeddings_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=500)

    _cache: dict = {"calls": []}

    def _load_from_db():
        """Fetch embedding raw_call rows from DB into _cache."""
        try:
            db = get_db_connection()
            with db.session_scope() as session:
                repository = RawCallRepository(session)
                all_calls = repository.query_raw_calls(modality="embedding", limit=500)
                calls_data = []
                for call in all_calls:
                    calls_data.append(
                        {
                            "id": call.id,
                            "model": call.model,
                            "provider": call.provider,
                            "status": call.status,
                            "latency_ms": call.latency_ms,
                            "request_json": call.request_json,
                            "response_json": call.response_json,
                            "metadata_json": call.metadata_json,
                            "created_at": call.created_at,
                        }
                    )
            _cache["calls"] = calls_data

            deployments = sorted({row["model"] for row in calls_data if row["model"]})
            deployment_options = ["All"] + deployments
            if deployment_filter.options != deployment_options:
                deployment_filter.options = deployment_options

        except Exception as exc:
            logger.error("Failed to load embeddings from DB: %s", exc, exc_info=True)
            _cache["calls"] = []

    def _refresh_display():
        """Apply current filter values to cached rows and update UI widgets."""
        import pandas as pd

        try:
            calls = list(_cache["calls"])

            if status_filter.value != "All":
                calls = [row for row in calls if row["status"] == status_filter.value]
            if deployment_filter.value != "All":
                calls = [row for row in calls if row["model"] == deployment_filter.value]

            total_embeddings = len(calls)
            successful = len([row for row in calls if row["status"] == "success"])
            failed = len([row for row in calls if row["status"] == "failed"])

            cache_hits = 0
            cache_misses = 0
            total_latency = 0.0
            latency_count = 0
            dimensions: set[int] = set()

            for row in calls:
                if row["latency_ms"]:
                    total_latency += float(row["latency_ms"])
                    latency_count += 1
                dim = _response_dimension(row["response_json"])
                if dim is not None:
                    dimensions.add(dim)
                meta = row["metadata_json"]
                if isinstance(meta, dict):
                    if meta.get("cached"):
                        cache_hits += 1
                    else:
                        cache_misses += 1

            avg_latency = total_latency / latency_count if latency_count > 0 else 0.0
            avg_dimension = sum(dimensions) / len(dimensions) if dimensions else 0.0
            cache_hit_rate = (
                cache_hits / (cache_hits + cache_misses) * 100
                if (cache_hits + cache_misses) > 0
                else 0.0
            )
            unique_deployments = len({row["model"] for row in calls if row["model"]})

            summary_output.object = f"""
**Total Embeddings:** {total_embeddings}  
**Successful:** {successful}  
**Failed:** {failed}  
**Cache Hit Rate:** {cache_hit_rate:.1f}% ({cache_hits} hits, {cache_misses} misses)  
**Average Latency:** {avg_latency:.2f} ms  
**Average Dimension:** {avg_dimension:.0f}  
**Unique Deployments:** {unique_deployments}  
**Unique Dimensions:** {len(dimensions)} ({', '.join(map(str, sorted(dimensions))) if dimensions else 'N/A'})
"""

            if calls:
                rows = []
                for row in calls[:100]:
                    request = row["request_json"] if isinstance(row["request_json"], dict) else {}
                    input_text = (
                        request.get("input", "")
                        or request.get("text", "")
                        or ""
                    )
                    meta = row["metadata_json"] if isinstance(row["metadata_json"], dict) else {}
                    dim = _response_dimension(row["response_json"])

                    rows.append(
                        {
                            "id": row["id"],
                            "deployment": row["model"] or "N/A",
                            "provider": row["provider"],
                            "input": (
                                input_text[:50] + "..."
                                if len(str(input_text)) > 50
                                else input_text
                            ),
                            "dimension": dim if dim is not None else "N/A",
                            "cached": "Yes" if bool(meta.get("cached")) else "No",
                            "status": row["status"],
                            "latency_ms": (
                                f"{row['latency_ms']:.2f}"
                                if row["latency_ms"] is not None
                                else "N/A"
                            ),
                            "created_at": (
                                row["created_at"].isoformat()
                                if row["created_at"] is not None
                                else ""
                            ),
                        }
                    )
                embeddings_table.object = pd.DataFrame(rows)
            else:
                embeddings_table.object = None

        except Exception as exc:
            logger.error("Failed to refresh embeddings display: %s", exc, exc_info=True)
            summary_output.object = f"Error loading embeddings: {exc}"
            embeddings_table.object = None

    def update_embeddings():
        _load_from_db()
        _refresh_display()

    def on_filter_change(_event):
        _refresh_display()

    deployment_filter.param.watch(on_filter_change, "value")
    status_filter.param.watch(on_filter_change, "value")
    refresh_button.on_click(lambda _event: update_embeddings())

    update_embeddings()

    return pn.Column(
        pn.pane.Markdown("## Embeddings Management"),
        pn.Row(refresh_button, deployment_filter, status_filter, pn.Spacer()),
        pn.pane.Markdown("### Summary Statistics"),
        summary_output,
        pn.pane.Markdown("### Embedding Calls"),
        embeddings_table,
        sizing_mode="stretch_width",
        margin=(10, 20),
    )
