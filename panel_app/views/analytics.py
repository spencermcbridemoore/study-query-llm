"""Analytics dashboard view for the Panel application."""

from __future__ import annotations

import panel as pn
import pandas as pd
from typing import Optional
from sqlalchemy import desc, func

from study_query_llm.db.models_v2 import (
    CallArtifact,
    EmbeddingVector,
    Group,
    GroupMember,
    ProvenancedRun,
    RawCall,
)
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN
from study_query_llm.services.study_service import StudyService
from study_query_llm.utils.logging_config import get_logger

from panel_app.helpers import get_db_connection

logger = get_logger(__name__)


def _v2_activity_markdown(session) -> str:
    n_groups = session.query(Group).count()
    n_raw = session.query(RawCall).count()
    n_provenanced = session.query(ProvenancedRun).count()
    n_mcq_groups = (
        session.query(Group).filter(Group.group_type == GROUP_TYPE_MCQ_RUN).count()
    )
    n_artifacts = session.query(CallArtifact).count()
    n_members = session.query(GroupMember).count()
    n_embeddings = session.query(EmbeddingVector).count()

    lines = [
        "**v2 table counts** (same semantics as the sidebar):",
        "",
        f"| Table / slice | Count |",
        f"| --- | ---: |",
        f"| `groups` | {n_groups:,} |",
        f"| `raw_calls` | {n_raw:,} |",
        f"| `provenanced_runs` | {n_provenanced:,} |",
        f"| `groups` where `group_type = mcq_run` | {n_mcq_groups:,} |",
        f"| `call_artifacts` | {n_artifacts:,} |",
        f"| `group_members` | {n_members:,} |",
        f"| `embedding_vectors` | {n_embeddings:,} |",
        "",
        "_**Summary statistics** and **provider comparison** below are computed from **`raw_calls`** only. "
        "MCQ sweep work and many workers write **`provenanced_runs`** / **`groups`** without one row per "
        "question in `raw_calls`, so use the tables above when `raw_calls` looks sparse._",
    ]
    return "\n".join(lines)


def _dataframe_or_none(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    return None if df is None or df.empty else df


def create_analytics_ui() -> pn.viewable.Viewable:
    """Create the analytics dashboard UI."""

    refresh_button = pn.widgets.Button(
        name="Refresh",
        button_type="primary",
        width=100,
    )

    v2_intro = pn.pane.Markdown("", sizing_mode="stretch_width")
    summary_output = pn.pane.Markdown("", sizing_mode="stretch_width")
    modality_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=160)
    status_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=140)
    groups_by_type_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=220)
    provenanced_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=220)
    recent_groups_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=220)
    provider_comparison_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=200)
    recent_table = pn.pane.DataFrame(None, sizing_mode="stretch_width", height=400)

    def update_analytics():
        """Update analytics display."""
        try:
            logger.debug("Updating analytics display...")
            db = get_db_connection()
            with db.session_scope() as session:
                repository = RawCallRepository(session)
                study_service = StudyService(repository)

                v2_intro.object = _v2_activity_markdown(session)

                modality_rows = (
                    session.query(RawCall.modality, func.count(RawCall.id))
                    .group_by(RawCall.modality)
                    .order_by(func.count(RawCall.id).desc())
                    .all()
                )
                modality_df = pd.DataFrame(
                    modality_rows, columns=["modality", "count"]
                )
                modality_table.object = _dataframe_or_none(modality_df)

                status_rows = (
                    session.query(RawCall.status, func.count(RawCall.id))
                    .group_by(RawCall.status)
                    .order_by(func.count(RawCall.id).desc())
                    .all()
                )
                status_df = pd.DataFrame(status_rows, columns=["status", "count"])
                status_table.object = _dataframe_or_none(status_df)

                gt_rows = (
                    session.query(Group.group_type, func.count(Group.id))
                    .group_by(Group.group_type)
                    .order_by(func.count(Group.id).desc())
                    .all()
                )
                groups_by_type_df = pd.DataFrame(
                    gt_rows, columns=["group_type", "count"]
                )
                groups_by_type_table.object = _dataframe_or_none(groups_by_type_df)

                pr_rows = (
                    session.query(ProvenancedRun)
                    .order_by(desc(ProvenancedRun.created_at))
                    .limit(20)
                    .all()
                )
                if pr_rows:
                    pr_data = []
                    for r in pr_rows:
                        meta = r.metadata_json if isinstance(r.metadata_json, dict) else {}
                        role = meta.get("execution_role", "")
                        pr_data.append(
                            {
                                "id": r.id,
                                "run_kind": r.run_kind,
                                "run_status": r.run_status,
                                "run_key": (r.run_key or "")[:60],
                                "execution_role": role,
                                "created_at": r.created_at,
                            }
                        )
                    provenanced_table.object = pd.DataFrame(pr_data)
                else:
                    provenanced_table.object = None

                rg = (
                    session.query(Group)
                    .order_by(desc(Group.created_at))
                    .limit(20)
                    .all()
                )
                if rg:
                    recent_groups_table.object = pd.DataFrame(
                        [
                            {
                                "id": g.id,
                                "group_type": g.group_type,
                                "name": (g.name or "")[:80],
                                "created_at": g.created_at,
                            }
                            for g in rg
                        ]
                    )
                else:
                    recent_groups_table.object = None

                stats = study_service.get_summary_stats()
                logger.debug("Analytics stats: %s", stats)
                summary_text = (
                    f"**Total raw_calls rows:** {stats['total_inferences']:,}  \n"
                    f"**Total tokens (parsed from tokens_json):** {stats['total_tokens']:,}  \n"
                    f"**Distinct providers in raw_calls:** {stats['unique_providers']:,}  \n\n"
                    "_Token and latency averages are only as good as populated `tokens_json` / `latency_ms`._"
                )
                summary_output.object = summary_text

                comparison_df = study_service.get_provider_comparison()
                if not comparison_df.empty:
                    display_cols = [
                        "provider",
                        "count",
                        "avg_tokens",
                        "avg_latency_ms",
                        "total_tokens",
                    ]
                    if "avg_cost_estimate" in comparison_df.columns:
                        display_cols.append("avg_cost_estimate")
                    provider_comparison_table.object = comparison_df[display_cols]
                else:
                    provider_comparison_table.object = None

                recent_df = study_service.get_recent_inferences(
                    limit=20, modality=None, status=None
                )
                if not recent_df.empty:
                    display_cols = [
                        "id",
                        "modality",
                        "status",
                        "prompt",
                        "provider",
                        "tokens",
                        "latency_ms",
                        "created_at",
                    ]
                    display_df = recent_df[display_cols].copy()
                    display_df["prompt"] = display_df["prompt"].apply(
                        lambda x: x[:50] + "..." if len(str(x)) > 50 else x
                    )
                    recent_table.object = display_df
                else:
                    recent_table.object = None

        except Exception as e:
            error_msg = f"**Error loading analytics:** `{e!s}`"
            logger.error("Failed to update analytics: %s", str(e), exc_info=True)
            v2_intro.object = ""
            summary_output.object = error_msg
            modality_table.object = None
            status_table.object = None
            groups_by_type_table.object = None
            provenanced_table.object = None
            recent_groups_table.object = None
            provider_comparison_table.object = None
            recent_table.object = None

    refresh_button.on_click(lambda e: update_analytics())

    update_analytics()

    return pn.Column(
        pn.pane.Markdown("## Analytics Dashboard"),
        pn.Row(refresh_button, pn.Spacer()),
        pn.pane.Markdown("### v2 activity (beyond raw_calls)"),
        v2_intro,
        pn.pane.Markdown("### Summary statistics (raw_calls)"),
        summary_output,
        pn.pane.Markdown("### Provider comparison (raw_calls)"),
        provider_comparison_table,
        pn.pane.Markdown("### raw_calls by modality"),
        modality_table,
        pn.pane.Markdown("### raw_calls by status"),
        status_table,
        pn.pane.Markdown("### Groups by type"),
        groups_by_type_table,
        pn.pane.Markdown("### Recent provenanced runs"),
        provenanced_table,
        pn.pane.Markdown("### Recent groups"),
        recent_groups_table,
        pn.pane.Markdown("### Recent raw_calls (all modalities and statuses)"),
        recent_table,
        sizing_mode="stretch_width",
        margin=(10, 20),
    )
