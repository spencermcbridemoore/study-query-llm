"""Groups management view for the Panel application."""

import panel as pn

from study_query_llm.utils.logging_config import get_logger

from panel_app.helpers import get_db_connection

logger = get_logger(__name__)


def create_groups_ui() -> pn.viewable.Viewable:
    """Create the groups management UI."""

    refresh_button = pn.widgets.Button(
        name="Refresh",
        button_type="primary",
        width=100
    )

    groups_table = pn.pane.DataFrame(None, sizing_mode='stretch_width', height=400)
    group_members_table = pn.pane.DataFrame(None, sizing_mode='stretch_width', height=400)
    groups_notice = pn.pane.Markdown("", sizing_mode="stretch_width")

    def update_groups():
        """Update groups display."""
        groups_notice.object = ""
        try:
            db = get_db_connection()
            with db.session_scope() as session:
                from sqlalchemy import func
                from study_query_llm.db.models_v2 import Group, GroupMember, RawCall
                import pandas as pd

                groups = session.query(Group).order_by(Group.created_at.desc()).limit(50).all()

                if groups:
                    group_ids = [g.id for g in groups]
                    member_counts = dict(
                        session.query(GroupMember.group_id, func.count(GroupMember.id))
                        .filter(GroupMember.group_id.in_(group_ids))
                        .group_by(GroupMember.group_id)
                        .all()
                    )

                    groups_data = []
                    for group in groups:
                        groups_data.append({
                            'id': group.id,
                            'group_type': group.group_type,
                            'name': group.name,
                            'description': group.description or '',
                            'member_count': member_counts.get(group.id, 0),
                            'created_at': group.created_at.isoformat() if group.created_at else '',
                        })

                    groups_table.object = pd.DataFrame(groups_data)
                else:
                    groups_table.object = None
                    groups_notice.object = "_No `Group` rows returned (schema is empty or limit 50)._"

                members = session.query(GroupMember).order_by(GroupMember.added_at.desc()).limit(100).all()

                if members:
                    related_group_ids = {m.group_id for m in members}
                    related_call_ids = {m.call_id for m in members}
                    groups_map = {
                        g.id: g for g in
                        session.query(Group).filter(Group.id.in_(related_group_ids)).all()
                    }
                    calls_map = {
                        c.id: c for c in
                        session.query(RawCall).filter(RawCall.id.in_(related_call_ids)).all()
                    }

                    members_data = []
                    for member in members:
                        group = groups_map.get(member.group_id)
                        call = calls_map.get(member.call_id)
                        members_data.append({
                            'group_id': member.group_id,
                            'group_name': group.name if group else 'N/A',
                            'call_id': member.call_id,
                            'role': member.role or '',
                            'position': member.position or '',
                            'provider': call.provider if call else 'N/A',
                            'status': call.status if call else 'N/A',
                            'added_at': member.added_at.isoformat() if member.added_at else '',
                        })

                    group_members_table.object = pd.DataFrame(members_data)
                else:
                    group_members_table.object = None
                    if groups:
                        groups_notice.object = "_No group membership rows in the last 100._"

        except Exception as e:
            logger.error(f"Failed to update groups: {str(e)}", exc_info=True)
            groups_table.object = None
            group_members_table.object = None
            groups_notice.object = f"**Error loading groups:** `{e!s}`"

    refresh_button.on_click(lambda e: update_groups())

    update_groups()

    return pn.Column(
        pn.pane.Markdown("## Groups Management"),
        pn.Row(refresh_button, pn.Spacer()),
        groups_notice,
        pn.pane.Markdown("### Groups"),
        groups_table,
        pn.pane.Markdown("### Group Members"),
        group_members_table,
        sizing_mode='stretch_width',
        margin=(10, 20)
    )
