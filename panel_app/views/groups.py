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

    def update_groups():
        """Update groups display."""
        try:
            db = get_db_connection()
            with db.session_scope() as session:
                from study_query_llm.db.models_v2 import Group, GroupMember, RawCall
                import pandas as pd

                groups = session.query(Group).order_by(Group.created_at.desc()).limit(50).all()

                if groups:
                    groups_data = []
                    for group in groups:
                        member_count = session.query(GroupMember).filter_by(group_id=group.id).count()

                        groups_data.append({
                            'id': group.id,
                            'group_type': group.group_type,
                            'name': group.name,
                            'description': group.description or '',
                            'member_count': member_count,
                            'created_at': group.created_at.isoformat() if group.created_at else '',
                        })

                    groups_df = pd.DataFrame(groups_data)
                    groups_table.object = groups_df
                else:
                    groups_table.object = None

                members = session.query(GroupMember).order_by(GroupMember.added_at.desc()).limit(100).all()

                if members:
                    members_data = []
                    for member in members:
                        group = session.query(Group).filter_by(id=member.group_id).first()
                        call = session.query(RawCall).filter_by(id=member.call_id).first()

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

                    members_df = pd.DataFrame(members_data)
                    group_members_table.object = members_df
                else:
                    group_members_table.object = None

        except Exception as e:
            error_msg = f"Error loading groups: {str(e)}"
            logger.error(f"Failed to update groups: {str(e)}", exc_info=True)
            groups_table.object = None
            group_members_table.object = None

    refresh_button.on_click(lambda e: update_groups())

    update_groups()

    return pn.Column(
        pn.pane.Markdown("## Groups Management"),
        pn.Row(refresh_button, pn.Spacer()),
        pn.pane.Markdown("### Groups"),
        groups_table,
        pn.pane.Markdown("### Group Members"),
        group_members_table,
        sizing_mode='stretch_width',
        margin=(10, 20)
    )
