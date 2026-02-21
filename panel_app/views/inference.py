"""Inference testing view for the Panel application."""

import panel as pn

from study_query_llm.config import config
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.utils.logging_config import get_logger

import panel_app.helpers as _helpers
from panel_app.helpers import get_db_connection, get_inference_service

logger = get_logger(__name__)


def create_inference_ui() -> pn.viewable.Viewable:
    """Create the inference testing UI."""

    # UI Components
    provider_select = pn.widgets.Select(
        name="Provider",
        options=config.get_available_providers() or ["No providers configured"],
        value=config.get_available_providers()[0] if config.get_available_providers() else None,
        width=200
    )

    deployment_select = pn.widgets.Select(
        name="Deployment",
        options=["Loading..."],
        value=None,
        width=200,
        visible=False
    )

    load_deployments_button = pn.widgets.Button(
        name="Load Deployments",
        button_type="default",
        width=150,
        visible=False
    )

    prompt_input = pn.widgets.TextAreaInput(
        name="Prompt",
        placeholder="Enter your prompt here...",
        height=150,
        sizing_mode='stretch_width'
    )

    async def load_deployments(event=None):
        """Load available deployments from the provider."""
        if provider_select.value not in config.get_available_providers():
            return

        try:
            status_output.object = "⏳ Loading deployments..."
            load_deployments_button.disabled = True

            factory = ProviderFactory()
            deployments = await factory.list_provider_deployments(provider_select.value)

            if deployments:
                deployment_select.options = deployments
                deployment_select.value = deployments[0]
                if provider_select.value == "azure":
                    azure_config = config.get_provider_config("azure")
                    azure_config.deployment_name = deployments[0]
                status_output.object = f"✅ Loaded {len(deployments)} deployment(s)"
            else:
                deployment_select.options = ["No deployments found"]
                status_output.object = "⚠️ No deployments found. Check provider configuration."
        except NotImplementedError:
            deployment_select.options = ["Not supported for this provider"]
            status_output.object = "ℹ️ This provider doesn't support listing deployments"
        except Exception as e:
            deployment_select.options = ["Error loading deployments"]
            status_output.object = f"❌ Error: {str(e)}"
        finally:
            load_deployments_button.disabled = False

    async def update_provider_ui(event):
        """Update UI when provider changes."""
        if provider_select.value == "azure":
            deployment_select.visible = True
            load_deployments_button.visible = True
            await load_deployments(event)
        else:
            deployment_select.visible = False
            load_deployments_button.visible = False

    provider_select.param.watch(update_provider_ui, 'value')
    load_deployments_button.on_click(lambda e: pn.state.execute(load_deployments, e))

    if provider_select.value == "azure":
        deployment_select.visible = True
        load_deployments_button.visible = True

    def update_deployment(event):
        """Update config when deployment is selected."""
        if provider_select.value == "azure" and deployment_select.value:
            try:
                _helpers._inference_service = None
                _helpers._inference_service_provider = None
                if hasattr(config, '_provider_configs') and 'azure' in config._provider_configs:
                    del config._provider_configs['azure']
                azure_config = config.get_provider_config("azure")
                azure_config.deployment_name = deployment_select.value
            except Exception:
                pass

    deployment_select.param.watch(update_deployment, 'value')

    temperature_slider = pn.widgets.FloatSlider(
        name="Temperature",
        start=0.0,
        end=2.0,
        value=0.7,
        step=0.1,
        width=300
    )

    max_tokens_input = pn.widgets.IntInput(
        name="Max Tokens",
        value=None,
        start=1,
        end=100000,
        width=150
    )

    enable_grouping = pn.widgets.Checkbox(
        name="Enable Grouping",
        value=False,
        width=150
    )

    group_type_select = pn.widgets.Select(
        name="Group Type",
        options=["batch", "experiment", "label", "custom"],
        value="batch",
        width=150,
        visible=False
    )

    group_name_input = pn.widgets.TextInput(
        name="Group Name",
        placeholder="Enter group name...",
        width=200,
        visible=False
    )

    group_role_input = pn.widgets.TextInput(
        name="Role (optional)",
        placeholder="e.g., 'input', 'output'",
        width=150,
        visible=False
    )

    def update_grouping_ui(event):
        """Update UI when grouping checkbox changes."""
        if enable_grouping.value:
            group_type_select.visible = True
            group_name_input.visible = True
            group_role_input.visible = True
        else:
            group_type_select.visible = False
            group_name_input.visible = False
            group_role_input.visible = False

    enable_grouping.param.watch(update_grouping_ui, 'value')

    run_button = pn.widgets.Button(
        name="Run Inference",
        button_type="primary",
        width=150
    )

    response_output = pn.pane.Str(
        "",
        styles={'background': '#f5f5f5', 'padding': '10px', 'border-radius': '5px'},
        sizing_mode='stretch_width'
    )

    metadata_output = pn.pane.Str(
        "",
        styles={'background': '#f0f0f0', 'padding': '10px', 'border-radius': '5px', 'font-size': '12px'},
        sizing_mode='stretch_width'
    )

    status_output = pn.pane.Str(
        "",
        styles={'color': '#666', 'font-size': '12px'},
        sizing_mode='stretch_width'
    )

    async def run_inference(event):
        """Run inference when button is clicked."""
        if not prompt_input.value or not prompt_input.value.strip():
            status_output.object = "⚠️ Please enter a prompt"
            return

        if provider_select.value == "No providers configured":
            status_output.object = "⚠️ No providers configured. Please set API keys in .env file"
            return

        try:
            status_output.object = "⏳ Running inference..."
            run_button.disabled = True
            response_output.object = ""
            metadata_output.object = ""

            deployment_name = None
            if provider_select.value == "azure":
                if not deployment_select.value or deployment_select.value == "Loading..." or deployment_select.value == "Error loading deployments":
                    status_output.object = "⚠️ Please load and select a deployment first"
                    run_button.disabled = False
                    return
                deployment_name = deployment_select.value

            service = get_inference_service(provider_select.value, deployment_name=deployment_name)

            db = get_db_connection()
            with db.session_scope() as session:
                repository = RawCallRepository(session)
                service.repository = repository

                batch_id = None
                group_id = None
                if enable_grouping.value and group_name_input.value:
                    from study_query_llm.db.models_v2 import Group
                    import uuid

                    batch_id = str(uuid.uuid4())

                    existing_groups = session.query(Group).filter(
                        Group.group_type == group_type_select.value,
                        Group.name == group_name_input.value
                    ).all()

                    if existing_groups:
                        group_id = existing_groups[0].id
                    else:
                        group_id = repository.create_group(
                            group_type=group_type_select.value,
                            name=group_name_input.value,
                            description="Group created from Panel UI",
                            metadata_json={"batch_id": batch_id}
                        )

                result = await service.run_inference(
                    prompt_input.value,
                    temperature=temperature_slider.value,
                    max_tokens=max_tokens_input.value if max_tokens_input.value else None,
                    batch_id=batch_id
                )

                if enable_grouping.value and group_name_input.value and result.get('id') and group_id:
                    if group_role_input.value:
                        from study_query_llm.db.models_v2 import GroupMember
                        existing_member = session.query(GroupMember).filter_by(
                            group_id=group_id,
                            call_id=result['id']
                        ).first()

                        if existing_member:
                            existing_member.role = group_role_input.value
                        else:
                            repository.add_call_to_group(
                                group_id=group_id,
                                call_id=result['id'],
                                role=group_role_input.value
                            )

            response_output.object = result.get('response', 'No response')

            metadata = result.get('metadata', {})
            group_info = ""
            if enable_grouping.value and group_name_input.value and result.get('group_id'):
                group_info = f"**Group ID:** {result.get('group_id', 'N/A')}  \n"

            metadata_text = f"""
**Provider:** {metadata.get('provider', 'N/A')}  
**Tokens:** {metadata.get('tokens', 'N/A')}  
**Latency:** {metadata.get('latency_ms', 'N/A'):.2f} ms  
**Temperature:** {metadata.get('temperature', 'N/A')}  
**Inference ID:** {result.get('id', 'N/A')}  
{group_info}
"""
            metadata_output.object = metadata_text
            status_output.object = "✅ Inference complete!"

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(
                f"Inference failed: provider={provider_select.value}, "
                f"deployment={deployment_name}, error={str(e)}",
                exc_info=True
            )
            status_output.object = error_msg
            response_output.object = ""
            metadata_output.object = ""
        finally:
            run_button.disabled = False

    run_button.on_click(run_inference)

    return pn.Column(
        pn.pane.Markdown("## Run Inference"),
        pn.Row(
            provider_select,
            deployment_select,
            load_deployments_button,
            pn.Spacer(width=20),
            temperature_slider,
            pn.Spacer(width=20),
            max_tokens_input,
        ),
        pn.pane.Markdown("### Grouping (Optional)"),
        pn.Row(
            enable_grouping,
            group_type_select,
            group_name_input,
            group_role_input,
        ),
        prompt_input,
        run_button,
        status_output,
        pn.pane.Markdown("### Response"),
        response_output,
        pn.pane.Markdown("### Metadata"),
        metadata_output,
        sizing_mode='stretch_width',
        margin=(10, 20)
    )
