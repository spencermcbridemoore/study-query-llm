"""Inference testing view for the Panel application."""

import uuid

import pandas as pd
import panel as pn

from study_query_llm.config import config
from study_query_llm.db.models_v2 import Group, RawCall as RawCallModel
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.utils.logging_config import get_logger

import panel_app.helpers as _helpers
from panel_app.helpers import get_db_connection, get_inference_service

logger = get_logger(__name__)


def _source_tag(session, call_id: int) -> None:
    """Write source='panel_app' into a RawCall's metadata_json."""
    raw_call = session.query(RawCallModel).filter_by(id=call_id).first()
    if raw_call:
        meta = raw_call.metadata_json or {}
        meta["source"] = "panel_app"
        raw_call.metadata_json = meta


def _find_or_create_run(session, repository: RawCallRepository, run_name: str) -> int:
    """Return the id of an existing run group or create a new one."""
    existing = session.query(Group).filter(
        Group.group_type == "run",
        Group.name == run_name,
    ).first()
    if existing:
        return existing.id
    return repository.create_group(
        group_type="run",
        name=run_name,
        description="Run group created from Panel UI",
        metadata_json={"source": "panel_app"},
    )


def create_inference_ui() -> pn.viewable.Viewable:
    """Create the inference testing UI with Standalone Call and Batch Run subtabs."""

    # ------------------------------------------------------------------ #
    # Shared: provider + deployment selector (used by both subtabs)       #
    # ------------------------------------------------------------------ #

    provider_select = pn.widgets.Select(
        name="Provider",
        options=config.get_available_providers() or ["No providers configured"],
        value=config.get_available_providers()[0] if config.get_available_providers() else None,
        width=200,
    )

    deployment_select = pn.widgets.Select(
        name="Deployment",
        options=["Loading..."],
        value=None,
        width=220,
        visible=False,
    )

    load_deployments_button = pn.widgets.Button(
        name="Load Deployments",
        button_type="default",
        width=150,
        visible=False,
    )

    # Status line shared between provider controls and both subtabs
    shared_status = pn.pane.Str(
        "",
        styles={"color": "#666", "font-size": "12px"},
        sizing_mode="stretch_width",
    )

    async def load_deployments(event=None):
        if provider_select.value not in config.get_available_providers():
            return
        try:
            shared_status.object = "Loading deployments..."
            load_deployments_button.disabled = True
            factory = ProviderFactory()
            deployment_infos = await factory.list_provider_deployments(
                provider_select.value, modality="chat"
            )
            deployments = [d.id for d in deployment_infos]
            if deployments:
                deployment_select.options = deployments
                deployment_select.value = deployments[0]
                if provider_select.value == "azure":
                    azure_config = config.get_provider_config("azure")
                    azure_config.deployment_name = deployments[0]
                shared_status.object = f"Loaded {len(deployments)} deployment(s)"
            else:
                deployment_select.options = ["No deployments found"]
                shared_status.object = "No deployments found. Check provider configuration."
        except NotImplementedError:
            deployment_select.options = ["Not supported for this provider"]
            shared_status.object = "This provider doesn't support listing deployments"
        except Exception as e:
            deployment_select.options = ["Error loading deployments"]
            shared_status.object = f"Error: {str(e)}"
        finally:
            load_deployments_button.disabled = False

    async def update_provider_ui(event):
        if provider_select.value == "azure":
            deployment_select.visible = True
            load_deployments_button.visible = True
            await load_deployments(event)
        else:
            deployment_select.visible = False
            load_deployments_button.visible = False

    provider_select.param.watch(update_provider_ui, "value")
    load_deployments_button.on_click(lambda e: pn.state.execute(load_deployments, e))

    if provider_select.value == "azure":
        deployment_select.visible = True
        load_deployments_button.visible = True

    def update_deployment(event):
        if provider_select.value == "azure" and deployment_select.value:
            try:
                _helpers._inference_service = None
                _helpers._inference_service_provider = None
                if hasattr(config, "_provider_configs") and "azure" in config._provider_configs:
                    del config._provider_configs["azure"]
                azure_config = config.get_provider_config("azure")
                azure_config.deployment_name = deployment_select.value
            except Exception:
                pass

    deployment_select.param.watch(update_deployment, "value")

    # ------------------------------------------------------------------ #
    # Subtab 1: Standalone Call                                           #
    # ------------------------------------------------------------------ #

    temperature_slider = pn.widgets.FloatSlider(
        name="Temperature",
        start=0.0,
        end=2.0,
        value=0.7,
        step=0.1,
        width=280,
    )

    max_tokens_input = pn.widgets.IntInput(
        name="Max Tokens",
        value=None,
        start=1,
        end=100000,
        width=140,
    )

    run_name_input = pn.widgets.TextInput(
        name="Assign to Run (optional)",
        placeholder="Run name — leave blank to skip grouping",
        width=280,
    )

    prompt_input = pn.widgets.TextAreaInput(
        name="Prompt",
        placeholder="Enter your prompt here...",
        height=150,
        sizing_mode="stretch_width",
    )

    run_button = pn.widgets.Button(
        name="Run Inference",
        button_type="primary",
        width=150,
    )

    status_output = pn.pane.Str(
        "",
        styles={"color": "#666", "font-size": "12px"},
        sizing_mode="stretch_width",
    )

    response_output = pn.pane.Str(
        "",
        styles={"background": "#f5f5f5", "padding": "10px", "border-radius": "5px"},
        sizing_mode="stretch_width",
    )

    metadata_output = pn.pane.Markdown(
        "",
        styles={"background": "#f0f0f0", "padding": "10px", "border-radius": "5px", "font-size": "12px"},
        sizing_mode="stretch_width",
    )

    async def run_inference(event):
        if not prompt_input.value or not prompt_input.value.strip():
            status_output.object = "Please enter a prompt"
            return
        if provider_select.value == "No providers configured":
            status_output.object = "No providers configured. Please set API keys in .env file"
            return

        try:
            status_output.object = "Running inference..."
            run_button.disabled = True
            response_output.object = ""
            metadata_output.object = ""

            deployment_name = None
            if provider_select.value == "azure":
                if not deployment_select.value or deployment_select.value in (
                    "Loading...", "Error loading deployments", "No deployments found"
                ):
                    status_output.object = "Please load and select a deployment first"
                    run_button.disabled = False
                    return
                deployment_name = deployment_select.value

            service = get_inference_service(provider_select.value, deployment_name=deployment_name)

            db = get_db_connection()
            with db.session_scope() as session:
                repository = RawCallRepository(session)
                service.repository = repository

                result = await service.run_inference(
                    prompt_input.value,
                    temperature=temperature_slider.value,
                    max_tokens=max_tokens_input.value if max_tokens_input.value else None,
                )

                run_group_id = None

                if result.get("id"):
                    _source_tag(session, result["id"])

                    run_name = run_name_input.value.strip()
                    if run_name:
                        run_group_id = _find_or_create_run(session, repository, run_name)
                        repository.add_call_to_group(run_group_id, result["id"])

            response_output.object = result.get("response", "No response")

            metadata = result.get("metadata", {})
            latency = metadata.get("latency_ms")
            latency_str = f"{latency:.2f} ms" if latency is not None else "N/A"
            group_line = (
                f"**Run:** {run_name_input.value.strip()} (id={run_group_id})  \n"
                if run_group_id else ""
            )
            metadata_output.object = (
                f"**Provider:** {metadata.get('provider', provider_select.value)}  \n"
                f"**Deployment:** {metadata.get('deployment', deployment_name or 'N/A')}  \n"
                f"**Tokens:** {metadata.get('tokens', result.get('tokens', 'N/A'))}  \n"
                f"**Latency:** {latency_str}  \n"
                f"**Temperature:** {metadata.get('temperature', temperature_slider.value)}  \n"
                f"**Inference ID:** {result.get('id', 'N/A')}  \n"
                f"{group_line}"
            )
            status_output.object = "Inference complete!"

        except Exception as e:
            logger.error(
                "Standalone inference failed: provider=%s deployment=%s error=%s",
                provider_select.value, deployment_name, str(e),
                exc_info=True,
            )
            status_output.object = f"Error: {str(e)}"
            response_output.object = ""
            metadata_output.object = ""
        finally:
            run_button.disabled = False

    run_button.on_click(run_inference)

    standalone_tab = pn.Column(
        pn.Row(temperature_slider, pn.Spacer(width=20), max_tokens_input),
        run_name_input,
        prompt_input,
        run_button,
        status_output,
        pn.pane.Markdown("#### Response"),
        response_output,
        pn.pane.Markdown("#### Metadata"),
        metadata_output,
        sizing_mode="stretch_width",
        margin=(10, 0),
    )

    # ------------------------------------------------------------------ #
    # Subtab 2: Batch Run                                                 #
    # ------------------------------------------------------------------ #

    batch_temperature_slider = pn.widgets.FloatSlider(
        name="Temperature",
        start=0.0,
        end=2.0,
        value=0.7,
        step=0.1,
        width=280,
    )

    batch_max_tokens_input = pn.widgets.IntInput(
        name="Max Tokens",
        value=None,
        start=1,
        end=100000,
        width=140,
    )

    batch_n_input = pn.widgets.IntInput(
        name="Number of Calls",
        value=5,
        start=2,
        end=100,
        width=140,
    )

    batch_id_input = pn.widgets.TextInput(
        name="Batch ID",
        value=str(uuid.uuid4()),
        width=320,
    )

    regenerate_batch_id_button = pn.widgets.Button(
        name="New ID",
        button_type="light",
        width=80,
    )
    regenerate_batch_id_button.on_click(lambda e: setattr(batch_id_input, "value", str(uuid.uuid4())))

    batch_run_name_input = pn.widgets.TextInput(
        name="Assign to Run (optional)",
        placeholder="Parent run name — leave blank to skip",
        width=280,
    )

    batch_prompt_input = pn.widgets.TextAreaInput(
        name="Prompt",
        placeholder="Enter the prompt to repeat N times...",
        height=150,
        sizing_mode="stretch_width",
    )

    batch_run_button = pn.widgets.Button(
        name="Run Batch",
        button_type="primary",
        width=150,
    )

    batch_status_output = pn.pane.Str(
        "",
        styles={"color": "#666", "font-size": "12px"},
        sizing_mode="stretch_width",
    )

    batch_results_table = pn.pane.DataFrame(
        None,
        sizing_mode="stretch_width",
        height=350,
    )

    async def run_batch(event):
        if not batch_prompt_input.value or not batch_prompt_input.value.strip():
            batch_status_output.object = "Please enter a prompt"
            return
        if provider_select.value == "No providers configured":
            batch_status_output.object = "No providers configured. Please set API keys in .env file"
            return

        try:
            n = batch_n_input.value
            batch_id = batch_id_input.value.strip() or str(uuid.uuid4())
            batch_id_input.value = batch_id  # show the id actually used

            batch_status_output.object = f"Running {n} calls (batch_id={batch_id})..."
            batch_run_button.disabled = True
            batch_results_table.object = None

            deployment_name = None
            if provider_select.value == "azure":
                if not deployment_select.value or deployment_select.value in (
                    "Loading...", "Error loading deployments", "No deployments found"
                ):
                    batch_status_output.object = "Please load and select a deployment first"
                    batch_run_button.disabled = False
                    return
                deployment_name = deployment_select.value

            service = get_inference_service(provider_select.value, deployment_name=deployment_name)

            db = get_db_connection()
            with db.session_scope() as session:
                repository = RawCallRepository(session)
                service.repository = repository

                results = await service.run_sampling_inference(
                    batch_prompt_input.value,
                    n=n,
                    batch_id=batch_id,
                    temperature=batch_temperature_slider.value,
                    max_tokens=batch_max_tokens_input.value if batch_max_tokens_input.value else None,
                )

                # Source-tag every call and collect valid ids
                call_ids = []
                for r in results:
                    if r.get("id"):
                        _source_tag(session, r["id"])
                        call_ids.append(r["id"])

                # Create a batch group and add all calls to it
                batch_group_id = repository.create_group(
                    group_type="batch",
                    name=batch_id,
                    description=f"Batch of {n} identical calls from Panel UI",
                    metadata_json={"source": "panel_app", "n": n},
                )
                for cid in call_ids:
                    repository.add_call_to_group(batch_group_id, cid)

                # Optionally link to a parent run via GroupLink(contains)
                run_name = batch_run_name_input.value.strip()
                if run_name:
                    parent_run_id = _find_or_create_run(session, repository, run_name)
                    repository.create_group_link(
                        parent_group_id=parent_run_id,
                        child_group_id=batch_group_id,
                        link_type="contains",
                    )

            # Build results table
            rows = []
            for r in results:
                meta = r.get("metadata", {})
                latency = meta.get("latency_ms")
                rows.append({
                    "id": r.get("id", ""),
                    "response": (r.get("response", "") or "")[:80],
                    "tokens": meta.get("tokens", r.get("tokens", "")),
                    "latency_ms": f"{latency:.1f}" if latency is not None else "N/A",
                    "status": "ok" if r.get("response") else "error",
                })
            if rows:
                batch_results_table.object = pd.DataFrame(rows)

            batch_status_output.object = (
                f"Batch complete — {len(call_ids)}/{n} calls succeeded. "
                f"Batch group id={batch_group_id}."
            )

        except Exception as e:
            logger.error(
                "Batch inference failed: provider=%s deployment=%s error=%s",
                provider_select.value, deployment_name, str(e),
                exc_info=True,
            )
            batch_status_output.object = f"Error: {str(e)}"
            batch_results_table.object = None
        finally:
            batch_run_button.disabled = False

    batch_run_button.on_click(run_batch)

    batch_tab = pn.Column(
        pn.Row(batch_temperature_slider, pn.Spacer(width=20), batch_max_tokens_input, pn.Spacer(width=20), batch_n_input),
        pn.Row(batch_id_input, regenerate_batch_id_button),
        batch_run_name_input,
        batch_prompt_input,
        batch_run_button,
        batch_status_output,
        pn.pane.Markdown("#### Results"),
        batch_results_table,
        sizing_mode="stretch_width",
        margin=(10, 0),
    )

    # ------------------------------------------------------------------ #
    # Assemble full view                                                  #
    # ------------------------------------------------------------------ #

    subtabs = pn.Tabs(
        ("Standalone Call", standalone_tab),
        ("Batch Run", batch_tab),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.pane.Markdown("## Run Inference"),
        pn.Row(provider_select, deployment_select, load_deployments_button),
        shared_status,
        subtabs,
        sizing_mode="stretch_width",
        margin=(10, 20),
    )
