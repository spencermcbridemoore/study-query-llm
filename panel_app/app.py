"""
Panel application for Study Query LLM.

Provides a web interface for running LLM inferences and analyzing results.
"""

import argparse
import os
import panel as pn
import asyncio
import time
from typing import Optional, Sequence, Set
import traceback

# Import core functionality
from study_query_llm.config import config
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.study_service import StudyService
from study_query_llm.utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

NOTEBOOK_THEME_RESET = """
body {
    background-color: var(--jp-layout-color0, inherit) !important;
}
"""

_extra_css = []
_state_env = getattr(pn.state, "env", None)
_is_notebook = bool(getattr(pn.state, "_is_notebook", False)) or _state_env == "notebook"
if _is_notebook:
    _extra_css.append(NOTEBOOK_THEME_RESET)

# Configure Panel extensions
pn.extension(
    design='material',
    sizing_mode='stretch_width',
    raw_css=_extra_css,
)


# Global state for database and services
_db_connection: Optional[DatabaseConnectionV2] = None
_inference_service: Optional[InferenceService] = None
_inference_service_provider: Optional[str] = None  # Track which provider the service is for
_study_service: Optional[StudyService] = None


def get_db_connection() -> DatabaseConnectionV2:
    """Get or create v2 database connection."""
    global _db_connection
    if _db_connection is None:
        logger.info("Initializing v2 database connection...")
        _db_connection = DatabaseConnectionV2(config.database.connection_string)
        _db_connection.init_db()
        logger.info("V2 database connection established")
    return _db_connection


def get_inference_service(provider_name: str, deployment_name: Optional[str] = None) -> InferenceService:
    """
    Get or create inference service for a provider.
    
    Args:
        provider_name: Name of the provider ('azure', 'openai', etc.)
        deployment_name: Optional deployment name to use (for Azure). If provided,
                        updates the config before creating the provider.
    """
    global _inference_service, _inference_service_provider
    
    # Always create a new service to ensure we use the latest deployment
    # (Don't cache since deployment can change)
    
    # Update config if deployment name provided (for Azure)
    if deployment_name and provider_name == "azure":
        # Clear cached config to force reload with new deployment
        if hasattr(config, '_provider_configs') and 'azure' in config._provider_configs:
            del config._provider_configs['azure']
        # Get config and update deployment
        azure_config = config.get_provider_config("azure")
        azure_config.deployment_name = deployment_name
    
    # Create factory and get provider from config
    factory = ProviderFactory()
    provider = factory.create_from_config(provider_name)
    
    # Create repository for database logging (use session scope for proper transaction handling)
    db = get_db_connection()
    session = db.get_session()
    repository = RawCallRepository(session)
    
    # Create service with repository (always create new to ensure fresh provider)
    service = InferenceService(provider, repository=repository)
    _inference_service_provider = provider_name
    
    return service


def get_study_service() -> StudyService:
    """Get or create study service for analytics."""
    global _study_service
    if _study_service is None:
        db = get_db_connection()
        session = db.get_session()
        repository = RawCallRepository(session)
        _study_service = StudyService(repository)
    return _study_service


def create_inference_ui() -> pn.viewable.Viewable:
    """Create the inference testing UI."""
    
    # UI Components
    provider_select = pn.widgets.Select(
        name="Provider",
        options=config.get_available_providers() or ["No providers configured"],
        value=config.get_available_providers()[0] if config.get_available_providers() else None,
        width=200
    )
    
    # Deployment selector (for Azure)
    deployment_select = pn.widgets.Select(
        name="Deployment",
        options=["Loading..."],
        value=None,
        width=200,
        visible=False  # Hidden by default, shown only for Azure
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
    
    # Load deployments function
    async def load_deployments(event=None):
        """Load available deployments from the provider."""
        if provider_select.value not in config.get_available_providers():
            return
        
        try:
            status_output.object = "⏳ Loading deployments..."
            load_deployments_button.disabled = True
            
            # Use factory to query deployments (provider-agnostic)
            factory = ProviderFactory()
            deployments = await factory.list_provider_deployments(provider_select.value)
            
            if deployments:
                deployment_select.options = deployments
                deployment_select.value = deployments[0]
                # Update the config with the selected deployment
                if provider_select.value == "azure":
                    azure_config = config.get_provider_config("azure")
                    azure_config.deployment_name = deployments[0]
                status_output.object = f"✅ Loaded {len(deployments)} deployment(s)"
            else:
                deployment_select.options = ["No deployments found"]
                status_output.object = "⚠️ No deployments found. Check provider configuration."
        except NotImplementedError:
            # Provider doesn't support listing deployments
            deployment_select.options = ["Not supported for this provider"]
            status_output.object = "ℹ️ This provider doesn't support listing deployments"
        except Exception as e:
            deployment_select.options = ["Error loading deployments"]
            status_output.object = f"❌ Error: {str(e)}"
        finally:
            load_deployments_button.disabled = False
    
    # Show/hide deployment selector based on provider
    async def update_provider_ui(event):
        """Update UI when provider changes."""
        if provider_select.value == "azure":
            deployment_select.visible = True
            load_deployments_button.visible = True
            # Try to load deployments automatically
            await load_deployments(event)
        else:
            deployment_select.visible = False
            load_deployments_button.visible = False
    
    provider_select.param.watch(update_provider_ui, 'value')
    load_deployments_button.on_click(lambda e: pn.state.execute(load_deployments, e))
    
    # Initialize UI if Azure is selected by default
    if provider_select.value == "azure":
        deployment_select.visible = True
        load_deployments_button.visible = True
    
    # Update deployment in config when selection changes (and clear service cache)
    def update_deployment(event):
        """Update config when deployment is selected."""
        global _inference_service, _inference_service_provider
        if provider_select.value == "azure" and deployment_select.value:
            try:
                # Clear cached service so it will be recreated with new deployment
                _inference_service = None
                _inference_service_provider = None
                # Clear cached config to force reload
                if hasattr(config, '_provider_configs') and 'azure' in config._provider_configs:
                    del config._provider_configs['azure']
                # Update the deployment in config
                azure_config = config.get_provider_config("azure")
                azure_config.deployment_name = deployment_select.value
            except Exception:
                pass  # Ignore errors during config update
    
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
    
    run_button = pn.widgets.Button(
        name="Run Inference",
        button_type="primary",
        width=150
    )
    
    # Response display
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
    
    # Async callback for running inference
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
            
            # Get service with selected deployment (if Azure)
            deployment_name = None
            if provider_select.value == "azure":
                if not deployment_select.value or deployment_select.value == "Loading..." or deployment_select.value == "Error loading deployments":
                    status_output.object = "⚠️ Please load and select a deployment first"
                    run_button.disabled = False
                    return
                deployment_name = deployment_select.value
            
            # Get service and run inference (service will use the deployment_name)
            service = get_inference_service(provider_select.value, deployment_name=deployment_name)
            
            # Run inference within database session scope
            db = get_db_connection()
            with db.session_scope() as session:
                # Create new repository with session for this transaction
                repository = RawCallRepository(session)
                service.repository = repository
                
                result = await service.run_inference(
                    prompt_input.value,
                    temperature=temperature_slider.value,
                    max_tokens=max_tokens_input.value if max_tokens_input.value else None
                )
            
            # Display results
            response_output.object = result.get('response', 'No response')
            
            # Format metadata
            metadata = result.get('metadata', {})
            metadata_text = f"""
**Provider:** {metadata.get('provider', 'N/A')}  
**Tokens:** {metadata.get('tokens', 'N/A')}  
**Latency:** {metadata.get('latency_ms', 'N/A'):.2f} ms  
**Temperature:** {metadata.get('temperature', 'N/A')}  
**Inference ID:** {result.get('id', 'N/A')}
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
    
    # Connect button to callback
    run_button.on_click(run_inference)
    
    # Layout
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
                
                # Get summary stats
                stats = study_service.get_summary_stats()
                logger.debug(f"Analytics stats: {stats}")
                summary_text = f"""
**Total Inferences:** {stats['total_inferences']}  
**Total Tokens:** {stats['total_tokens']:,}  
**Unique Providers:** {stats['unique_providers']}
"""
                summary_output.object = summary_text
                
                # Get provider comparison
                comparison_df = study_service.get_provider_comparison()
                if not comparison_df.empty:
                    # Select relevant columns for display
                    display_cols = ['provider', 'count', 'avg_tokens', 'avg_latency_ms', 'total_tokens']
                    if 'avg_cost_estimate' in comparison_df.columns:
                        display_cols.append('avg_cost_estimate')
                    provider_comparison_table.object = comparison_df[display_cols]
                else:
                    provider_comparison_table.object = None
                
                # Get recent inferences
                recent_df = study_service.get_recent_inferences(limit=20)
                if not recent_df.empty:
                    # Select and format columns for display
                    display_df = recent_df[['id', 'prompt', 'provider', 'tokens', 'latency_ms', 'created_at']].copy()
                    # Truncate long prompts for display
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
    
    # Initial load
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


def create_dashboard() -> pn.viewable.Viewable:
    """Create the main dashboard with tabs."""
    
    inference_tab = create_inference_ui()
    analytics_tab = create_analytics_ui()
    
    tabs = pn.Tabs(
        ("Inference", inference_tab),
        ("Analytics", analytics_tab),
        sizing_mode='stretch_width'
    )
    
    return tabs


def create_app() -> pn.template.FastListTemplate:
    """Create and return the Panel application template."""
    logger.info("Creating Panel application...")
    dashboard = create_dashboard()
    template = pn.template.FastListTemplate(
        title="Study Query LLM",
        sidebar=[],
        main=[dashboard],
        header_background="#2596be",
        header_color="#FFFFFF",
    )
    return template


def serve_app(
    address: str = "localhost",
    port: int = 5006,
    route: Optional[str] = None,
    open_browser: bool = False,
    **kwargs,
):
    """Start the dashboard on a background server and return the handle and URL."""
    app = create_app()

    if route:
        normalized = route.strip("/")
        if normalized:
            objects = {normalized: app}
            url_path = f"/{normalized}"
        else:
            objects = app
            url_path = "/"
    else:
        objects = app
        url_path = "/"

    serve_kwargs = {
        "address": address,
        "port": port,
        "show": open_browser,
        "start": True,
    }
    serve_kwargs.update(kwargs)
    serve_kwargs.setdefault("threaded", True)

    server = pn.serve(objects, **serve_kwargs)

    resolved_address = address or "localhost"
    resolved_port = getattr(server, "port", port)
    url = f"http://{resolved_address}:{resolved_port}{url_path}"

    return server, url


def _add_health_route(server) -> None:
    """Expose `/health` via the underlying Tornado application."""
    try:
        from tornado.web import RequestHandler
    except Exception as exc:  # pragma: no cover - tornado available in runtime
        logger.warning("Unable to import Tornado for healthcheck: %s", exc)
        return

    tornado_app = getattr(server, "_tornado", None)
    if tornado_app is None:
        logger.debug("Server has no Tornado application; skipping health route")
        return

    class HealthHandler(RequestHandler):
        def get(self):
            self.set_header("Content-Type", "application/json")
            self.write({"status": "ok"})

    tornado_app.add_handlers(r".*", [(r"/health", HealthHandler)])


def run_server(address: str, port: int, extra_origins: Optional[Set[str]] = None) -> None:
    """Start the Panel server using CLI/environment configuration."""
    allowed_ws: Set[str] = {
        f"{address}:{port}",
        "localhost:5006",
        "127.0.0.1:5006",
    }
    env_extra = os.getenv("PANEL_ALLOW_WS_ORIGINS", "")
    if env_extra:
        allowed_ws.update(
            {origin.strip() for origin in env_extra.split(",") if origin.strip()}
        )
    if extra_origins:
        allowed_ws.update(extra_origins)

    logger.info(
        "Starting Panel server on %s:%s with allowed origins %s",
        address,
        port,
        ", ".join(sorted(allowed_ws)),
    )
    server, url = serve_app(
        address=address,
        port=port,
        open_browser=False,
        threaded=False,
        start=False,
        allow_websocket_origin=sorted(allowed_ws),
    )
    _add_health_route(server)
    server.start()
    logger.info("Panel application available at %s", url)

    try:
        if hasattr(server, "io_loop"):
            server.io_loop.start()
        elif hasattr(server, "join"):
            server.join()
        else:
            logger.info("Server handle lacks io_loop/join; sleeping until interrupted")
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received, stopping Panel server")
    finally:
        if hasattr(server, "stop"):
            server.stop()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for running the Panel application."""
    parser = argparse.ArgumentParser(description="Run the Study Query LLM Panel app")
    parser.add_argument(
        "--address",
        default=os.getenv("PANEL_ADDRESS", "0.0.0.0"),
        help="Interface to bind the Panel server (env: PANEL_ADDRESS)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PANEL_PORT", "5006")),
        help="Port for the Panel server (env: PANEL_PORT)",
    )
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        dest="allow_origins",
        help="Additional websocket origins (repeatable).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    extra_origins = {origin for origin in args.allow_origins or []}
    run_server(address=args.address, port=args.port, extra_origins=extra_origins)


def _running_inside_panel_server() -> bool:
    """Detect if the module is being executed inside `panel serve`."""
    try:
        session_context = getattr(pn.state.curdoc, "session_context", None)
        return session_context is not None or __name__.startswith("bokeh_app")
    except Exception:
        return False


if _running_inside_panel_server():
    create_app().servable()
elif __name__ == "__main__":
    main()
