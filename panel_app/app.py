"""
Panel application for Study Query LLM.

Provides a web interface for running LLM inferences and analyzing results.
"""

import argparse
import os
import panel as pn
import time
from typing import Optional, Sequence, Set

from study_query_llm.utils.logging_config import get_logger, setup_logging

from panel_app.helpers import HEADER_BG, HEADER_FG
from panel_app.views.inference import create_inference_ui
from panel_app.views.analytics import create_analytics_ui
from panel_app.views.groups import create_groups_ui
from panel_app.views.embeddings import create_embeddings_ui
from panel_app.views.sweep_explorer import create_sweep_explorer_ui

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
    'plotly',
    design='material',
    sizing_mode='stretch_width',
    raw_css=_extra_css,
)


def create_dashboard() -> pn.viewable.Viewable:
    """Create the main dashboard with tabs."""

    inference_tab = create_inference_ui()
    analytics_tab = create_analytics_ui()
    groups_tab = create_groups_ui()
    embeddings_tab = create_embeddings_ui()
    sweep_explorer_tab = create_sweep_explorer_ui()

    tabs = pn.Tabs(
        ("Inference", inference_tab),
        ("Analytics", analytics_tab),
        ("Embeddings", embeddings_tab),
        ("Groups", groups_tab),
        ("Sweep Explorer", sweep_explorer_tab),
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
        header_background=HEADER_BG,
        header_color=HEADER_FG,
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
