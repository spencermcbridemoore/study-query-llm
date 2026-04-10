"""
Panel application for Study Query LLM.

Provides a web interface for running LLM inferences and analyzing results.
"""

import argparse
import os
import signal
import socket
import sys
import threading
import panel as pn
import time
from pathlib import Path
from typing import Optional, Sequence, Set

# Ensure the project root is on sys.path so `panel serve panel_app/app.py`
# can resolve package imports without a manual PYTHONPATH override.
_project_root_path = Path(__file__).resolve().parent.parent
_project_root = str(_project_root_path)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load repo .env before any study_query_llm import so DATABASE_URL is set under
# ``panel serve`` (Bokeh cwd is not always the repo root; empty DATABASE_URL in
# the environment would otherwise skip dotenv merge with override=False).
try:
    from dotenv import load_dotenv

    _env_file = _project_root_path / ".env"
    if _env_file.is_file() and not str(os.environ.get("DATABASE_URL", "") or "").strip():
        load_dotenv(_env_file, encoding="utf-8", override=True)
except ImportError:
    pass

from study_query_llm.utils.logging_config import get_logger, setup_logging

from panel_app.helpers import HEADER_BG, HEADER_FG, get_database_health_markdown
from panel_app.views.inference import create_inference_ui
from panel_app.views.analytics import create_analytics_ui
from panel_app.views.groups import create_groups_ui
from panel_app.views.embeddings import create_embeddings_ui
from panel_app.views.sweep_explorer import create_sweep_explorer_ui
from panel_app.views.sweep_explorer_perspective import create_sweep_explorer_perspective_ui
from panel_app.views.storage_stats import create_storage_stats_ui

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Module-level state for graceful shutdown coordination
_active_server = None
_shutdown_flag = False

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
    'perspective',
    design='material',
    sizing_mode='stretch_width',
    raw_css=_extra_css,
)


def _check_port_available(address: str, port: int) -> None:
    """Fail fast if *port* is already occupied by another process."""
    bind_addr = "127.0.0.1" if address == "localhost" else address
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((bind_addr, port))
    except OSError:
        logger.error(
            "Port %d is already in use on %s. "
            "Another Panel server may be running. "
            "Stop the existing process or use --port to choose a different port.",
            port,
            address,
        )
        raise SystemExit(1)
    finally:
        sock.close()


def _request_shutdown() -> None:
    """Gracefully stop the active server, falling back to SIGTERM."""
    global _shutdown_flag
    if _shutdown_flag:
        return
    _shutdown_flag = True

    server = _active_server
    if server is not None:
        io_loop = getattr(server, "io_loop", None)
        if io_loop is not None:
            logger.info("Requesting graceful IO-loop shutdown (PID %s).", os.getpid())
            io_loop.add_callback(io_loop.stop)
            return

    logger.info(
        "No server handle available; sending SIGTERM to self (PID %s).", os.getpid()
    )
    os.kill(os.getpid(), signal.SIGTERM)


def create_dashboard() -> pn.viewable.Viewable:
    """Create the main dashboard with tabs."""

    connection_banner = pn.pane.Markdown(
        get_database_health_markdown(),
        sizing_mode="stretch_width",
    )

    inference_tab = create_inference_ui()
    analytics_tab = create_analytics_ui()
    groups_tab = create_groups_ui()
    embeddings_tab = create_embeddings_ui()
    sweep_explorer_tab = create_sweep_explorer_ui()
    sweep_explorer_perspective_tab = create_sweep_explorer_perspective_ui()
    storage_stats_tab = create_storage_stats_ui()

    tabs = pn.Tabs(
        ("Inference", inference_tab),
        ("Analytics", analytics_tab),
        ("Embeddings", embeddings_tab),
        ("Groups", groups_tab),
        ("Storage / DB stats", storage_stats_tab),
        ("Sweep Explorer", sweep_explorer_tab),
        ("Sweep Explorer (Perspective)", sweep_explorer_perspective_tab),
        sizing_mode='stretch_width',
        # Lazy tabs can leave some panes blank until visited; render all up front.
        dynamic=False,
    )

    return pn.Column(
        connection_banner,
        pn.layout.Divider(),
        tabs,
        sizing_mode="stretch_width",
    )


def create_app() -> pn.template.FastListTemplate:
    """Create and return the Panel application template."""
    logger.info("Creating Panel application...")
    dashboard = create_dashboard()

    exit_button = pn.widgets.Button(
        name="⏻  Exit Server",
        button_type="danger",
        width=130,
    )

    def _shutdown(event):
        logger.info(
            "Exit button clicked — shutting down Panel server (PID %s).", os.getpid()
        )
        exit_button.disabled = True
        exit_button.name = "Shutting down…"
        threading.Timer(0.5, _request_shutdown).start()

    exit_button.on_click(_shutdown)

    template = pn.template.FastListTemplate(
        title="Study Query LLM",
        sidebar=[],
        main=[dashboard],
        header_background=HEADER_BG,
        header_color=HEADER_FG,
        header=[exit_button],
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
    global _active_server, _shutdown_flag
    _shutdown_flag = False

    _check_port_available(address, port)

    allowed_ws: Set[str] = {
        f"{address}:{port}",
        f"localhost:{port}",
        f"127.0.0.1:{port}",
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
        "Starting Panel server on %s:%s (PID %s) with allowed origins %s",
        address,
        port,
        os.getpid(),
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
    _active_server = server
    server.start()
    logger.info("Panel application available at %s (PID %s)", url, os.getpid())

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
        _active_server = None


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
