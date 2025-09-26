"""Minimal Panel application that serves a static dashboard."""

import panel as pn
from typing import Optional

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

# Configure Panel extensions once on import
pn.extension(
    design='material',
    sizing_mode='stretch_width',
    raw_css=_extra_css,
)


def create_dashboard() -> pn.viewable.Viewable:
    """Return the static dashboard content."""
    return pn.Column(
        pn.pane.Markdown("""
        # Study Query Dashboard
        
        Welcome to the barebones dashboard. This view is intentionally simple
        and does not expose any interactive controls or external data sources.
        """.strip()),
        pn.pane.Markdown("""
        ## Highlights
        - Static content only
        - No CSV or database dependencies
        - Ready to customise for your own metrics
        """.strip()),
        pn.layout.HSpacer(),
        sizing_mode="stretch_width",
    )


def create_app() -> pn.template.FastListTemplate:
    """Create and return the Panel application template."""
    dashboard = create_dashboard()
    template = pn.template.FastListTemplate(
        title="Barebones Dashboard",
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


def main() -> pn.template.FastListTemplate:
    """Main entry point for the application."""
    app = create_app()
    app.show(port=5006, open=True)
    return app


if __name__ == "__main__":
    app = create_app()
    app.servable()

if __name__.startswith("bokeh_app"):
    create_app().servable()
