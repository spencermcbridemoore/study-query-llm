"""Main application entry point"""

import panel as pn
import pandas as pd
import numpy as np
from pathlib import Path

from .components.dashboard import Dashboard
from .components.widgets import create_sidebar_widgets
from .utils.data_loader import load_sample_data

# Configure Panel extensions
pn.extension(
    'tabulator',  # For advanced tables
    design='material',  # Material design theme
    sizing_mode='stretch_width'  # Responsive layout
)

# Cache data loading for performance
@pn.cache
def get_data():
    """Load and cache the application data"""
    return load_sample_data()

def create_app():
    """Create and return the Panel application"""
    
    # Load data
    data = get_data()
    
    # Create dashboard
    dashboard = Dashboard(data)
    
    # Create sidebar widgets
    sidebar_widgets = create_sidebar_widgets(dashboard)
    
    # Create template
    template = pn.template.FastListTemplate(
        title="Panel Starter Dashboard",
        sidebar=[
            "# Controls",
            pn.pane.Markdown("""
            Use these controls to interact with the dashboard.
            """),
            *sidebar_widgets,
            pn.layout.Divider(),
            "## About",
            pn.pane.Markdown("""
            This is a starter template for Panel applications.
            
            **Features:**
            - Interactive widgets
            - Real-time updates
            - Data visualization
            - Responsive design
            
            Built with ❤️ using [Panel](https://panel.holoviz.org)
            """),
        ],
        main=[dashboard.view()],
        header_background='#2596be',
        header_color='#FFFFFF',
    )
    
    return template

def main():
    """Main entry point for the application"""
    app = create_app()
    
    # Serve the application
    app.show(port=5006, open=True)
    
    return app

# Make the app servable
if __name__ == "__main__":
    app = create_app()
    app.servable()
    
# For panel serve
if __name__.startswith("bokeh_app"):
    create_app().servable()