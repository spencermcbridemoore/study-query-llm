Barebones Panel Dashboard

A minimal Panel project that renders a static dashboard page. Use this as a clean starting point without data dependencies, widgets, or complex component wiring.

Features:
- Simple FastListTemplate layout with static content
- Works as a standalone server or from Jupyter notebooks
- Helper to serve the app on any address or port
- Lightweight dependency footprint (Panel only)

Installation:
1. git clone https://github.com/yourusername/study-query-llm.git
2. cd study-query-llm
3. pip install -e .

(Optional) Install Jupyter if you plan to run the demo notebook: pip install jupyter

Standalone server:
panel serve panel_app/app.py --show --port 5006

Python entry point:
from panel_app.app import main
if __name__ == "__main__":
    main()

Notebook snippet:
from panel_app.app import create_app, serve_app
app = create_app()
app.servable()
server, url = serve_app(port=5050)
print(url)

Project structure:
study-query-llm/
├── panel_app/ (app.py plus empty component and util packages)
├── notebooks/
├── assets/
└── requirements.txt

Customisation:
- Edit create_dashboard() in panel_app/app.py to add panes or layouts.
- Add modules under panel_app/components/ or panel_app/utils/ as needed.
- Extend requirements.txt when you introduce additional libraries.

License: MIT License - see LICENSE for details.
