# Panel Starter Application

A modern, interactive data application built with [Panel](https://panel.holoviz.org/).

## Features

- 🎯 Interactive dashboard with real-time updates
- 📊 Data visualization with hvplot/holoviews
- 🔧 Modular component architecture
- 📓 Jupyter notebook integration
- 🚀 Ready for deployment

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/study-query-llm.git
cd study-query-llm

# Install in development mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

### Development Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Run as Standalone App

```bash
# Basic usage
panel serve panel_app/app.py --show

# Development mode with auto-reload
panel serve panel_app/app.py --show --dev

# Custom port
panel serve panel_app/app.py --port 5007 --show
```

### Use in Jupyter Notebook

```python
import panel as pn
from panel_app.app import create_app

pn.extension()

app = create_app()
app.servable()
```

### Run from Python Script

```python
from panel_app.app import main

if __name__ == "__main__":
    main()
```

## Project Structure

```
study-query-llm/
├── panel_app/           # Main application package
│   ├── app.py          # Main application entry point
│   ├── components/     # Reusable UI components
│   └── utils/          # Utility functions
├── notebooks/          # Jupyter notebooks
├── data/              # Sample data
├── assets/            # Static assets (CSS, images)
└── tests/             # Unit tests
```

## Customization

### Adding New Components

1. Create a new component in `panel_app/components/`
2. Import and use in `app.py`

### Styling

- Modify `assets/styles.css` for custom CSS
- Use Panel's built-in themes: `pn.extension(design='material')`
- Available themes: 'material', 'bootstrap', 'fast'

## Deployment

### Local Deployment

```bash
panel serve panel_app/app.py --port 5006 --allow-websocket-origin="*"
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["panel", "serve", "panel_app/app.py", "--port", "5006", "--allow-websocket-origin", "*"]
```

### Cloud Deployment

- **Heroku**: Add `Procfile` with `web: panel serve panel_app/app.py --port $PORT --allow-websocket-origin="*"`
- **Azure/AWS/GCP**: Use Docker container or custom deployment
- **Panel Server**: Deploy to any Python web server

## Features Included

- ✅ Responsive layout with FastListTemplate
- ✅ Interactive widgets (sliders, selects, buttons)
- ✅ Data visualization with hvplot
- ✅ File upload capability
- ✅ Caching for performance
- ✅ Modular component structure

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [Panel](https://panel.holoviz.org/) by HoloViz
- Inspired by awesome-panel.org
- Uses the HoloViz ecosystem