# Study Query LLM

A Panel-based web application for running LLM inference experiments across multiple providers (Azure OpenAI, OpenAI, Hyperbolic) and analyzing the results stored in a PostgreSQL database with Langfuse integration.

## Overview

This application enables you to:
- Run LLM inferences across different providers with a unified interface
- Automatically log all inference runs (prompts, responses, metadata) to a database
- Query and analyze historical inference data
- Compare provider performance (latency, token usage, cost)
- Search and explore past experiments
- Visualize trends over time

## Architecture

The project follows a clean, layered architecture:

```
GUI (Panel) → Services (Business Logic) → Providers (LLM APIs) + Repository (Database)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Implementation Status

**Current Status:** Minimal Panel boilerplate - core functionality in development

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the complete phased implementation plan.

**Next Step:** Phase 1 - Provider abstraction layer

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL (or SQLite for development)
- API keys for LLM providers you want to use

### Setup

1. **Clone and create virtual environment**
```bash
git clone <repo-url>
cd study-query-llm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -e .
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys and database URL
```

4. **Initialize database**
```bash
# When database layer is implemented
python -c "from study_query_llm.db.connection import DatabaseConnection; db = DatabaseConnection('sqlite:///study_query_llm.db'); db.init_db()"
```

## Usage

### Running the Panel App

**As a server:**
```bash
panel serve panel_app/app.py --show
```

**In a Jupyter notebook:**
```python
from panel_app import init_panel, create_app, serve_app

init_panel()
app = create_app()
app.servable()
```

### Configuration

Set environment variables or create a `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/study_query_llm
# Or for development:
DATABASE_URL=sqlite:///study_query_llm.db

# Azure OpenAI
AZURE_API_KEY=your-azure-key
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_DEPLOYMENT=gpt-4

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4

# Hyperbolic
HYPERBOLIC_API_KEY=your-hyperbolic-key

# Panel Theme (optional)
PANEL_APP_THEME=dark  # or 'default'
```

## Development

### Project Structure

```
study-query-llm/
├── src/
│   └── study_query_llm/    # Core package (framework-agnostic)
│       ├── __init__.py
│       ├── providers/      # LLM provider abstractions
│       ├── services/       # Business logic layer (to be created)
│       └── db/             # Database models and repository (to be created)
├── panel_app/              # Panel GUI application
│   ├── __init__.py
│   └── app.py
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   └── IMPLEMENTATION_PLAN.md  # Development roadmap
├── .claude/                # AI assistant context
│   └── context.md          # Quick reference for Claude
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suite (to be created)
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=study_query_llm tests/
```

### Adding a New LLM Provider

1. Create `src/study_query_llm/providers/your_provider.py`
2. Implement the `BaseLLMProvider` interface
3. Add to `ProviderFactory`
4. Update configuration
5. Add tests

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Features

### Current
- ✅ Panel web interface foundation
- ✅ Reactive UI components
- ✅ Theme customization

### Planned (See IMPLEMENTATION_PLAN.md)
- [ ] Multi-provider LLM support (Azure, OpenAI, Hyperbolic)
- [ ] Automatic inference logging to database
- [ ] Query and analytics interface
- [ ] Provider performance comparison
- [ ] Retry logic with exponential backoff
- [ ] Prompt preprocessing and templates
- [ ] Multi-turn conversation support
- [ ] Request batching and deduplication
- [ ] Cost tracking
- [ ] Vector similarity search
- [ ] Export functionality

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System design, layer responsibilities, data flow
- **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** - Phased development roadmap with detailed steps
- **[Claude Context](.claude/context.md)** - Quick reference for AI-assisted development

## Contributing

This project follows a bottom-up, incremental development approach. Each component is designed to be:
- Testable in isolation
- Independent of layers above it
- Composable with other components

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the current development phase and next steps.

## License

[Add your license here]

## Contact

[Add your contact information here]
