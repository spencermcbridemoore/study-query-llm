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

**Current Status:** ✅ **Production Ready** - All core features implemented and tested

### Completed Phases
- ✅ **Phase 1:** LLM Provider Abstraction Layer (Azure, OpenAI, Hyperbolic)
- ✅ **Phase 1.5:** Provider Factory
- ✅ **Phase 2:** Business Logic Layer (retry, preprocessing, batch/sampling)
- ✅ **Phase 3:** Database Layer (SQLAlchemy, repository pattern)
- ✅ **Phase 4:** Analytics/Study Service
- ✅ **Phase 5:** GUI Integration (Panel web interface)
- ✅ **Phase 6.1:** Error Handling and Logging
- ✅ **Phase 6.2:** Unit Tests (112 tests passing)

### Current Features
- Multi-provider LLM support (Azure OpenAI, OpenAI, Hyperbolic)
- Automatic inference logging to database
- Analytics dashboard with provider comparison
- Batch and sampling inference support
- Retry logic with exponential backoff
- Prompt preprocessing
- Comprehensive error handling and logging
- Full test coverage (112 tests)

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the complete implementation plan.

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
pip install -e ".[dev]"
```

   This installs the package with development dependencies including pytest.

3. **Configure environment**
   
   Create a `.env` file in the project root:
   ```bash
   # Database (SQLite for development, PostgreSQL for production)
   DATABASE_URL=sqlite:///study_query_llm.db
   
   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your-azure-api-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   
   # OpenAI (optional)
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL=gpt-4
   
   # Hyperbolic (optional)
   HYPERBOLIC_API_KEY=your-hyperbolic-api-key
   HYPERBOLIC_ENDPOINT=https://api.hyperbolic.xyz
   ```

4. **Initialize database**
```bash
python -c "from study_query_llm.db.connection import DatabaseConnection; from study_query_llm.config import config; db = DatabaseConnection(config.database.connection_string); db.init_db()"
```

### Docker (Optional)

The repository ships with a multi-stage `Dockerfile` and a `docker-compose.yml`
stack for running the Panel app plus optional Postgres instance:

```bash
docker build -t study-query-llm:local .
docker compose up --build
```

See `docs/DEPLOYMENT.md` for detailed instructions, environment variables, and
Postgres profile usage.

## Usage

### Running the Panel App

**As a server:**
```bash
panel serve panel_app/app.py --show
```

**In a Jupyter notebook:**
```python
from panel_app.app import create_app

app = create_app()
app.servable()
```

### Quick Start Guide

1. **Start the application:**
   ```bash
   panel serve panel_app/app.py --show
   ```

2. **Use the Inference tab:**
   - Select a provider (Azure, OpenAI, or Hyperbolic)
   - For Azure: Click "Load Deployments" and select a deployment
   - Enter your prompt
   - Adjust temperature and max tokens (optional)
   - Click "Run Inference"

3. **View analytics:**
   - Switch to the Analytics tab
   - See summary statistics, provider comparison, and recent inferences
   - Click "Refresh" to update data

For detailed usage instructions, see the [User Guide](docs/USER_GUIDE.md).

## Development

### Project Structure

```
study-query-llm/
├── src/
│   └── study_query_llm/    # Core package (framework-agnostic)
│       ├── __init__.py
│       ├── providers/      # LLM provider abstractions (Azure, OpenAI, Hyperbolic)
│       ├── services/       # Business logic layer (inference, study, preprocessing)
│       ├── db/             # Database models and repository
│       ├── utils/          # Utilities (logging, etc.)
│       └── config.py       # Configuration management
├── panel_app/              # Panel GUI application
│   ├── __init__.py
│   └── app.py              # Main application entry point
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── IMPLEMENTATION_PLAN.md  # Development roadmap
│   └── USER_GUIDE.md       # User guide
├── tests/                  # Test suite (112 tests)
│   ├── test_providers/     # Provider tests
│   ├── test_services/      # Service tests
│   └── test_db/            # Database tests
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest --cov=study_query_llm tests/

# Run specific test file
pytest tests/test_providers/test_azure.py
```

**Test Coverage:** 112 tests covering providers, services, and database operations.

### Adding a New LLM Provider

1. Create `src/study_query_llm/providers/your_provider.py`
2. Implement the `BaseLLMProvider` interface
3. Add to `ProviderFactory`
4. Update configuration
5. Add tests

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Features

### Core Features
- ✅ **Multi-provider LLM support** - Azure OpenAI, OpenAI, Hyperbolic
- ✅ **Automatic inference logging** - All runs saved to database
- ✅ **Analytics dashboard** - Provider comparison, statistics, recent inferences
- ✅ **Batch and sampling inference** - Run multiple prompts or sample the same prompt
- ✅ **Retry logic** - Automatic retry with exponential backoff on transient errors
- ✅ **Prompt preprocessing** - Whitespace cleaning, truncation, PII removal
- ✅ **Error handling** - Comprehensive logging and graceful error handling
- ✅ **Database support** - SQLite (development) and PostgreSQL (production)

### Future Enhancements
- [ ] Multi-turn conversation support
- [ ] Request deduplication
- [ ] Cost tracking and estimation
- [ ] Vector similarity search
- [ ] Export functionality (CSV/JSON)
- [ ] Advanced filtering and search

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete user guide with usage instructions and troubleshooting
- **[Architecture](docs/ARCHITECTURE.md)** - System design, layer responsibilities, data flow
- **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** - Phased development roadmap with detailed steps
- **[API Documentation](docs/API.md)** - Programmatic API reference

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
