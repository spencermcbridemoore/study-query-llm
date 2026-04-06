# Study Query LLM

A Panel-based web application for running LLM inference experiments across multiple providers and analyzing results in a PostgreSQL-backed v2 schema.

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

See [docs/living/ARCHITECTURE_CURRENT.md](docs/living/ARCHITECTURE_CURRENT.md) for current architecture and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for historical context.

## Implementation Status

**Current status references:**

- Current capabilities and defaults: [docs/living/CURRENT_STATE.md](docs/living/CURRENT_STATE.md)
- Documentation parity evidence: [docs/review/DOC_PARITY_LEDGER.md](docs/review/DOC_PARITY_LEDGER.md)
- Historical phased implementation narrative: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

### Current Feature Snapshot
- Multi-provider chat/embedding workflows (including Azure and OpenAI-compatible endpoints)
- Automatic inference logging to database
- Analytics dashboard with provider comparison
- Batch and sampling inference support
- Retry logic with exponential backoff
- Prompt preprocessing
- Comprehensive error handling and logging
- Sweep/job orchestration CLI and runbook workflows

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL (v2 default; SQLite legacy compatibility still exists)
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

   **Miniconda / conda (optional):** create or refresh the dedicated env **`study-query-llm`** from the repo root:

   ```bash
   conda env create -f environment.yml          # first time only
   conda env update -f environment.yml -n study-query-llm --prune   # refresh existing env
   conda activate study-query-llm
   ```

   The file `environment.yml` installs Python 3.11, scientific stack, Panel/plotly, pytest, and `pip install -e .` for this package.

3. **Configure environment**
   
   Create a `.env` file in the project root:
   ```bash
   # Database (v2 recommended)
   DATABASE_URL=postgresql://user:password@localhost:5432/study_query_llm_v2
   
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

4. **Initialize database (v2)**
```bash
python -c "from study_query_llm.db.connection_v2 import DatabaseConnectionV2; import os; db = DatabaseConnectionV2(os.getenv('DATABASE_URL')); db.init_db()"
```

   For legacy v1-only initialization, see `docs/history/USER_GUIDE_V1_LEGACY.md`.

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

Use CI and local `pytest` output as the canonical test-status source.

### Adding a New LLM Provider

1. Create `src/study_query_llm/providers/your_provider.py`
2. Implement the `BaseLLMProvider` interface
3. Add to `ProviderFactory`
4. Update configuration
5. Add tests

See [docs/living/ARCHITECTURE_CURRENT.md](docs/living/ARCHITECTURE_CURRENT.md) for current details.

## Features

### Core Features
- ✅ **Multi-provider LLM support** - Azure + OpenAI-compatible provider flows
- ✅ **Automatic inference logging** - All runs saved to database
- ✅ **Analytics dashboard** - Provider comparison, statistics, recent inferences
- ✅ **Batch and sampling inference** - Run multiple prompts or sample the same prompt
- ✅ **Retry logic** - Automatic retry with exponential backoff on transient errors
- ✅ **Prompt preprocessing** - Whitespace cleaning, truncation, PII removal
- ✅ **Error handling** - Comprehensive logging and graceful error handling
- ✅ **Database support** - v2 PostgreSQL-first with legacy compatibility paths retained

### Future Enhancements
- [ ] Multi-turn conversation support
- [ ] Request deduplication
- [ ] Cost tracking and estimation
- [ ] Vector similarity search
- [ ] Export functionality (CSV/JSON)
- [ ] Advanced filtering and search

## Documentation

- **[Docs Index](docs/README.md)** - Taxonomy and navigation (`living`, `runbooks`, `history`, `deprecated`)
- **[Current State](docs/living/CURRENT_STATE.md)** - Authoritative "what exists/works now"
- **[Current Architecture](docs/living/ARCHITECTURE_CURRENT.md)** - v2-first architecture reference
- **[Current API Quick Reference](docs/living/API_CURRENT.md)** - Current factory/service/repository entrypoints
- **[User Guide](docs/USER_GUIDE.md)** - v2-first user workflow

## Contributing

This project follows a bottom-up, incremental development approach. Each component is designed to be:
- Testable in isolation
- Independent of layers above it
- Composable with other components

See [docs/living/CURRENT_STATE.md](docs/living/CURRENT_STATE.md) for current behavior and [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for historical roadmap context.

## License

[Add your license here]

## Contact

[Add your contact information here]
