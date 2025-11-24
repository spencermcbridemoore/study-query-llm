# syntax=docker/dockerfile:1.6

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder

RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

COPY requirements.txt .
COPY setup.py .
COPY README.md .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY panel_app ./panel_app
COPY assets ./assets
COPY src ./src
COPY tests ./tests
COPY pytest.ini .
COPY test_*.py ./

RUN pip install --no-cache-dir -e ".[dev]"

ARG RUN_TESTS=0
RUN if [ "$RUN_TESTS" = "1" ]; then \
        pytest tests/test_e2e_verification.py || pytest tests/test_phase_1_1.py; \
    else \
        echo "Skipping build-time tests"; \
    fi

FROM base AS runtime

ENV PATH="$VENV_PATH/bin:$PATH" \
    PANEL_ADDRESS=0.0.0.0 \
    PANEL_PORT=5006 \
    DATABASE_URL=sqlite:////data/study_query_llm.db

COPY --from=builder "$VENV_PATH" "$VENV_PATH"
COPY --from=builder /app/panel_app ./panel_app
COPY --from=builder /app/assets ./assets
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./README.md
COPY --from=builder /app/setup.py ./setup.py

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && mkdir -p /data \
    && chown -R app:app /app /data

VOLUME ["/data"]

EXPOSE 5006

USER app

ENTRYPOINT ["python", "-m", "panel_app.app"]
CMD ["--address", "0.0.0.0", "--port", "5006"]

