# Docker Deployment Guide

This guide explains how to containerize and run Study Query LLM by using the
multi-stage `Dockerfile` and `docker-compose.yml` that ship with the repo.

## 1. Requirements

- Docker Desktop 4.0+ (or engine with BuildKit)
- Docker Compose v2 (bundled with recent Docker Desktop)
- API keys stored in an `.env` file at the repo root (re-uses the same keys as
  the local workflow)

## 2. Build the Image

```bash
# Build release image (uses `runtime` stage by default)
docker build -t study-query-llm:local .

# (Optional) Run the slow build-time tests inside the image
docker build -t study-query-llm:test --build-arg RUN_TESTS=1 .
```

### Useful Build Arguments

| Arg | Default | Description |
| --- | --- | --- |
| `RUN_TESTS` | `0` | When set to `1`, runs `pytest tests/test_e2e_verification.py` (falls back to `tests/test_phase_1_1.py` if data is missing). |

## 3. Run with `docker run`

The runtime stage exposes Panel on port `5006`, reads env vars, and persists the
SQLite database under `/data`.

```bash
docker run --rm \
  -p 5006:5006 \
  -v study_query_data:/data \
  --env-file .env \
  -e DATABASE_URL=sqlite:////data/study_query_llm.db \
  study-query-llm:local
```

- Override `PANEL_ADDRESS`, `PANEL_PORT`, or `PANEL_ALLOW_WS_ORIGINS` to match
  your ingress.
- Mount a host directory instead of the named volume if you want to inspect the
  SQLite file directly.

## 4. Run with Docker Compose

```bash
# Default stack (Panel + SQLite)
docker compose up --build

# Tear down
docker compose down
```

### Environment Variables

The compose file loads `.env` automatically and also exposes safe defaults. Key
variables include:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PANEL_ADDRESS` | Interface bound by the Panel server | `0.0.0.0` |
| `PANEL_PORT` | Panel server port (also controls host port binding) | `5006` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:////data/study_query_llm.db` |
| `AZURE_OPENAI_*`, `OPENAI_*`, `HYPERBOLIC_*` | Provider credentials | none |

All secrets should continue to live in `.env`; Compose will inject them into the
container.

### Persisted Volumes

| Volume | Mount Point | Purpose |
| --- | --- | --- |
| `study_query_data` | `/data` | SQLite database + any exports |
| `study_query_pgdata` | `/var/lib/postgresql/data` | Postgres storage when the optional DB is enabled |

### Optional Postgres Profile

The compose file ships with a `db` service that is hidden behind the `postgres`
profile. This is useful for testing against a production-like Postgres target.

```bash
# Start Panel + Postgres (uses defaults defined in compose file)
COMPOSE_PROFILES=postgres \
DATABASE_URL=postgresql+psycopg2://study:study@db:5432/study_query_llm \
docker compose up --build
```

- Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB` in `.env` if you
  need custom credentials.
- The `DATABASE_URL` must explicitly point at the `db` service hostname so the
  app talks to Postgres instead of SQLite.

### Health Check

The runtime automatically exposes `GET /health`, which the compose stack polls
via `curl` to mark the service as healthy. You can hit the same endpoint to
confirm the container is ready:

```bash
curl http://localhost:5006/health
# {"status": "ok"}
```

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `curl` health check fails | Run `docker logs study-query-llm-app` and confirm API keys + database configuration are valid. |
| Cannot reach Panel UI | Confirm port `5006` is free, then re-run `docker compose up --build`. |
| Postgres profile fails | Ensure `COMPOSE_PROFILES=postgres` and `DATABASE_URL` points to `db`. Check `docker compose logs db` for startup errors. |

## 6. Next Steps

- Integrate the image build into CI/CD (GitHub Actions or Azure DevOps) to push
  tagged releases.
- Publish the image to an internal registry once secrets are configured via
  container runtime (Azure Container Apps, ECS, etc.).

## 7. Validation & CI Recommendations

### Container Smoke Tests

Run these quick checks before publishing or tagging a release:

1. `docker run --rm study-query-llm:local --address 127.0.0.1 --port 5007`
   - Verify logs show `Panel application available at http://...`
2. `curl http://localhost:5007/health` (from host) returns `{"status": "ok"}`
3. `docker exec` into the container and run `python -m pytest tests/test_phase_1_1.py`
4. If API keys are available, run a real inference through the UI and confirm it
   lands in the database volume.

### CI Pipeline Sketch

```yaml
jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.ref_type == 'tag' }}
          tags: ghcr.io/your-org/study-query-llm:${{ github.ref_name }}
          build-args: RUN_TESTS=1
```

Add a second step to run `docker compose up -d` followed by the smoke tests
above to ensure the compose stack is functional before pushing images.

