# Jetstream2 Deployment — Study Query LLM

This directory contains **infrastructure-only** assets for running the Panel app
as a 24/7 password-protected service on a Jetstream2 VM.

## Boundary Rules

1. **No edits to notebook or runtime env files.**
   The root `.env` and any notebook configs are owned by the local development
   environment.  Jetstream config lives exclusively in
   `deploy/jetstream/.env.jetstream`.

2. **No dependency on local GPU, Docker daemon, or local ports.**
   The Jetstream deployment consumes a pre-built container image referenced by
   `IMAGE_REF` in `.env.jetstream`.

3. **All runtime configuration comes from Jetstream env files.**
   Provider keys, database URLs, domain names, and image references are set in
   `.env.jetstream` on the VM — never hard-coded in these files.

4. **Isolated data stores.**
   `DATABASE_URL` on Jetstream must point to the Jetstream-local Postgres
   instance (or a dedicated remote DB), never to the local experiment database.

## Local development (from your PC)

To run tools against Jetstream Postgres while on your laptop, open an SSH tunnel and point the repo root `.env` at `127.0.0.1` — see **[LOCAL_DEV_TUNNEL.md](LOCAL_DEV_TUNNEL.md)**. That flow is separate from `.env.jetstream` on the VM.

## Directory Layout

```
deploy/jetstream/
  README.md                         # This file
  LOCAL_DEV_TUNNEL.md               # SSH tunnel from your PC to VM Postgres (127.0.0.1)
  .env.jetstream.example            # Template for VM runtime config
  docker-compose.jetstream.yml      # Compose stack (app + Postgres + Caddy)
  Caddyfile                         # HTTPS reverse proxy + basic auth
  systemd/
    study-query-llm.service         # systemd unit for compose stack
  RUNBOOK.md                        # Ops runbook (troubleshooting, backup, rollback)
  PROVISION.md                      # VM provisioning checklist (manual Phase 1)
  build-and-push.sh                 # Image build + push helper script
```

## Quick Start (on Jetstream VM)

```bash
# 1. Clone repo (or copy deploy/jetstream/ only)
git clone <repo-url> ~/app && cd ~/app/deploy/jetstream

# 2. Copy and fill env
cp .env.jetstream.example .env.jetstream

# 3. Pull image + start stack
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream up -d

# 4. Install Caddy and start reverse proxy
sudo cp Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy

# 5. (Optional) Install systemd unit for reboot persistence
sudo cp systemd/study-query-llm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now study-query-llm
```

## References

- [Jetstream2 Instance Flavors](https://docs.jetstream-cloud.org/general/vmsizes/)
- [Jetstream2 Web Server + HTTPS Tutorial](https://docs.jetstream-cloud.org/general/webserver/)
- [Jetstream2 Security Groups](https://docs.jetstream-cloud.org/ui/horizon/security_group)
- [Jetstream2 Firewalls](https://docs.jetstream-cloud.org/general/firewalls)
- [Jetstream2 ACCESS Credits](https://docs.jetstream-cloud.org/general/access/)
