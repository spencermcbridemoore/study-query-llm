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
  setup_boot_services.sh            # Enable Docker + compose + Caddy on boot (run on VM)
  rotate_caddy_basic_auth.sh        # Set a new Caddy basic-auth password (run on VM)
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

# 5. Reboot persistence (writes systemd unit for *this* clone path, enables Docker + stack)
chmod +x setup_boot_services.sh
./setup_boot_services.sh
# Optional: also copy repo Caddyfile to /etc/caddy and enable Caddy
# ./setup_boot_services.sh --with-caddyfile

# Manual alternative (fixed path ~/app — edit unit if your clone differs):
# sudo cp systemd/study-query-llm.service /etc/systemd/system/
# sudo systemctl daemon-reload
# sudo systemctl enable --now study-query-llm
```

## Rotate Caddy basic-auth password (on the VM)

If you forgot the Panel login password or want to change it: **`Caddyfile` only stores a bcrypt hash** — you cannot recover the plaintext. Run (after `git pull`):

```bash
cd ~/app/deploy/jetstream   # your clone path
chmod +x rotate_caddy_basic_auth.sh
./rotate_caddy_basic_auth.sh --generate
```

`--generate` creates a long random password with Python’s `secrets`, **prints it once** in the terminal (save it to a password manager), then hashes it and updates Caddy. Run without flags to type a password interactively.

It runs `caddy hash-password`, then updates the **bcrypt line for user `admin`** (override with `CADDY_AUTH_USER`) in **`/etc/caddy/Caddyfile` and any file reached by Caddy `import`**, with a per-file `.bak.<timestamp>` backup. Finally it runs `caddy validate` and `systemctl reload caddy`. If your main file is only `import ...`, credentials can live in an imported snippet — the script follows those paths. Copy the printed `CADDY_AUTH_HASH` into `.env.jetstream` on the VM if you keep credentials there for documentation.

## References

- [Jetstream2 Instance Flavors](https://docs.jetstream-cloud.org/general/vmsizes/)
- [Jetstream2 Web Server + HTTPS Tutorial](https://docs.jetstream-cloud.org/general/webserver/)
- [Jetstream2 Security Groups](https://docs.jetstream-cloud.org/ui/horizon/security_group)
- [Jetstream2 Firewalls](https://docs.jetstream-cloud.org/general/firewalls)
- [Jetstream2 ACCESS Credits](https://docs.jetstream-cloud.org/general/access/)
