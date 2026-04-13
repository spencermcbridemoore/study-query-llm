# Operations Runbook — Jetstream2 Panel Deployment

## Service Architecture

```
Internet --> Caddy (443/TLS + basic auth) --> Panel app (localhost:5006) --> Postgres (container)
```

---

## Health Checks

### Quick status

```bash
# Compose stack status
docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream ps

# Panel health endpoint (from VM)
curl -fsS http://127.0.0.1:5006/health

# Caddy status
sudo systemctl status caddy

# systemd unit status
sudo systemctl status study-query-llm
```

### Full external check

```bash
# From any machine (should prompt for basic auth)
curl -u admin:YourPassword https://YOUR_DOMAIN/health
```

---

## Common Issues

### Panel app unhealthy / container restarting

1. Check logs:
   ```bash
   docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream logs app --tail 100
   ```
2. Common causes:
   - Missing or invalid `DATABASE_URL` in `.env.jetstream`.
   - Postgres container not yet healthy (app starts before DB is ready — the
     `depends_on` condition should handle this, but check DB logs too).
   - Missing provider API keys (app may start but fail on first request).

### Caddy TLS certificate not provisioning

1. Verify the domain resolves to the VM floating IP:
   ```bash
   dig +short YOUR_DOMAIN
   ```
2. Check that ports 80 and 443 are open in both the security group and UFW:
   ```bash
   sudo ufw status
   ```
3. Check Caddy logs:
   ```bash
   sudo journalctl -u caddy --no-pager -n 50
   ```
4. If DNS is new, it may take up to a few hours to propagate. Caddy will retry
   automatically.

### HTTPS fails in the browser (`SSL_ERROR_INTERNAL_ERROR_ALERT`) or curl shows TLS alert

1. **Site block must not stay as the literal `YOUR_DOMAIN`.** Replace it with a
   hostname that has a **forward** DNS `A`/`AAAA` record to the floating IP, or
   Caddy cannot obtain a Let’s Encrypt certificate. Check:
   `dig +short your.name.projects.jetstream-cloud.org` from the public internet
   (not only reverse DNS / PTR).
2. **Floating IP only (no working public hostname yet):** use **`tls internal`**
   on the IP site block and set a global **`default_sni`** to that IP so clients
   that omit SNI for literal IPv4 (common on Windows) still complete TLS:
   ```caddyfile
   {
       default_sni YOUR.FLOATING.IP
   }
   YOUR.FLOATING.IP {
       tls internal
       basicauth * { ... }
       reverse_proxy localhost:5006
   }
   ```
   Browsers will show a one-time warning for the Caddy local CA; accept it to proceed.
3. After edits: `sudo caddy validate --config /etc/caddy/Caddyfile` then
   `sudo systemctl reload caddy` (or `restart` if reload is insufficient).

### WebSocket connection refused / Panel UI loads but widgets don't work

The Panel app requires WebSocket connections. If the browser console shows
WebSocket errors:

1. Check that `PANEL_ALLOW_WS_ORIGINS` in `.env.jetstream` lists **Bokeh-style**
   entries: **`host` or `host:port` only** (no `http://` / `https://` prefixes).
   Include the address users type in the browser, e.g. `example.projects.jetstream-cloud.org:443`
   and `:80`, or `YOUR.IP:443` / `YOUR.IP:80` / `YOUR.IP` when using HTTPS by IP.
2. Restart the app container after changing the env value:
   ```bash
   docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream up -d app
   ```

### SSH access lost

1. Use Exosphere Web Shell (port 49528) as emergency console access.
2. If UFW is blocking SSH, use the Web Shell to run:
   ```bash
   sudo ufw allow ssh
   ```

---

## Backup and Restore

### Postgres backup (manual)

```bash
docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream \
    exec db pg_dump -U sqllm study_query_jetstream \
    | gzip > ~/backups/sqllm-$(date -u +%Y%m%d-%H%M%S).sql.gz
```

### Scheduled backup (cron)

Add to `exouser` crontab (`crontab -e`):

```
0 4 * * * cd /home/exouser/app/deploy/jetstream && \
    docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream \
    exec -T db pg_dump -U sqllm study_query_jetstream \
    | gzip > /home/exouser/backups/sqllm-$(date -u +\%Y\%m\%d).sql.gz
```

Create the backups directory first: `mkdir -p ~/backups`

### Restore from backup

```bash
gunzip -c ~/backups/sqllm-YYYYMMDD.sql.gz | \
    docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream \
    exec -T db psql -U sqllm study_query_jetstream
```

---

## Deployment / Update

### One-shot: git pull + new image + restart (recommended)

From the VM, after a new image is published (CI or `./deploy/jetstream/build-and-push.sh --push` on a build machine), run:

```bash
cd ~/app/deploy/jetstream
chmod +x redeploy_panel_from_origin.sh
NEW_IMAGE_REF='ghcr.io/yourorg/study-query-llm@sha256:...' ./redeploy_panel_from_origin.sh
```

This script:

1. **`git fetch` + `git pull origin main`** under `~/app` (so compose files and helpers match the repo).
2. Optionally **rewrites `IMAGE_REF=`** in `.env.jetstream` when `NEW_IMAGE_REF` is set (reads existing file as UTF-8 with replacement for invalid bytes; rewrites UTF-8; timestamped `.bak.*` backup).
3. Runs **`docker compose pull app`** then **`up -d`** with **`--env-file .env.jetstream`** and project **`sqllm-jetstream`**.
4. **`curl`** checks **`http://127.0.0.1:5006/health`**.

Skip git when you only need the image roll: `SKIP_GIT=1 NEW_IMAGE_REF='...' ./redeploy_panel_from_origin.sh`.

See `redeploy_panel_from_origin.sh` header for all environment knobs (`GIT_REF`, `REPO`, `SKIP_COMPOSE_PULL`, etc.).

### Deploy a new image version (manual)

1. On your build machine (local or CI):
   ```bash
   cd /path/to/study-query-llm
   ./deploy/jetstream/build-and-push.sh --push
   ```
2. Note the `IMAGE_REF` printed at the end.
3. SSH into the Jetstream VM:
   ```bash
   cd ~/app/deploy/jetstream
   # Edit .env.jetstream — update IMAGE_REF to the new value
   docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream pull app
   docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream up -d app
   ```
4. Verify health:
   ```bash
   curl -fsS http://127.0.0.1:5006/health
   ```

### Rollback to previous image

1. Edit `.env.jetstream` — set `IMAGE_REF` back to the previous tag/digest (or restore a `.bak.*` from `redeploy_panel_from_origin.sh`).
2. Pull and restart:
   ```bash
   cd ~/app/deploy/jetstream
   docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream pull app
   docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream up -d app
   ```

---

## Allocation Monitoring

Jetstream2 policies to be aware of:

- **Overdraw**: If your allocation is overdrawn, instances are shelved at
  10 days and **deleted at 30 days**.
- **Expiration**: Same timeline applies when the allocation expires without
  renewal.
- **Shelved instance retention**: Instances shelved for over 1 year may be
  deleted by admins.
- **Floating IP retention**: IPs on shelved instances for 90+ days can be
  reclaimed.

Monitor your credit balance in the
[ACCESS Allocations Portal](https://allocations.access-ci.org/).

### SU burn rates (for reference)

| Flavor   | SU/hr | Daily | Monthly (~30d) | Annual  |
|----------|-------|-------|----------------|---------|
| m3.tiny  | 1     | 24    | 720            | 8,760   |
| m3.small | 2     | 48    | 1,440          | 17,520  |

---

## Running the Cached-Job Supervisor (one-engine benchmarks)

For high-worker-count, single-engine runs (e.g. scaling benchmarks), use the cached-job supervisor instead of the engine supervisor. It uses a single DB client for claim/complete and distributes work via in-process queues.

```bash
# From project root, with DATABASE_URL set
python scripts/run_cached_job_supervisor.py \
  --request-id N \
  --worker-count 32 \
  --engine "Qwen/Qwen3-Embedding-0.6B" \
  --provider-label local_docker_tei_shared \
  --tei-endpoint http://localhost:8080/v1
```

Workers use the DB only for read-only embedding cache (L3/L2). Ensure TEI is running and reachable at `--tei-endpoint` before starting.

---

## Emergency Procedures

### Full stack restart

```bash
sudo systemctl restart study-query-llm
sudo systemctl restart caddy
```

### Nuclear reset (destroy and recreate containers)

```bash
cd ~/app/deploy/jetstream
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream down
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream up -d
```

Postgres data survives because it is on a named volume (`pg_data`).

### VM snapshot for disaster recovery

Take a snapshot from Exosphere before any risky operation:

Exosphere > Instance details > Actions > **Image**
