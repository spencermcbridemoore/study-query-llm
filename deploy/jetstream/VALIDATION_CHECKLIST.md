# Go-Live Validation Checklist

Run through these checks after deploying to the Jetstream VM.
All checks must pass before the service is considered live.

## Smoke Checks

- [ ] **HTTPS loads at domain**
  ```bash
  curl -I https://YOUR_DOMAIN
  # Expect: HTTP/2 401 (basic auth challenge)
  ```

- [ ] **Basic auth challenge appears**
  ```bash
  curl -u admin:YourPassword -I https://YOUR_DOMAIN
  # Expect: HTTP/2 200
  ```

- [ ] **Panel UI renders after login**
  Open `https://YOUR_DOMAIN` in a browser, enter credentials, confirm tabs
  (Inference, Analytics, Embeddings, Groups, Sweep Explorer) load.

- [ ] **Health endpoint reports OK behind proxy**
  ```bash
  curl -u admin:YourPassword https://YOUR_DOMAIN/health
  # Expect: {"status": "ok"}
  ```

- [ ] **Health endpoint responds locally (bypass Caddy)**
  ```bash
  curl -fsS http://127.0.0.1:5006/health
  # Expect: {"status": "ok"}
  ```

- [ ] **Run a simple inference**
  In the Panel Inference tab, enter a test prompt and confirm a response is
  returned with metadata (provider, tokens, latency).

## Security Checks

- [ ] **Direct Panel port NOT publicly exposed**
  From an external machine:
  ```bash
  curl http://YOUR_DOMAIN:5006/health
  # Expect: connection refused or timeout (NOT a 200 response)
  ```

- [ ] **SSH CIDR restriction verified**
  From an IP outside your admin CIDR:
  ```bash
  ssh exouser@YOUR_DOMAIN
  # Expect: connection refused or timeout
  ```

- [ ] **UFW status shows only expected ports**
  On the VM:
  ```bash
  sudo ufw status verbose
  # Expect: 22, 80, 443, 49528 only
  ```

- [ ] **No unprotected routes**
  ```bash
  curl -I https://YOUR_DOMAIN
  # Expect: 401 on ALL paths without credentials
  ```

## Operations Checks

- [ ] **Projected SU burn confirmed**
  Flavor `m3.small` = 2 SU/hr = 48 SU/day = ~1,460 SU/month.
  Confirm this fits within allocation budget.

- [ ] **Allocation renewal date noted**
  Check [ACCESS Allocations Portal](https://allocations.access-ci.org/) and
  set a calendar reminder 90 days before expiry.

- [ ] **Instance locked in Exosphere**
  Exosphere > Instance details > Actions > Lock. Confirm lock icon shows.

- [ ] **Compose stack restarts on reboot**
  ```bash
  sudo reboot
  # After reboot, SSH back in:
  docker compose -f docker-compose.jetstream.yml -p sqllm-jetstream ps
  curl -fsS http://127.0.0.1:5006/health
  # Expect: both containers running and healthy
  ```

- [ ] **Postgres backup cron scheduled**
  ```bash
  crontab -l | grep sqllm
  # Expect: daily pg_dump cron line (see RUNBOOK.md)
  ```

## Sign-Off

When all boxes are checked:

- Record the date and operator in this file.
- Take a VM snapshot (Exosphere > Image) named `sqllm-panel-golive-YYYY-MM-DD`.

**Go-live date:** _______________
**Operator:** _______________
