# Local-only Jetstream helpers

Files in this directory except this `README.md` are **gitignored** (see [.gitignore](../../.gitignore): `scratch/local/*` with `!scratch/local/README.md`).

## `jetstream-remote-build-and-restart.ps1`

Runs on **your Windows PC**. Over **SSH** it will on the Jetstream VM:

1. **`docker compose … stop app`** — Panel container only; **Postgres stays up**.
2. **`git pull`** under `$HOME/<RemoteRepoDir>` (default `app`).
3. **`docker build`** from that repo root into a **VM-local image tag** (default `study-query-llm:jetstream-vm-<timestamp>`).
4. **Rewrite only `IMAGE_REF=`** in `deploy/jetstream/.env.jetstream` via Python (UTF-8, timestamped `.bak.*`); it does **not** print the full env file.
5. **`docker compose … up -d --pull never --force-recreate app`** then **`curl /health`**.

### Example

From the repo root in **PowerShell**:

```powershell
.\scratch\local\jetstream-remote-build-and-restart.ps1
```

Override host, identity, or branch:

```powershell
.\scratch\local\jetstream-remote-build-and-restart.ps1 `
  -JetstreamHost "exouser@YOUR.FLOATING.IP" `
  -SshIdentity "$env:USERPROFILE\.ssh\your_key" `
  -GitRef "main" `
  -LocalImageTag "study-query-llm:jetstream-vm-manual1"
```

Parameter values must match the script’s allowed character set (`[a-zA-Z0-9._/@:-]+`) so they can be embedded safely in the remote bash script.

### Security

- Do not put **API keys** or **`.env.jetstream` contents** into this script.
- Keep your SSH private key **outside** the repo; pass `-SshIdentity` to its path.

### Related

- Registry-based redeploy (no VM build): [deploy/jetstream/redeploy_panel_from_origin.sh](../../deploy/jetstream/redeploy_panel_from_origin.sh) and [deploy/jetstream/RUNBOOK.md](../../deploy/jetstream/RUNBOOK.md).
