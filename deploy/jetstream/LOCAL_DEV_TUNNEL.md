# Local development: SSH tunnel to Jetstream Postgres

Use this when you run tools, tests, or the app **on your own machine** but want **`DATABASE_URL`** to point at the PostgreSQL instance that runs **inside** the Jetstream compose stack.

On the VM, Postgres listens on `127.0.0.1:5432` only. It is not exposed to the public internet, so your PC needs an **SSH local port forward** through the same user account you use to administer the VM (typically `exouser`).

## Quick start

1. In the **repo root** `.env`, set:
   - `JETSTREAM_SSH_HOST` — VM public IPv4 or Jetstream hostname (e.g. `*.projects.jetstream-cloud.org`).
   - `JETSTREAM_SSH_USER` — usually `exouser`.
   - `JETSTREAM_SSH_LOCAL_PORT` — local port (default `5433` avoids clashing with other Postgres on your PC).
   - `JETSTREAM_POSTGRES_*` and `JETSTREAM_DATABASE_URL` — credentials and URL for the Jetstream DB user (see `.env.example`).
   - Point `DATABASE_URL` at `JETSTREAM_DATABASE_URL` while using the tunnel (or set `DATABASE_URL` explicitly to the same `127.0.0.1:<local_port>` URL).

2. Start the tunnel and **leave it running** in a dedicated terminal:

   ```bash
   python scripts/start_jetstream_postgres_tunnel.py
   ```

3. Run your app or scripts in another terminal. The forward is:

   `127.0.0.1:<JETSTREAM_SSH_LOCAL_PORT>` → SSH → VM `127.0.0.1:5432`.

## Connection URL shape

Use query parameters for SSL, not a bogus host suffix:

- Correct: `postgresql://USER:PASS@127.0.0.1:5433/DBNAME?sslmode=prefer`
- Wrong: putting `sslmode` in the host part so Postgres treats the whole string as the database name.

See comments in `.env.example` next to `JETSTREAM_DATABASE_URL`.

## SSH keys (OpenSSH, not PuTTY `.ppk`)

The tunnel script passes your private key to **OpenSSH** (`ssh -i`) or to **Paramiko** via `sshtunnel`, depending on options. **PuTTY `.ppk` files are not supported.** In PuTTYgen, use **Conversions → Export OpenSSH key** and point `JETSTREAM_SSH_KEY` at that file.

Jetstream documents supported public-key types (RSA and Ed25519) in their UI docs, e.g. [SSH Keys in Horizon](https://docs.jetstream-cloud.org/ui/horizon/ssh_keys/).

### Passphrase behavior

- **`JETSTREAM_SSH_KEY` set, `JETSTREAM_SSH_KEY_PASSPHRASE` empty or unset** — the script uses `ssh -i` (unencrypted keys work non-interactively; encrypted keys may prompt unless `ssh-agent` has the key).
- **Both key and non-empty `JETSTREAM_SSH_KEY_PASSPHRASE`** — the script uses `sshtunnel` so the passphrase can be supplied from the environment (install optional extras: `pip install -e ".[jetstream-tunnel]"` — pins `paramiko>=3.4,<4` for compatibility).

Prefer not storing SSH passwords or key passphrases in `.env` when you can use `ssh-agent` and an unencrypted session key instead.

## Other auth options

- **Password SSH** — set `JETSTREAM_SSH_PASSWORD` (requires `sshtunnel`). If unset, the script may fall back to `JETSTREAM_POSTGRES_PASSWORD`, which is the **database** password; the Linux login password is often different.
- **Debug** — `JETSTREAM_SSH_DEBUG=1` for verbose SSH logs.

Full env reference: docstring in `scripts/start_jetstream_postgres_tunnel.py` and the `JETSTREAM_*` block in `.env.example`.

## Changing or rotating keys

The OpenStack “key pair” selected at **instance creation** is fixed in Horizon, but you can still add or replace keys on the VM by editing `~/.ssh/authorized_keys` for `exouser` when you have access. See Jetstream’s [Managing SSH keys from the CLI](https://docs.jetstream-cloud.org/ui/cli/managing-ssh-keys/).

## Port clash with local Docker Postgres

If both URLs use the same loopback host and port (including `127.0.0.1` vs `localhost`), only one process can listen. Either start the SSH tunnel first (so `127.0.0.1:5433` forwards to Jetstream), use a different tunnel local port (e.g. `5434`) and update `JETSTREAM_DATABASE_URL`, or stop the local Postgres container while testing Jetstream.

To compare row counts, manifests, and Azure `db-backups` blobs: `python scripts/verify_db_backup_inventory.py` (repo root).

## Related

- VM-side deployment and compose: [README.md](README.md) in this directory.
- Neon → Jetstream migration: [MIGRATION_FROM_NEON.md](MIGRATION_FROM_NEON.md).
- Back up local Postgres, then clone Jetstream into local Docker: [LOCAL_DB_CLONE_FROM_JETSTREAM.md](../../docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md).
