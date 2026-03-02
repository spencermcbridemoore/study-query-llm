# Jetstream2 VM Provisioning Checklist (Manual Phase 1)

Follow these steps once to create the Jetstream2 VM and network configuration.
After the deployment is stable, codify this in Terraform (see Phase 2 in the
plan).

## Prerequisites

- Active ACCESS allocation with credits exchanged for **Jetstream2 (CPU)**.
- ACCESS CI login credentials.

## 1. Create the Instance

1. Log in to [Exosphere](https://jetstream2.exosphere.app/exosphere/home).
2. Select your allocation.
3. Click **Create > Instance**.
4. Choose image: **Ubuntu 24.04** (featured image).
5. Name: `sqllm-panel` (or similar descriptive name).
6. Flavor: **m3.small** (2 vCPU, 6 GB RAM, 20 GB disk).
   - Decision gate: downsize to `m3.tiny` after 2 weeks if CPU utilization
     stays under 10%.
7. Root disk size: leave default (20 GB).
8. Enable Web Desktop: **No** (not needed for headless server).
9. SSH key: add your public key.
10. Advanced options: confirm "Install operating system updates?" is **Yes**.
    Disable Guacamole.
11. Click **Create** and wait for status **Ready**.

## 2. Note the Hostname and IP

After the instance is ready, note:

- **Floating IP**: e.g. `149.165.xxx.xxx`
- **DNS hostname**: `sqllm-panel.xxx000000.projects.jetstream-cloud.org`

The DNS hostname is automatically created via Designate.

## 3. Configure Security Groups

In [Horizon](https://js2.jetstream-cloud.org/) > Network > Security Groups:

### Create group: `sqllm-ssh`
| Direction | Protocol | Port  | Remote CIDR          | Description            |
|-----------|----------|-------|----------------------|------------------------|
| Ingress   | TCP      | 22    | YOUR_ADMIN_CIDR/32   | SSH from admin IP only |

### Create group: `sqllm-web`
| Direction | Protocol | Port  | Remote CIDR | Description          |
|-----------|----------|-------|-------------|----------------------|
| Ingress   | TCP      | 80    | 0.0.0.0/0   | HTTP (Caddy redirect)|
| Ingress   | TCP      | 443   | 0.0.0.0/0   | HTTPS (Caddy TLS)    |

### Create group: `sqllm-icmp`
| Direction | Protocol | Port  | Remote CIDR | Description          |
|-----------|----------|-------|-------------|----------------------|
| Ingress   | ICMP     | All   | 0.0.0.0/0   | Ping for health      |

Attach all three groups to the `sqllm-panel` instance.

## 4. Apply Host Firewall (UFW)

SSH into the instance and run:

```bash
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 49528/tcp          # Exosphere Web Shell (keep for emergency access)
sudo ufw enable
sudo ufw status verbose
```

## 5. Install Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker exouser
# Log out and back in for group membership to take effect.
```

## 6. Install Caddy

Follow [Caddy install docs for Ubuntu](https://caddyserver.com/docs/install#debian-ubuntu-raspbian):

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

## 7. Lock the Instance

In Exosphere > Instance details > Actions > **Lock**.

This prevents accidental deletion or resizing from any Jetstream2 interface.

## 8. Create a Snapshot

After Docker + Caddy are installed and basic networking is verified, take a
snapshot/image for fast recovery:

Exosphere > Instance details > Actions > **Image** > name it
`sqllm-panel-base-YYYY-MM-DD`.

## Cost Check

| Flavor    | SU/hr | Daily  | Monthly  | Annual    |
|-----------|-------|--------|----------|-----------|
| m3.tiny   | 1     | 24     | ~730     | 8,760     |
| m3.small  | 2     | 48     | ~1,460   | 17,520    |

With 10,000 credits: `m3.tiny` runs ~13.7 months; `m3.small` ~6.8 months.
With 200,000 credits: `m3.small` runs ~11.4 years (allocation renewal is the
real constraint, not credits).
