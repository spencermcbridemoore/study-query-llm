# Terraform — Jetstream2 Infrastructure (Phase 2)

This directory is a **placeholder** for Phase 2 of the Jetstream deployment plan.

## When to use Terraform

Add Terraform here after the manual deployment (Phase 1) has been running
stably for at least 2-4 weeks.  Terraform codifies what you did manually in
`PROVISION.md` so that the VM, network, security groups, and floating IP can be
recreated reproducibly.

## Planned scope

```
terraform/
  main.tf              # OpenStack provider + instance + network
  variables.tf         # Parameterised inputs (flavor, image, CIDR, etc.)
  outputs.tf           # Floating IP, hostname, etc.
  terraform.tfvars     # Actual values (gitignored)
```

## Prerequisites

- Terraform CLI installed.
- OpenStack CLI credentials sourced (`openrc.sh` from Horizon > API Access).
- Jetstream2 OpenStack provider configuration.

## Getting started

See the [Jetstream2 Terraform docs](https://docs.jetstream-cloud.org/general/terraform/)
for provider setup and examples.

## What NOT to put here

- Application code or dependencies.
- Secrets or API keys (use `terraform.tfvars` which is gitignored).
- Anything that changes the app container image — that belongs in
  `build-and-push.sh` and `.env.jetstream`.
