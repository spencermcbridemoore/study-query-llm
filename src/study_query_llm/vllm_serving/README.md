# `vllm_serving` — self-hosted vLLM for the first-token logprob MCQ lane

Launch an OpenAI-compatible `vllm/vllm-openai` server — **locally in Docker** on
the operator's RTX 4090 (PROVEN) or on a **paid vast.ai cloud GPU** (UNPROVEN,
shape only) — and point the production first-token-logprob lane
(`mcq_logprob_basic`, `provider=local_llm`) at it.

The module is deliberately isolated: the **only** import from existing
`study_query_llm` code anywhere in it is
`from study_query_llm.providers.managers.protocol import ModelManager`
(in `manager.py`, used by `is_model_manager`). Everything else is stdlib,
third-party, or intra-module. No existing file was edited to add it.

- [What's in here](#whats-in-here)
- [Prerequisites (NOT in `setup.py`)](#prerequisites-not-in-setuppy)
- [LOCAL Docker recipe (verified)](#local-docker-recipe-verified)
- [CLI equivalent](#cli-equivalent)
- [TLS-interception workaround (local only)](#tls-interception-workaround-local-only)
- [The gate-(a)/(b) probe](#the-gate-ab-probe)
- [Thinking-off DRIFT NOTE (read this)](#thinking-off-drift-note-read-this)
- [Lane wiring (`LOCAL_LLM_ENDPOINT`)](#lane-wiring-local_llm_endpoint)
- [vast.ai walkthrough (UNPROVEN)](#vastai-walkthrough-unproven)
- [Cost & display-safety guarantees](#cost--display-safety-guarantees)

---

## What's in here

| File | Responsibility |
|------|----------------|
| `config.py` | `VLLMServeConfig` — the single source of serve flags (`to_serve_args`, `probe_extra_body`); `VLLM_IMAGE`, `INTERNAL_PORT`. |
| `vram.py` | `query_vram` / `assert_vram_headroom` — the **display-safety** guard. |
| `probe.py` | `wait_for_models_ready`, `probe_gate_a`, `probe_gate_b` — health + the gate probes. |
| `hf_download.py` | `snapshot_to_local` — host-side HF download for the local TLS workaround. |
| `vast_client.py` | `VastClient` seam + `VastCLIClient` (UNPROVEN `vastai` CLI shell-out). |
| `backends.py` | `LocalDockerBackend` (PROVEN) / `VastAIBackend` (UNPROVEN, shape only). |
| `manager.py` | `VLLMModelManager` — context manager + signals + atexit + idle timer; `is_model_manager`. |
| `__main__.py` | the CLI (`python -m study_query_llm.vllm_serving ...`). |

---

## Prerequisites (NOT in `setup.py`)

These are **module prerequisites only**. Per the isolation contract, the module
adds **zero edits to existing files**, so it does **not** declare any of these in
`setup.py`. Install them into your environment by hand. Every one of them is
reached through an injectable seam, so unit tests never need them installed.

| Package | Needed by | When |
|---------|-----------|------|
| `docker` (Python SDK) | `LocalDockerBackend` | local backend launch/teardown |
| Docker Engine + NVIDIA Container Toolkit | `LocalDockerBackend` | local `--gpus all` GPU passthrough |
| `pynvml` | `vram.query_vram` | local VRAM safety check (falls back to `nvidia-smi` if absent) |
| `huggingface_hub` | `hf_download.snapshot_to_local` | local TLS workaround (host-side download) |
| `requests` | `probe.*` | health wait + gate probes (any backend) |
| `paramiko` + `sshtunnel` | `VastAIBackend` (`served_via="ssh"`) | vast SSH local-forward |
| `vastai` CLI | `VastCLIClient` | vast offer search / create / show / destroy |

Suggested install (do this manually; nothing here touches `setup.py`):

```bash
pip install docker pynvml huggingface_hub requests paramiko sshtunnel vastai
```

```powershell
pip install docker pynvml huggingface_hub requests paramiko sshtunnel vastai
```

---

## LOCAL Docker recipe (verified)

The recipe below was run successfully on the local RTX 4090 (24 GB). It used
~10 GB of VRAM, gate (a) returned OpenAI-schema `top_logprobs`, and gate (b) for
`Qwen2.5-7B-Instruct-AWQ` returned a clean first-token top-5 of `A, B, D, C, E`.

> **Display safety:** `--gpu-memory-utilization 0.4` (≈10 GB of the 24 GB card)
> and `--enforce-eager` are deliberate. A too-greedy vLLM has crashed this
> operator's display. Keep both unless you are certain no display shares the GPU.

The serving machinery handles model loading via the [TLS workaround](#tls-interception-workaround-local-only)
on this network. The exact raw `docker run` that was verified (offline /
host-mounted form) is:

```bash
# 1. Download the snapshot on the HOST (TLS works here, not in the container):
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct-AWQ', \
  local_dir='$HOME/vllm_models/qwen2.5-7b-instruct-awq', \
  ignore_patterns=['*.pt','*.bin'])"

# 2. Run vLLM offline against the mounted snapshot:
docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$HOME/vllm_models/qwen2.5-7b-instruct-awq:/model:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai:latest \
    --model /model \
    --served-model-name qwen2.5-7b-instruct-awq \
    --quantization awq \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --enforce-eager \
    --max-num-seqs 2 \
    --port 8000
```

Key facts baked into the module so you don't have to remember them:

- **Image:** `vllm/vllm-openai:latest` (module constant `VLLM_IMAGE`).
- **`--gpus all`:** rendered by the backend as a Docker
  `DeviceRequest(count=-1, capabilities=[["gpu"]])`.
- **Port mapping:** vLLM **always** listens on internal port `8000`
  (`INTERNAL_PORT`); the backend maps `host:<port> -> container:8000`. The serve
  args therefore always end in `--port 8000` regardless of the host port.
- **Serve-flag order** (canonical, emitted by `VLLMServeConfig.to_serve_args`,
  optional flags omitted when unset):
  `--model <ref> --served-model-name <name> [--quantization <q>] [--dtype <d>]
  --gpu-memory-utilization 0.4 --max-model-len 2048 --enforce-eager
  --max-num-seqs 2 [--reasoning-parser <r>] [--chat-template <c>] --port 8000
  [*extra_serve_args]`.
- **VRAM cap default:** `gpu_memory_utilization=0.4`; the safety ceiling default
  is `DEFAULT_VRAM_CEILING_MB = 16000` (override with `vram_ceiling_mb=` /
  `--vram-ceiling-gb`).
- **`--enforce-eager`:** on by default (`enforce_eager=True`).

---

## CLI equivalent

Same recipe via the module CLI — this wraps the verified `docker run` in the
[VRAM guard](#cost--display-safety-guarantees), the [TLS workaround](#tls-interception-workaround-local-only),
and [guaranteed teardown](#cost--display-safety-guarantees):

```bash
# Print the current VRAM picture, launch, print the lane-wiring block, tear down:
python -m study_query_llm.vllm_serving --backend local \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq

# Same, but probe BOTH gates and HOLD the server until Ctrl-C:
python -m study_query_llm.vllm_serving --backend local \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --probe --serve
```

```powershell
python -m study_query_llm.vllm_serving --backend local `
  --model Qwen/Qwen2.5-7B-Instruct-AWQ `
  --quantization awq `
  --probe --serve
```

Equivalently, from Python:

```python
from study_query_llm.vllm_serving import (
    VLLMModelManager, LocalDockerBackend, VLLMServeConfig,
)

config = VLLMServeConfig(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    served_model_name="qwen2.5-7b-instruct-awq",
    quantization="awq",
    # gpu_memory_utilization=0.4, max_model_len=2048, max_num_seqs=2,
    # enforce_eager=True  -> all defaults
)
with VLLMModelManager(LocalDockerBackend(), config) as mgr:
    print(mgr.endpoint_url)   # http://localhost:8000/v1
```

Selected local CLI flags (defaults match the verified recipe):

| Flag | Default | Meaning |
|------|---------|---------|
| `--backend {local,vast}` | (required) | where to run vLLM |
| `--model <repo-or-path>` | (required) | HF repo id, or a local model dir (served offline) |
| `--served-name` | basename of `--model` | the `"model"` clients/lane send |
| `--quantization` | unset | e.g. `awq` |
| `--gpu-mem-util` | `0.4` | `--gpu-memory-utilization` |
| `--max-model-len` | `2048` | context cap |
| `--max-num-seqs` | `2` | max concurrent sequences |
| `--port` | `8000` | **host** port (internal is always 8000) |
| `--vram-ceiling-gb` | `16` | hard reservation cap (→ 16000 MB) |
| `--no-vram-check` | off | SKIP the display-safety guard (not recommended) |
| `--offline` / `--download` | auto | force the [TLS workaround](#tls-interception-workaround-local-only) on / off |
| `--gpu-index` | `0` | GPU to probe for VRAM |
| `--reasoning-parser` / `--chat-template` | unset | serve-layer thinking-off (see [DRIFT NOTE](#thinking-off-drift-note-read-this)) |
| `--no-thinking` | off | thinking-off **intent marker** (probe-only; see DRIFT NOTE) |
| `--probe` | off | run gate (a) then gate (b) after readiness |
| `--serve` | off | hold the server until Ctrl-C (else tear down after print/probe) |
| `--idle-timeout` | `1800` | seconds idle before auto-teardown |

---

## TLS-interception workaround (local only)

On the operator's **local Windows network**, an intercepting TLS proxy presents a
CA that Windows trusts but that is **not** trusted inside the container, so vLLM
in-container cannot reach huggingface.co (`CERTIFICATE_VERIFY_FAILED`).

**Workaround (verified):** download the snapshot on the HOST (TLS works there),
then mount it read-only and run vLLM fully offline:

```bash
# Host-side download (TLS works on the host):
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct-AWQ', \
  local_dir='$HOME/vllm_models/qwen2.5-7b-instruct-awq', \
  ignore_patterns=['*.pt','*.bin'])"

# Run offline against the mounted dir:
docker run --rm --gpus all -p 8000:8000 \
  -v "$HOME/vllm_models/qwen2.5-7b-instruct-awq:/model:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai:latest --model /model --served-model-name qwen2.5-7b-instruct-awq ...
```

The `LocalDockerBackend` does this automatically. Its `offline` toggle resolves
the model in three cases (`_resolve_model`):

1. `config.model` **is an existing local directory** → mount at `/model:ro`,
   serve `--model /model`, offline env on.
2. **offline mode** (`offline=True`, or `offline=None` auto-detected `True` on
   win32) → `snapshot_to_local(config.model, <models_dir>/<sanitized>)` on the
   host, mount `/model:ro`, set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`,
   serve `--model /model`.
3. otherwise → **in-container download**: serve `--model <repo>` with the host HF
   cache (`~/.cache/huggingface`) mounted read-write; no offline env.

> **This is a per-backend, local-only concern.** The vast.ai (Linux cloud)
> backend has no intercepting proxy, downloads the model **in-container**, and
> sets **no** offline env — it never uses this workaround.

Force the behaviour explicitly with `--offline` (force host-download + mount) or
`--download` (force in-container download).

---

## The gate-(a)/(b) probe

Two gates establish whether a freshly-launched server is usable for the
first-token-logprob lane. Both are reusable utilities in `probe.py` and parse
**plain JSON dicts** (`response.json()`), not `openai` SDK objects.

**Gate (a)** — does the server return the OpenAI logprob *schema* at all?
vLLM **guarantees** this; a failure means a broken server / unsupported request /
non-vLLM endpoint. `probe_gate_a` POSTs a minimal `chat/completions`
(`temperature=0, max_tokens=16, logprobs=true, top_logprobs=5`) and **PASSES iff**
`choices[0].logprobs.content[0].top_logprobs` is a non-empty list.

**Gate (b)** — does *this specific model* place an A–E option letter in the
**first decoded token's** `top_logprobs`? This is **per-model** and **MUST be
re-probed before any bulk run.** `probe_gate_b` follows the exact protocol:

- `system = "Reply with exactly one letter (A, B, C, D, or E). Do not explain."`
- `user = ` a 5-option (A–E) MCQ ending in `Answer:`
- `temperature=0, max_tokens=16, logprobs=true, top_logprobs=5`
- any `extra_body` is **merged** into the request body
- **PASSES iff** any token in `content[0].top_logprobs` normalizes to an A–E
  letter. `predicted_letter` = the highest-logprob A–E token; `top_tokens` = the
  full `(token, logprob)` list.

Token→letter normalization is **reimplemented locally**
(`_normalize_letter_token`): strip leading `▁` (`▁`) / `Ġ` (`Ġ`) /
whitespace marker glyphs, then `.strip().upper()`. (Mirrors
`mcq_logprob_basic._token_matches_letter`; it is **not** imported.)

Run both via the CLI:

```bash
python -m study_query_llm.vllm_serving --backend local \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --probe
```

Or directly:

```python
from study_query_llm.vllm_serving import probe_gate_a, probe_gate_b

a = probe_gate_a("http://localhost:8000/v1", "qwen2.5-7b-instruct-awq")
b = probe_gate_b("http://localhost:8000/v1", "qwen2.5-7b-instruct-awq")
print(a.passed, b.passed, b.predicted_letter, b.top_tokens)
```

> **Re-probe gate (b) per model.** vLLM guarantees gate (a) for any model, but
> whether a model emits an A–E letter as its first token is model-specific.
> Always pass gate (b) for the exact model before kicking off a bulk run.

---

## Thinking-off DRIFT NOTE (read this)

Reasoning models (`qwen3-*`) emit a `<think>` token first, which poisons the
first-token logprob probe — the first decoded token must be a **letter**, not
`<think>`.

There are two distinct levers, and **which one works depends on the lane:**

- **Serve layer** — `--reasoning-parser` / a no-think `--chat-template`, or simply
  picking a **non-thinking snapshot** repo (e.g. `...-Instruct-2507`).
- **Per-request `extra_body`** — `{"chat_template_kwargs": {"enable_thinking":
  false}}`.

**DRIFT NOTE (verified against the live codebase):** the production lane's method
runner (`mcq_logprob_basic.py`) has env hooks `MCQ_LOGPROB_MAX_TOKENS` and
`MCQ_LOGPROB_PROVIDER_PIN`, but **NO `MCQ_LOGPROB_EXTRA_BODY` hook** (that is an
unimplemented proposal). The runner sends **no per-request `extra_body`**, so a
per-request `extra_body` would **never reach the provider**.

> **Therefore, for the PRODUCTION lane, thinking-off MUST be applied at the
> SERVE layer** — `--reasoning-parser` / `--chat-template` / a non-thinking
> snapshot repo. A per-request `extra_body` is silently dropped by the
> production runner.

The standalone **gate-(b) probe CAN use `extra_body` directly** to verify a
reasoning model — that is exactly what `VLLMServeConfig.probe_extra_body()`
returns (`{"chat_template_kwargs": {"enable_thinking": false}}` when
`thinking_off=True`, else `None`), and the CLI passes it to `probe_gate_b` when
`--no-thinking` is set.

Consequently `VLLMServeConfig.thinking_off` (CLI `--no-thinking`) is only an
**intent marker**: on its own it is a **no-op at serve time**. If you set it
without `reasoning_parser`/`chat_template`, `VLLMServeConfig` logs a warning, and
the production lane will still see thinking output.

```bash
# CORRECT for the production lane (serve-layer thinking-off):
python -m study_query_llm.vllm_serving --backend local \
  --model Qwen/Qwen3-8B --served-name qwen3-8b \
  --reasoning-parser deepseek_r1 \
  --probe --no-thinking          # --no-thinking ALSO lets the probe verify via extra_body

# Or just use a non-thinking snapshot (no serve flags needed):
python -m study_query_llm.vllm_serving --backend local \
  --model Qwen/Qwen3-8B-Instruct-2507 --probe
```

> `--no-thinking` **alone** (no `--reasoning-parser`/`--chat-template`, no
> non-thinking snapshot) only affects the probe. The bulk lane will still emit
> `<think>`.

---

## Lane wiring (`LOCAL_LLM_ENDPOINT`)

After the manager's `start()` returns the URL (the CLI prints both blocks
automatically), wire the production lane to the self-hosted server. Run the lane
with `provider=local_llm`, and `model` = the **served-model-name**.

```bash
# --- bash / sh ---
export LOCAL_LLM_ENDPOINT=http://localhost:8000/v1
export LOCAL_LLM_API_KEY=not-needed
export MCQ_LOGPROB_PROVIDER_PIN=""
# then run the lane with provider=local_llm, model=<served-model-name>
```

```powershell
# --- PowerShell ---
$env:LOCAL_LLM_ENDPOINT="http://localhost:8000/v1"
$env:LOCAL_LLM_API_KEY="not-needed"
$env:MCQ_LOGPROB_PROVIDER_PIN=""
# then run the lane with provider=local_llm, model=<served-model-name>
```

> **`MCQ_LOGPROB_PROVIDER_PIN=""` is load-bearing.** An empty pin means **no**
> OpenRouter provider-routing field is sent — which is exactly right for a
> self-hosted vLLM server (it is not OpenRouter and would reject / ignore
> provider routing). A non-empty pin would send routing that a self-hosted
> server does not understand.

---

## vast.ai walkthrough (UNPROVEN)

> **UNPROVEN end-to-end.** `VastAIBackend` and `VastCLIClient` mirror the
> create/poll/health/teardown **shape** of `providers/managers/aci_tei.py`. They
> are **never run live by default**, the exact `vastai` CLI flag spellings and
> parsed-JSON field names are best-effort guesses, and they have **not** been
> validated against a live account. Treat this section as a structural map, not a
> tested procedure. The local backend is the proven path.

The vast Linux host has **no** TLS-interception proxy, so the model downloads
**in-container** and **no** offline env is set (unlike the local backend).

```bash
# PAID. Prompts for confirmation unless --yes (see the cost gate below):
python -m study_query_llm.vllm_serving --backend vast \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-name qwen2.5-7b-instruct \
  --serve
```

Lifecycle, end to end (`VastAIBackend.start` → `VastClient`):

1. **search** — `search_offers(gpu_name="RTX 4090", min_gpu_ram_mb=24000,
   num_gpus=1, max_dph=...)`; the backend picks the **cheapest** offer (min
   `dph`) and errors if none match.
2. **create** — `create_instance(offer_id, image=vllm/vllm-openai:latest,
   onstart=<vLLM launch on 0.0.0.0:8000>, disk_gb=40, label="vllm-serving")`.
   The `onstart` command uses the **same** `config.to_serve_args` as the local
   backend (so the lanes can't diverge), plus `--host 0.0.0.0`. **The instance
   id is recorded the instant `create_instance` returns — before any wait — so
   teardown can always find it.**
3. **wait** — `wait_until_running(instance_id)` polls until the instance reports
   a running status **and** publishes an SSH endpoint.
4. **access** — `served_via="ssh"` (default): open an SSH local-forward
   `localhost:<port> -> instance:8000` via the `ssh_forwarder` seam (default
   `sshtunnel.SSHTunnelForwarder`), base URL `http://localhost:<port>/v1`.
   `served_via="open_ports"`: talk straight to `http://<public_ip>:<port>/v1`
   (an `api_key` is **required** because the port is internet-reachable).
5. **destroy** — `stop()` closes the tunnel (best-effort) then
   `destroy_instance(instance_id)`, guarded so it fires at most once.

### PAID-provision confirmation gate

Before a single paid instance is created, the CLI prints a cost warning + the
search criteria and **requires confirmation**. It aborts unless `--yes` was
passed or the operator types exactly `yes` at the prompt. A non-interactive
stdin (EOF) without `--yes` aborts — a paid instance is never created without an
affirmative answer.

```text
======================================================================
WARNING: this will provision a PAID vast.ai GPU instance.
It bills per hour for as long as it runs. The CLI tears it down on
exit / Ctrl-C, but verify with 'vastai show instances' afterwards.
Search criteria:
  gpu_name        = 'RTX 4090'
  min_gpu_ram_mb  = 24000
  ...
======================================================================
Type 'yes' to provision a PAID vast.ai instance:
```

### Printed instance id + manual teardown fallback

On readiness the CLI prints the instance id and the **exact** manual teardown
command, so even if automatic teardown fails you can clean up by hand:

```text
vast.ai instance id: 1234567
manual teardown: vastai destroy instance 1234567
```

If `stop()`'s automatic `destroy_instance` raises, the backend logs at ERROR
that the instance **may still be running and billing**, along with the same
`vastai destroy instance <id>` command. **Always** verify afterward:

```bash
vastai show instances
vastai destroy instance <id>      # manual fallback
```

---

## Cost & display-safety guarantees

**Display safety (local backend):**

- Before launch, `LocalDockerBackend` probes free VRAM (`query_vram`, pynvml →
  `nvidia-smi` fallback) and calls `assert_vram_headroom`, which **refuses to
  launch** (raises `VRAMSafetyError`, container never started) if either:
  (1) the reservation `int(gpu_memory_utilization * total_mb)` exceeds the
  configured ceiling (`DEFAULT_VRAM_CEILING_MB = 16000`, override with
  `--vram-ceiling-gb`); or (2) currently-free VRAM is below
  `reservation + 512 MB` headroom.
- `gpu_memory_utilization` defaults to `0.4` (~10 GB on the 24 GB card) and
  `--enforce-eager` is **on by default** — both deliberate guards because a
  too-greedy vLLM has crashed this operator's display.
- The CLI prints the current VRAM status line **before** launching. The guard
  can be skipped with `--no-vram-check` (not recommended with an active display).

**Cost safety (vast backend):**

- The CLI's PAID-provision confirmation gate blocks creation until `--yes` or a
  typed `yes`.
- The instance id is recorded **immediately** on creation, before any wait, so
  teardown can always find a paid resource.
- `stop()` destroys the instance **at most once** and is idempotent; failures are
  logged loudly with the manual `vastai destroy instance <id>` fallback.

**Guaranteed teardown (both backends):** `VLLMModelManager` tears the server
down via four overlapping mechanisms, so a VRAM-hogging container or a
billing-by-the-minute instance is never orphaned:

1. **context manager** (`__enter__`/`__exit__`) — tears down on normal exit,
   exception, or `KeyboardInterrupt`;
2. **signal handlers** — `SIGINT` + `SIGTERM` (main thread only) tear down,
   restore the previous handler, then re-deliver the signal so default
   Ctrl-C / kill semantics still apply;
3. **atexit backstop** — a one-shot `atexit` hook catches any exit path the
   above missed;
4. **idle timer** — a daemon `threading.Timer` tears the server down after
   `idle_timeout_seconds` (default 1800) without a `ping()`.

`stop()` is lock-guarded and idempotent, so the overlap is safe.
