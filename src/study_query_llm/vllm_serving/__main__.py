"""
vLLM serving CLI -- launch a local-Docker or paid-vast.ai vLLM server, print the
lane-wiring environment block, optionally probe it, and tear it down safely.

This is the operator-facing front door to the ``vllm_serving`` module.  It wires
the pieces together and adds three operator-safety affordances on top of the
library:

1. **Paid-provision confirmation gate (vast).**  Provisioning a cloud GPU costs
   real money every minute it runs.  Before a single paid instance is created the
   CLI prints a cost warning + the search criteria and REQUIRES confirmation: it
   aborts unless ``--yes`` was passed or the operator types ``yes`` at the
   interactive prompt.  (The local backend has no such cost, so no gate.)

2. **Guaranteed teardown.**  The server always runs inside the
   :class:`~study_query_llm.vllm_serving.manager.VLLMModelManager` context
   manager, so the container / paid instance is torn down on normal exit,
   exception, or Ctrl-C.

3. **Lane wiring you can paste.**  On readiness it prints the exact
   ``LOCAL_LLM_*`` / ``MCQ_LOGPROB_PROVIDER_PIN`` export block in BOTH bash and
   PowerShell form so the production logprob lane
   (``mcq_logprob_basic``, ``provider=local_llm``) can be pointed at the
   self-hosted server with a copy-paste.

Thinking-off note (verified against the live codebase -- IMPORTANT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The production method runner (``mcq_logprob_basic``) honours
``MCQ_LOGPROB_MAX_TOKENS`` and ``MCQ_LOGPROB_PROVIDER_PIN`` but has **no**
``MCQ_LOGPROB_EXTRA_BODY`` hook, so a per-request ``extra_body`` never reaches the
provider.  For the PRODUCTION lane, thinking-off (for ``qwen3-*`` reasoning
models) MUST therefore be configured at the SERVE layer -- pass
``--reasoning-parser`` / ``--chat-template`` (or use a non-thinking snapshot
repo).  ``--no-thinking`` only sets the intent marker; on its own it is a no-op at
serve time but DOES let the standalone gate-(b) ``--probe`` disable thinking via
``probe_extra_body()``.

Usage examples::

    # Local Docker, just print the lane wiring and tear down:
    python -m study_query_llm.vllm_serving --backend local \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq

    # Local, probe both gates and hold the server until Ctrl-C:
    python -m study_query_llm.vllm_serving --backend local \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --probe --serve

    # vast.ai (PAID -- prompts for confirmation unless --yes):
    python -m study_query_llm.vllm_serving --backend vast \\
        --model Qwen/Qwen2.5-7B-Instruct --serve
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from .backends import LocalDockerBackend, VastAIBackend, VLLMBackend
from .config import VLLMServeConfig
from .manager import VLLMModelManager
from .probe import GateResult, probe_gate_a, probe_gate_b
from .vram import DEFAULT_VRAM_CEILING_MB, VRAMSafetyError, query_vram

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _derive_served_name(model: str) -> str:
    """Derive a default served-model-name from ``--model``.

    Uses the basename of the repo id / path (the part after the last ``/`` or
    ``\\``) so ``Qwen/Qwen2.5-7B-Instruct-AWQ`` -> ``Qwen2.5-7B-Instruct-AWQ``
    and ``C:\\models\\qwen`` -> ``qwen``.  The served-model-name is what clients
    (and the lane) put in the request ``"model"`` field, so it is kept
    human-meaningful rather than aggressively sanitized.
    """
    cleaned = model.strip().rstrip("/\\")
    base = cleaned.replace("\\", "/").rsplit("/", 1)[-1]
    return base or cleaned or "model"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m study_query_llm.vllm_serving",
        description=(
            "Launch a vLLM OpenAI-compatible server (local Docker or paid "
            "vast.ai), print the lane-wiring env block, optionally probe it, and "
            "tear it down safely."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--backend",
        choices=("local", "vast"),
        required=True,
        help="Where to run vLLM: 'local' (Docker on this GPU) or 'vast' (PAID cloud).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF repo id (downloaded) or a local model directory path (served offline).",
    )
    parser.add_argument(
        "--served-name",
        default=None,
        help=(
            "Name clients pass as the request 'model' field (== the lane's model). "
            "Defaults to the sanitized basename of --model."
        ),
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="e.g. 'awq' -> adds --quantization awq (omit to let vLLM auto-detect).",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.4,
        help="vLLM --gpu-memory-utilization (fraction of TOTAL VRAM). Default 0.4.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM --max-model-len (context window cap). Default 2048.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        help="vLLM --max-num-seqs (max concurrent sequences). Default 2.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HOST port to expose (vLLM always listens on internal 8000). Default 8000.",
    )
    parser.add_argument(
        "--reasoning-parser",
        default=None,
        help="Serve-layer thinking-off: vLLM --reasoning-parser <v> (reaches the production lane).",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Serve-layer thinking-off: vLLM --chat-template <v> (reaches the production lane).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Set the thinking_off intent marker. NOTE: a no-op at the serve layer "
            "unless paired with --reasoning-parser/--chat-template; it only lets "
            "--probe disable thinking per-request (the production lane sends no "
            "extra_body)."
        ),
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="After readiness, run gate (a) then gate (b) and print PASS/FAIL + top tokens.",
    )
    parser.add_argument(
        "--vram-ceiling-gb",
        type=float,
        default=16.0,
        help="Local display-safety: hard cap on the reservation (GB). Default 16 (-> 16000 MB).",
    )
    parser.add_argument(
        "--no-vram-check",
        action="store_true",
        help="Local only: SKIP the display-safety VRAM guard (NOT recommended with an active display).",
    )

    offline_group = parser.add_mutually_exclusive_group()
    offline_group.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=None,
        help="Local TLS workaround: force host-download + offline mount.",
    )
    offline_group.add_argument(
        "--download",
        dest="offline",
        action="store_false",
        help="Local: force in-container download (no offline mount).",
    )

    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=1800,
        help="Seconds of inactivity before auto-teardown (frees VRAM / stops billing). Default 1800.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Local: GPU index to probe for the VRAM guard. Default 0.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the PAID-PROVISION confirmation prompt for the vast backend.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Hold the server up until Ctrl-C; otherwise tear down after printing/probing.",
    )

    return parser


def _build_config(args: argparse.Namespace, served_name: str) -> VLLMServeConfig:
    """Translate parsed CLI args into a frozen :class:`VLLMServeConfig`."""
    return VLLMServeConfig(
        model=args.model,
        served_model_name=served_name,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        # enforce_eager defaults True (display-safety); not exposed -- keep it on.
        reasoning_parser=args.reasoning_parser,
        chat_template=args.chat_template,
        thinking_off=args.no_thinking,
        port=args.port,
    )


def _build_backend(args: argparse.Namespace) -> VLLMBackend:
    """Construct the requested backend from CLI args."""
    if args.backend == "local":
        return LocalDockerBackend(
            port=args.port,
            offline=args.offline,
            enable_vram_check=not args.no_vram_check,
            vram_ceiling_mb=int(args.vram_ceiling_gb * 1000),
            gpu_index=args.gpu_index,
        )
    # vast
    return VastAIBackend(port=args.port)


# --------------------------------------------------------------------------- #
# Paid-provision confirmation gate (vast only)
# --------------------------------------------------------------------------- #
def _confirm_paid_provision(args: argparse.Namespace, backend: VLLMBackend) -> bool:
    """Cost-confirmation gate for the (paid) vast backend.

    Prints a clear cost warning + the search criteria, then REQUIRES explicit
    confirmation: returns ``True`` only if ``--yes`` was passed or the operator
    types exactly ``yes`` at the prompt.  Returns ``False`` (abort) otherwise.

    A non-interactive stdin (EOF) without ``--yes`` aborts -- we never provision a
    paid instance without an affirmative answer.
    """
    print("=" * 70)
    print("WARNING: this will provision a PAID vast.ai GPU instance.")
    print("It bills per hour for as long as it runs. The CLI tears it down on")
    print("exit / Ctrl-C, but verify with 'vastai show instances' afterwards.")
    print("Search criteria:")
    print(f"  gpu_name        = {backend.gpu_name!r}")
    print(f"  min_gpu_ram_mb  = {backend.min_gpu_ram_mb}")
    print(f"  num_gpus        = {backend.num_gpus}")
    print(f"  max_dph         = {backend.max_dph if backend.max_dph is not None else 'unset'}")
    print(f"  image           = {backend.image}")
    print(f"  served_via      = {backend.served_via}")
    print("=" * 70)

    if args.yes:
        print("--yes supplied: skipping interactive confirmation.")
        return True

    try:
        answer = input("Type 'yes' to provision a PAID vast.ai instance: ")
    except EOFError:
        print("No interactive input available and --yes not set; aborting.")
        return False

    if answer.strip().lower() == "yes":
        return True

    print("Confirmation not received ('yes' required); aborting -- nothing provisioned.")
    return False


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _print_vram_status(gpu_index: int) -> None:
    """Print the current VRAM status line before a local launch (best-effort)."""
    try:
        status = query_vram(gpu_index)
    except Exception as exc:  # noqa: BLE001 - informational only; launch guard runs separately
        print(f"VRAM status: unavailable ({type(exc).__name__}: {exc})")
        return
    print(
        f"VRAM status: GPU {status.index} ({status.name}) -- "
        f"total {status.total_mb} MB, free {status.free_mb} MB, used {status.used_mb} MB"
    )


def _print_lane_wiring(url: str, served_model_name: str) -> None:
    """Print the exact bash AND PowerShell lane-wiring export blocks."""
    print()
    print(f"vLLM server READY at: {url}")
    print()
    print("# --- bash / sh ---")
    print(f"export LOCAL_LLM_ENDPOINT={url}")
    print("export LOCAL_LLM_API_KEY=not-needed")
    print('export MCQ_LOGPROB_PROVIDER_PIN=""')
    print(f"# then run the lane with provider=local_llm, model={served_model_name}")
    print()
    print("# --- PowerShell ---")
    print(f'$env:LOCAL_LLM_ENDPOINT="{url}"')
    print('$env:LOCAL_LLM_API_KEY="not-needed"')
    print('$env:MCQ_LOGPROB_PROVIDER_PIN=""')
    print(f"# then run the lane with provider=local_llm, model={served_model_name}")
    print()


def _print_gate_result(result: GateResult) -> None:
    """Pretty-print a single gate probe result."""
    verdict = "PASS" if result.passed else "FAIL"
    print(f"  gate ({result.gate}): {verdict} -- {result.detail}")
    if result.top_tokens:
        rendered = ", ".join(
            f"{tok!r}={lp:.4f}" if lp is not None else f"{tok!r}=<none>"
            for tok, lp in result.top_tokens
        )
        print(f"    first-token top tokens: {rendered}")
    if result.predicted_letter is not None:
        print(f"    predicted letter: {result.predicted_letter}")


def _run_probes(url: str, config: VLLMServeConfig) -> bool:
    """Run gate (a) then gate (b); return ``True`` iff both pass."""
    print()
    print("Running readiness probes ...")
    gate_a = probe_gate_a(url, config.served_model_name)
    _print_gate_result(gate_a)

    # Pass the probe-only extra_body (thinking-off) when --no-thinking was set.
    extra_body = config.probe_extra_body()
    gate_b = probe_gate_b(url, config.served_model_name, extra_body=extra_body)
    _print_gate_result(gate_b)
    print()
    return gate_a.passed and gate_b.passed


def _block_until_interrupt() -> None:
    """Hold the process until Ctrl-C / termination, then return.

    The manager's signal handlers (or this loop's ``KeyboardInterrupt`` catch)
    drive teardown; we just keep the main thread alive so the server stays up.
    ``signal.pause`` does not exist on Windows, so fall back to a sleep loop.
    """
    print("Serving. Press Ctrl-C to stop and tear down.")
    try:
        pause = getattr(signal, "pause", None)
        if pause is not None:
            while True:
                pause()
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        print("\nInterrupted -- tearing down ...")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    served_name = args.served_name or _derive_served_name(args.model)
    config = _build_config(args, served_name)
    backend = _build_backend(args)

    if args.backend == "local":
        # Show the current VRAM picture before launch (the backend separately
        # enforces the display-safety guard unless --no-vram-check).
        _print_vram_status(args.gpu_index)
    else:
        # PAID-PROVISION confirmation gate -- abort unless --yes or "yes".
        if not _confirm_paid_provision(args, backend):
            return 1

    manager = VLLMModelManager(
        backend,
        config,
        idle_timeout_seconds=args.idle_timeout,
    )

    try:
        # Context manager => guaranteed teardown on normal exit, exception, Ctrl-C.
        with manager as mgr:
            url = mgr.endpoint_url or ""

            _print_lane_wiring(url, config.served_model_name)

            # For vast: surface the instance id + the manual teardown command.
            if backend.is_remote():
                instance_id = getattr(backend, "_instance_id", None)
                if instance_id:
                    print(f"vast.ai instance id: {instance_id}")
                    print(f"manual teardown: vastai destroy instance {instance_id}")
                    print()

            probes_ok = True
            if args.probe:
                probes_ok = _run_probes(url, config)

            if args.serve:
                _block_until_interrupt()
                # Falling out of the `with` block tears the server down.
                return 0

            # Not holding the server: teardown happens on `with` exit below.
            if args.probe and not probes_ok:
                print("One or more probes FAILED.")
                return 2
            return 0
    except VRAMSafetyError as exc:
        # The display-safety guard refused the launch; nothing was started.
        print(f"VRAM safety check refused the launch: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        # Teardown already ran via the context manager / signal handler.
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean nonzero exit
        logger.exception("vLLM serving failed: %s", exc)
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
