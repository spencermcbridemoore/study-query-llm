"""
vLLM serve configuration -- the single source of truth for how a vLLM
``vllm/vllm-openai`` container is launched.

A :class:`VLLMServeConfig` is a frozen, backend-agnostic description of *what*
to serve and *how* the OpenAI-compatible server should behave.  Both backends
(local Docker, remote vast.ai) translate the same config into the identical
``vllm serve`` argument vector via :meth:`VLLMServeConfig.to_serve_args`, so the
two lanes can never silently diverge in their serve flags.

Canonical defaults come from the recipe that was verified live on an RTX 4090
(24 GB): ``--gpu-memory-utilization 0.4 --max-model-len 2048 --enforce-eager
--max-num-seqs 2``.  These are deliberately conservative -- ``enforce_eager`` and
the 0.4 utilization cap exist to protect the operator's display (a too-greedy
vLLM has crashed displays here); raise them knowingly, not by accident.

Thinking-off lanes
~~~~~~~~~~~~~~~~~~
Reasoning models (``qwen3-*``) emit a ``<think>`` token first, which poisons the
first-token logprob probe.  There are two distinct levers, and *which one works
depends on the lane*:

* **Serve layer** -- ``--reasoning-parser`` / ``--chat-template`` (or simply
  picking a non-thinking snapshot repo).  This is the ONLY lever that reaches the
  production method runner, because that runner sends no per-request
  ``extra_body`` (it has ``MCQ_LOGPROB_MAX_TOKENS`` and
  ``MCQ_LOGPROB_PROVIDER_PIN`` hooks but no ``MCQ_LOGPROB_EXTRA_BODY`` hook -- a
  per-request ``extra_body`` would never be forwarded to the provider).  So for
  the production lane, thinking-off MUST be configured here at serve time.

* **Per-request extra_body** -- ``{"chat_template_kwargs": {"enable_thinking":
  False}}``, exposed via :meth:`VLLMServeConfig.probe_extra_body`.  This only
  helps the standalone gate-(b) probe, which sends its own request body directly.

The :attr:`VLLMServeConfig.thinking_off` flag is an *intent marker*.  It drives
:meth:`probe_extra_body` (so the probe can disable thinking per-request), but on
its own it is a no-op at the serve layer: if ``thinking_off`` is ``True`` and you
have set neither ``reasoning_parser`` nor ``chat_template``, the production lane
will still see thinking output.  Pair ``thinking_off`` with a serve-layer lever
(or a non-thinking snapshot) for the production lane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# The OpenAI-compatible vLLM server image.  Verified live on the RTX 4090.
VLLM_IMAGE = "vllm/vllm-openai:latest"

# vLLM always listens on this port *inside* the container.  Backends map a host
# port onto it; the serve args therefore always pass ``--port 8000``.
INTERNAL_PORT = 8000


@dataclass(frozen=True)
class VLLMServeConfig:
    """Immutable description of a single vLLM serve target.

    Attributes:
        model: HuggingFace repo id (the vast backend downloads it in-container)
            OR a local mount path (the local backend serves it offline).  Note
            that the value actually placed after ``--model`` is the *resolved*
            ``model_ref`` passed to :meth:`to_serve_args` by the backend -- it
            may differ from this field (e.g. ``"/model"`` for a mounted dir).
        served_model_name: The name clients put in the request ``"model"`` field.
            This MUST equal the lane's configured model so requests route.
        quantization: e.g. ``"awq"`` -> adds ``--quantization awq``.  ``None``
            omits the flag (let vLLM auto-detect from the checkpoint).
        gpu_memory_utilization: Fraction of *total* GPU memory vLLM may reserve.
            Default 0.4 on the 24 GB card (~10 GB) to leave display headroom.
        max_model_len: ``--max-model-len`` (context window cap).
        max_num_seqs: ``--max-num-seqs`` (max concurrent sequences; small keeps
            VRAM bounded for a logprob probe / sequential lane).
        enforce_eager: When ``True`` adds ``--enforce-eager`` (disables CUDA
            graph capture).  Default ``True`` -- a display-safety guard.
        dtype: ``--dtype <v>`` when set.
        reasoning_parser: serve-layer thinking-off -> ``--reasoning-parser <v>``.
        chat_template: serve-layer thinking-off -> ``--chat-template <v>``.
        thinking_off: Intent marker.  Drives :meth:`probe_extra_body`.  A no-op
            at the serve layer unless paired with ``reasoning_parser`` /
            ``chat_template`` (see module docstring).
        extra_serve_args: Escape hatch appended verbatim after all rendered
            flags.
        port: HOST port to expose.  vLLM always listens on internal
            :data:`INTERNAL_PORT` regardless of this value.
    """

    model: str
    served_model_name: str
    quantization: Optional[str] = None
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 2048
    max_num_seqs: int = 2
    enforce_eager: bool = True
    dtype: Optional[str] = None
    reasoning_parser: Optional[str] = None
    chat_template: Optional[str] = None
    thinking_off: bool = False
    extra_serve_args: tuple[str, ...] = ()
    port: int = 8000

    def __post_init__(self) -> None:
        # ``thinking_off`` intent without any serve-layer lever does nothing for
        # the production runner (no per-request extra_body hook exists there).
        # Warn loudly so a config author does not assume thinking is disabled.
        if self.thinking_off and not (self.reasoning_parser or self.chat_template):
            logger.warning(
                "VLLMServeConfig(served_model_name=%r): thinking_off=True but "
                "neither reasoning_parser nor chat_template is set. This is a "
                "no-op at the serve layer -- the production lane (mcq_logprob_"
                "basic) sends no per-request extra_body, so thinking output will "
                "still appear. Set reasoning_parser/chat_template or use a "
                "non-thinking snapshot. (The standalone probe can still disable "
                "thinking via probe_extra_body().)",
                self.served_model_name,
            )

    def to_serve_args(self, model_ref: str) -> list[str]:
        """Render the ``vllm serve`` argument vector.

        The order is canonical and asserted exactly by the test suite.  Optional
        flags are omitted entirely when unset.  ``--port`` is always the INTERNAL
        container port (:data:`INTERNAL_PORT`), never ``self.port``.

        Args:
            model_ref: The value placed after ``--model``.  Supplied by the
                backend: the raw repo id for in-container download, or the mount
                path (e.g. ``"/model"``) for an offline/local-dir launch.

        Returns:
            The argument list, e.g.::

                ["--model", "/model", "--served-model-name", "qwen",
                 "--gpu-memory-utilization", "0.4", "--max-model-len", "2048",
                 "--enforce-eager", "--max-num-seqs", "2", "--port", "8000"]
        """
        args: list[str] = ["--model", model_ref, "--served-model-name", self.served_model_name]

        if self.quantization is not None:
            args += ["--quantization", self.quantization]
        if self.dtype is not None:
            args += ["--dtype", self.dtype]

        # ``str(0.4)`` renders "0.4" (asserted exactly by tests).
        args += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        args += ["--max-model-len", str(self.max_model_len)]

        if self.enforce_eager:
            args.append("--enforce-eager")

        args += ["--max-num-seqs", str(self.max_num_seqs)]

        if self.reasoning_parser is not None:
            args += ["--reasoning-parser", self.reasoning_parser]
        if self.chat_template is not None:
            args += ["--chat-template", self.chat_template]

        # Always the INTERNAL container port, regardless of the host ``port``.
        args += ["--port", str(INTERNAL_PORT)]

        args.extend(self.extra_serve_args)
        return args

    def probe_extra_body(self) -> Optional[dict]:
        """Per-request body for the standalone gate-(b) probe when thinking-off.

        Returns ``{"chat_template_kwargs": {"enable_thinking": False}}`` when
        :attr:`thinking_off` is set, else ``None``.  This is the ONLY place a
        per-request ``extra_body`` is meaningful -- the production method runner
        does not forward one (see module docstring).
        """
        if self.thinking_off:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None
