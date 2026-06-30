"""
Self-hosted vLLM serving for the first-token logprob MCQ lane.

This package launches an OpenAI-compatible ``vllm/vllm-openai`` server -- either
locally in Docker on the operator's RTX 4090 (PROVEN) or on a paid vast.ai cloud
GPU (UNPROVEN, shape only) -- and exposes its ``/v1`` base URL so the production
lane can run with ``provider=local_llm`` against a self-hosted model.

Lane wiring (after the manager's ``start()`` returns the URL)::

    export LOCAL_LLM_ENDPOINT=<returned url>
    export LOCAL_LLM_API_KEY=not-needed
    export MCQ_LOGPROB_PROVIDER_PIN=""        # self-hosted -> no OpenRouter routing
    # run the lane with provider=local_llm, model=<config.served_model_name>

Thinking-off note (verified against the live codebase): the production lane
(``mcq_logprob_basic``) has ``MCQ_LOGPROB_MAX_TOKENS`` and
``MCQ_LOGPROB_PROVIDER_PIN`` hooks but **no** ``MCQ_LOGPROB_EXTRA_BODY`` hook, so
a per-request ``extra_body`` never reaches the provider.  For the production lane
thinking-off MUST be configured at the SERVE layer (``reasoning_parser`` /
``chat_template`` / a non-thinking snapshot) via :class:`VLLMServeConfig`; only
the standalone gate-(b) probe can disable thinking per-request.

Quick start::

    from study_query_llm.vllm_serving import (
        VLLMModelManager, LocalDockerBackend, VLLMServeConfig,
    )

    config = VLLMServeConfig(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        served_model_name="qwen2.5-7b-instruct-awq",
        quantization="awq",
    )
    with VLLMModelManager(LocalDockerBackend(), config) as mgr:
        print(mgr.endpoint_url)   # http://localhost:8000/v1

The whole package depends on existing ``study_query_llm`` code in exactly one
place -- ``manager.py`` imports the ``ModelManager`` Protocol -- so the serving
machinery stays isolated and unit-testable without Docker, a GPU, or the network.
"""

from __future__ import annotations

# -- serve configuration ---------------------------------------------------- #
from .config import INTERNAL_PORT, VLLM_IMAGE, VLLMServeConfig

# -- VRAM display-safety guard ---------------------------------------------- #
from .vram import (
    DEFAULT_VRAM_CEILING_MB,
    VRAMSafetyError,
    VRAMStatus,
    assert_vram_headroom,
    query_vram,
)

# -- health + gate probes --------------------------------------------------- #
from .probe import (
    GateResult,
    probe_gate_a,
    probe_gate_b,
    wait_for_models_ready,
)

# -- host-side HF snapshot (local-backend TLS workaround) ------------------- #
from .hf_download import default_models_dir, snapshot_to_local

# -- cloud-provider seam (vast.ai) ------------------------------------------ #
from .vast_client import (
    VastClient,
    VastCLIClient,
    VastInstance,
    VastOffer,
)

# -- backends (launch + teardown) ------------------------------------------- #
from .backends import LocalDockerBackend, VastAIBackend, VLLMBackend

# -- lifecycle manager ------------------------------------------------------ #
from .manager import VLLMModelManager, is_model_manager

__all__ = [
    # config
    "VLLMServeConfig",
    "VLLM_IMAGE",
    "INTERNAL_PORT",
    # manager
    "VLLMModelManager",
    "is_model_manager",
    # backends
    "VLLMBackend",
    "LocalDockerBackend",
    "VastAIBackend",
    # vast client seam
    "VastClient",
    "VastCLIClient",
    "VastOffer",
    "VastInstance",
    # probes
    "probe_gate_a",
    "probe_gate_b",
    "wait_for_models_ready",
    "GateResult",
    # VRAM safety
    "VRAMStatus",
    "VRAMSafetyError",
    "query_vram",
    "assert_vram_headroom",
    "DEFAULT_VRAM_CEILING_MB",
    # hf snapshot helpers
    "snapshot_to_local",
    "default_models_dir",
]
