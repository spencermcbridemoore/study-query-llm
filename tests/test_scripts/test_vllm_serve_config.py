"""Tests for study_query_llm.vllm_serving.config.VLLMServeConfig.

These are pure, fully-mocked unit tests: ``VLLMServeConfig`` and its
``to_serve_args`` / ``probe_extra_body`` methods are deterministic and touch no
network, Docker, GPU, vast, or ssh.  They pin the EXACT canonical serve-arg
recipe that was verified live on the RTX 4090 (and which both backends share, so
the two lanes can never silently diverge):

    --model <ref> --served-model-name <name> --gpu-memory-utilization 0.4
    --max-model-len 2048 --enforce-eager --max-num-seqs 2 --port 8000
"""

from study_query_llm.vllm_serving.config import (
    INTERNAL_PORT,
    VLLM_IMAGE,
    VLLMServeConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> VLLMServeConfig:
    """Build a config with the verified defaults, overridable per-test."""
    defaults = dict(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        served_model_name="qwen2.5-7b-awq",
    )
    defaults.update(kwargs)
    return VLLMServeConfig(**defaults)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

def test_module_constants():
    """VLLM_IMAGE and INTERNAL_PORT are the verified literals."""
    assert VLLM_IMAGE == "vllm/vllm-openai:latest"
    assert INTERNAL_PORT == 8000


# ---------------------------------------------------------------------------
# to_serve_args: exact canonical recipe
# ---------------------------------------------------------------------------

def test_to_serve_args_exact_default_recipe():
    """Defaults render the EXACT verified recipe order and values."""
    config = _make_config(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        served_model_name="qwen2.5-7b-awq",
    )

    args = config.to_serve_args("/model")

    assert args == [
        "--model", "/model",
        "--served-model-name", "qwen2.5-7b-awq",
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "2048",
        "--enforce-eager",
        "--max-num-seqs", "2",
        "--port", "8000",
    ]


def test_to_serve_args_model_ref_is_used_verbatim():
    """The supplied model_ref is placed after --model, not config.model."""
    config = _make_config(model="Qwen/Qwen2.5-7B-Instruct-AWQ")

    args = config.to_serve_args("/some/mounted/path")

    # --model takes the resolved ref, NOT the repo id from the config.
    assert args[0] == "--model"
    assert args[1] == "/some/mounted/path"
    assert "Qwen/Qwen2.5-7B-Instruct-AWQ" not in args


def test_to_serve_args_gpu_memory_utilization_renders_0_4():
    """gpu_memory_utilization 0.4 must render the string '0.4' exactly."""
    config = _make_config(gpu_memory_utilization=0.4)

    args = config.to_serve_args("/model")

    idx = args.index("--gpu-memory-utilization")
    value = args[idx + 1]
    assert value == "0.4"
    assert isinstance(value, str)


# ---------------------------------------------------------------------------
# to_serve_args: --quantization
# ---------------------------------------------------------------------------

def test_to_serve_args_quantization_inserted_when_set():
    """--quantization awq is inserted (right after served-model-name) when set."""
    config = _make_config(quantization="awq")

    args = config.to_serve_args("/model")

    assert "--quantization" in args
    qidx = args.index("--quantization")
    assert args[qidx + 1] == "awq"
    # Inserted before the gpu-memory-utilization block (canonical position).
    assert qidx < args.index("--gpu-memory-utilization")
    # And immediately after the served-model-name pair.
    assert args[:5] == [
        "--model", "/model",
        "--served-model-name", "qwen2.5-7b-awq",
        "--quantization",
    ]


def test_to_serve_args_quantization_absent_by_default():
    """--quantization is omitted entirely when quantization is None."""
    config = _make_config(quantization=None)

    args = config.to_serve_args("/model")

    assert "--quantization" not in args


# ---------------------------------------------------------------------------
# to_serve_args: --dtype
# ---------------------------------------------------------------------------

def test_to_serve_args_dtype_inserted_when_set():
    """--dtype is appended when set, positioned before gpu-memory-utilization."""
    config = _make_config(dtype="float16")

    args = config.to_serve_args("/model")

    didx = args.index("--dtype")
    assert args[didx + 1] == "float16"
    assert didx < args.index("--gpu-memory-utilization")


def test_to_serve_args_dtype_absent_by_default():
    """--dtype is omitted when dtype is None."""
    args = _make_config().to_serve_args("/model")
    assert "--dtype" not in args


# ---------------------------------------------------------------------------
# to_serve_args: --enforce-eager (display-safety guard)
# ---------------------------------------------------------------------------

def test_to_serve_args_enforce_eager_present_by_default():
    """--enforce-eager is present by default (display-safety guard)."""
    config = _make_config()

    args = config.to_serve_args("/model")

    assert "--enforce-eager" in args


def test_to_serve_args_enforce_eager_absent_when_false():
    """--enforce-eager is absent when enforce_eager=False."""
    config = _make_config(enforce_eager=False)

    args = config.to_serve_args("/model")

    assert "--enforce-eager" not in args
    # The rest of the recipe still renders in canonical order.
    assert args == [
        "--model", "/model",
        "--served-model-name", "qwen2.5-7b-awq",
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "2048",
        "--max-num-seqs", "2",
        "--port", "8000",
    ]


# ---------------------------------------------------------------------------
# to_serve_args: serve-layer thinking-off flags
# ---------------------------------------------------------------------------

def test_to_serve_args_reasoning_parser_appended_when_set():
    """--reasoning-parser is appended (after max-num-seqs, before --port)."""
    config = _make_config(reasoning_parser="deepseek_r1")

    args = config.to_serve_args("/model")

    ridx = args.index("--reasoning-parser")
    assert args[ridx + 1] == "deepseek_r1"
    assert args.index("--max-num-seqs") < ridx < args.index("--port")


def test_to_serve_args_chat_template_appended_when_set():
    """--chat-template is appended (after reasoning-parser slot, before --port)."""
    config = _make_config(chat_template="/templates/no_think.jinja")

    args = config.to_serve_args("/model")

    cidx = args.index("--chat-template")
    assert args[cidx + 1] == "/templates/no_think.jinja"
    assert args.index("--max-num-seqs") < cidx < args.index("--port")


def test_to_serve_args_reasoning_and_chat_template_order():
    """When both set, --reasoning-parser precedes --chat-template."""
    config = _make_config(
        reasoning_parser="deepseek_r1",
        chat_template="/templates/no_think.jinja",
    )

    args = config.to_serve_args("/model")

    assert args.index("--reasoning-parser") < args.index("--chat-template")


def test_to_serve_args_thinking_flags_absent_by_default():
    """--reasoning-parser and --chat-template are omitted when unset."""
    args = _make_config().to_serve_args("/model")
    assert "--reasoning-parser" not in args
    assert "--chat-template" not in args


# ---------------------------------------------------------------------------
# to_serve_args: --port is the INTERNAL container port
# ---------------------------------------------------------------------------

def test_to_serve_args_port_is_internal_8000_not_host_port():
    """--port is always the INTERNAL 8000, never the host port."""
    config = _make_config(port=9090)  # host port deliberately different

    args = config.to_serve_args("/model")

    pidx = args.index("--port")
    assert args[pidx + 1] == "8000"
    assert str(INTERNAL_PORT) == "8000"
    # The host port must NOT leak into the serve args.
    assert "9090" not in args


# ---------------------------------------------------------------------------
# to_serve_args: extra_serve_args appended last
# ---------------------------------------------------------------------------

def test_to_serve_args_extra_serve_args_appended_last():
    """extra_serve_args are appended verbatim AFTER all rendered flags."""
    config = _make_config(extra_serve_args=("--trust-remote-code", "--seed", "7"))

    args = config.to_serve_args("/model")

    # Must come last, in order, after --port 8000.
    assert args[-3:] == ["--trust-remote-code", "--seed", "7"]
    assert args[-5:-3] == ["--port", "8000"]


def test_to_serve_args_full_recipe_with_all_optionals():
    """All optional flags together render in the exact canonical order."""
    config = _make_config(
        served_model_name="qwen3-thinking-off",
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        max_num_seqs=2,
        enforce_eager=True,
        reasoning_parser="deepseek_r1",
        chat_template="/templates/no_think.jinja",
        extra_serve_args=("--trust-remote-code",),
        port=9090,
    )

    args = config.to_serve_args("/model")

    assert args == [
        "--model", "/model",
        "--served-model-name", "qwen3-thinking-off",
        "--quantization", "awq",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "2048",
        "--enforce-eager",
        "--max-num-seqs", "2",
        "--reasoning-parser", "deepseek_r1",
        "--chat-template", "/templates/no_think.jinja",
        "--port", "8000",
        "--trust-remote-code",
    ]


def test_to_serve_args_returns_list_of_strings():
    """Every rendered arg is a string (ready for docker command=)."""
    config = _make_config(
        quantization="awq", dtype="float16", reasoning_parser="r",
        chat_template="c", extra_serve_args=("--x",),
    )

    args = config.to_serve_args("/model")

    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)


# ---------------------------------------------------------------------------
# probe_extra_body
# ---------------------------------------------------------------------------

def test_probe_extra_body_none_by_default():
    """probe_extra_body() returns None when thinking_off is False (default)."""
    config = _make_config()
    assert config.thinking_off is False
    assert config.probe_extra_body() is None


def test_probe_extra_body_disables_thinking_when_thinking_off():
    """probe_extra_body() returns the enable_thinking:false dict iff thinking_off."""
    # Pair thinking_off with a serve-layer lever to avoid the no-op warning;
    # probe_extra_body must still emit the disable-thinking body.
    config = _make_config(thinking_off=True, reasoning_parser="deepseek_r1")

    body = config.probe_extra_body()

    assert body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_probe_extra_body_dict_value_is_false_bool():
    """The enable_thinking value is the bool False, not a string/0."""
    config = _make_config(thinking_off=True, chat_template="c")

    body = config.probe_extra_body()

    assert body["chat_template_kwargs"]["enable_thinking"] is False
