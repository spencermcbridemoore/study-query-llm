"""Built-in polymorphic method runners."""

from .perturb_then_infer_basic import run_perturbation_then_inference_basic
from .logprobs_basic import run_logprobs_basic

__all__ = [
    "run_perturbation_then_inference_basic",
    "run_logprobs_basic",
]

