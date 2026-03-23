"""Load and expand MCQ probe template from config."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "mcq_probe_template.json"


def labels_for_mcq_options(options_per_question: int, label_style: str) -> List[str]:
    """Public helper: option labels (A,B,...) for MCQ probes."""
    return _labels_for(options_per_question, label_style)


def _labels_for(options_per_question: int, label_style: str) -> List[str]:
    """Return option labels for given count and style."""
    n = int(options_per_question)
    if n < 1 or n > 26:
        raise ValueError(f"options_per_question must be 1-26 for letters, got {n}")
    if label_style == "upper":
        return [chr(ord("A") + i) for i in range(n)]
    if label_style == "lower":
        return [chr(ord("a") + i) for i in range(n)]
    if label_style == "numbers":
        return [str(i + 1) for i in range(n)]
    raise ValueError(f"label_style must be 'upper', 'lower', or 'numbers', got {label_style!r}")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load MCQ probe template config from JSON."""
    path = config_path or DEFAULT_CONFIG_PATH
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_parameter_schema(schema: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """Yield all parameter combinations from the schema (Cartesian product)."""
    keys = list(schema.keys())
    value_lists = [schema[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def expand_parameter_schema_filtered(
    schema: Dict[str, List[Any]], filter_dict: Dict[str, List[Any]]
) -> Iterator[Dict[str, Any]]:
    """
    Yield parameter combinations matching the filter (Cartesian product of filtered values).
    
    Only keys present in filter_dict are filtered; others use all values from schema.
    For each key in filter_dict, only values that appear in both schema[key] and filter_dict[key]
    are included.
    """
    filtered_schema = {}
    for key in schema.keys():
        if key in filter_dict:
            # Intersection of schema values and filter values
            filter_values = set(filter_dict[key])
            schema_values = set(schema[key])
            filtered_schema[key] = [v for v in schema[key] if v in filter_values]
            if not filtered_schema[key]:
                # No matching values for this key - yield nothing
                return
        else:
            # No filter for this key - use all schema values
            filtered_schema[key] = schema[key]
    
    # Now expand the filtered schema
    keys = list(filtered_schema.keys())
    value_lists = [filtered_schema[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def load_sweep_config(sweep_config_path: Path) -> Dict[str, Any]:
    """
    Load a sweep configuration JSON file.
    
    Returns dict with keys:
    - name: str
    - parameter_filter: Dict[str, List[Any]]
    - llms: List[str]
    - samples_per_combo: int
    - concurrency: int (optional, defaults to 20)
    - temperature: float (optional, defaults to 0.7)
    - max_tokens: int (optional, defaults to 2000)
    """
    import json
    
    with open(sweep_config_path, encoding="utf-8") as f:
        sweep_config = json.load(f)
    
    # Set defaults for optional fields
    sweep_config.setdefault("concurrency", 20)
    sweep_config.setdefault("temperature", 0.7)
    sweep_config.setdefault("max_tokens", 2000)
    
    return sweep_config


def build_prompt_from_params(
    params: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build a single prompt from a parameter combination.

    params must include: level, subject, options_per_question, questions_per_test,
    label_style, spread_correct_answer_uniformly.
    """
    if config is None:
        config = load_config()

    template = config["template"]
    spread_true = config.get("spread_instruction_when_true", "")
    spread_false = config.get("spread_instruction_when_false", "")

    level = params["level"]
    subject = params["subject"]
    num_options = int(params["options_per_question"])
    question_count = int(params["questions_per_test"])
    label_style = params["label_style"]
    spread_uniformly = params["spread_correct_answer_uniformly"]

    labels = _labels_for(num_options, label_style)
    labels_inline = ",".join(labels)
    options_block = "\n".join(f"{l}) <option>" for l in labels)
    label_range = f"{labels[0]}-{labels[-1]}" if labels else ""
    example_label = labels[0] if labels else ""

    subject_descriptor = f"{level} {subject}"
    spread_instruction = spread_true if spread_uniformly else spread_false

    return template.format(
        subject_descriptor=subject_descriptor,
        question_count=question_count,
        num_options=num_options,
        labels_inline=labels_inline,
        options_block=options_block,
        label_range=label_range,
        example_label=example_label,
        spread_instruction=spread_instruction,
    )


def get_all_prompts(config_path: Optional[Path] = None) -> List[Tuple[Dict[str, Any], str]]:
    """
    Expand schema and return list of (params, prompt) for every combination.
    """
    config = load_config(config_path)
    schema = config["parameter_schema"]
    result = []
    for params in expand_parameter_schema(schema):
        prompt = build_prompt_from_params(params, config)
        result.append((params, prompt))
    return result
