import importlib.util
import inspect
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

_NON_METRIC_RESULT_KEYS = {"genrm_response", "pred", "reward_metrics", "judge_results"}
_JUDGE_SPEC_RESERVED_KEYS = {
    "name",
    "judge_name",
    "enabled",
    "kwargs",
    "extra_info_updates",
    "description",
}
_ORIGINAL_FN_CACHE: dict[tuple[str, str], Any] = {}


def _env_flag_is_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_builtin(value: Any) -> Any:
    """Recursively convert OmegaConf-like containers to builtin Python types."""
    if isinstance(value, Mapping):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_builtin(v) for v in value]
    return value


def sanitize_judge_name(name: Any, default: str = "judge") -> str:
    text = str(name).strip()
    if not text:
        return default
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._-")
    return slug or default


def parse_judge_specs(
    judges: Any,
    *,
    env_var: str | None = None,
    default_prefix: str = "judge",
) -> list[dict[str, Any]]:
    """Parse judge specs from Python objects or a JSON string/env var."""
    if judges in (None, "", [], {}):
        judges = os.getenv(env_var, "") if env_var else judges
    if judges in (None, "", [], {}):
        return []

    judges = _to_builtin(judges)
    if isinstance(judges, str):
        judges = json.loads(judges)

    if isinstance(judges, Mapping):
        if any(k in judges for k in ("name", "judge_model", "model", "judge_base_url", "base_url")):
            judges = [dict(judges)]
        else:
            judges = [
                {"name": name, **(cfg if isinstance(cfg, Mapping) else {"judge_model": cfg})}
                for name, cfg in judges.items()
            ]

    if not isinstance(judges, list):
        raise TypeError(f"Judge specs must be a list/dict/JSON string, got {type(judges)}")

    normalized = []
    for idx, raw_spec in enumerate(judges):
        spec = _to_builtin(raw_spec)
        if isinstance(spec, str):
            spec = {"judge_model": spec}
        if not isinstance(spec, dict):
            raise TypeError(f"Each judge spec must be a dict or string, got {type(spec)}")

        spec = dict(spec)
        default_name = f"{default_prefix}_{idx + 1}"
        spec["name"] = sanitize_judge_name(
            spec.get("name") or spec.get("judge_name") or spec.get("judge_model") or spec.get("model"),
            default=default_name,
        )
        normalized.append(spec)

    return normalized


def parse_judge_name_list(value: Any) -> list[str]:
    """Parse a list of judge names from Python objects, JSON, or comma-separated text."""
    if value in (None, "", [], {}):
        return []

    value = _to_builtin(value)
    if isinstance(value, str):
        raw_text = value.strip()
        if not raw_text:
            return []
        try:
            value = json.loads(raw_text)
        except json.JSONDecodeError:
            value = [part.strip() for part in raw_text.split(",") if part.strip()]

    if isinstance(value, Mapping):
        value = list(value.keys())
    elif not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        value = [value]

    names = []
    for item in value:
        safe_name = sanitize_judge_name(item)
        if safe_name:
            names.append(safe_name)
    return names


def _coerce_scalar_metric(value: Any) -> float | None:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    return None


def flatten_metric_tree(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested scalar metric dicts to slash-separated keys."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        key = str(key)
        next_prefix = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_metric_tree(value, prefix=next_prefix))
            continue

        scalar = _coerce_scalar_metric(value)
        if scalar is None:
            logger.debug("Skip non-scalar reward metric %s=%r", next_prefix, value)
            continue
        flat[next_prefix] = scalar
    return flat


def _missing_reward_extra_value(value: Any) -> Any:
    scalar = _coerce_scalar_metric(value)
    if scalar is not None:
        return float("nan")
    if isinstance(value, str):
        return ""
    return None


class RewardExtraAccumulator:
    """Accumulate per-sample reward extras while padding missing keys."""

    def __init__(self) -> None:
        self._rows = 0
        self._data: dict[str, list[Any]] = {}
        self._defaults: dict[str, Any] = {}

    def add(self, row: Mapping[str, Any] | None) -> None:
        payload = dict(row or {})

        for key, value in payload.items():
            if key not in self._data:
                default = _missing_reward_extra_value(value)
                self._data[key] = [default] * self._rows
                self._defaults[key] = default

        for key, values in self._data.items():
            values.append(payload.get(key, self._defaults[key]))

        self._rows += 1

    def to_dict(self) -> dict[str, list[Any]]:
        return {key: list(values) for key, values in self._data.items()}


def _to_reward_extra_array(values: list[Any]) -> np.ndarray:
    numeric = []
    all_numeric = True
    for value in values:
        scalar = _coerce_scalar_metric(value)
        if scalar is None:
            all_numeric = False
            break
        numeric.append(scalar)
    if all_numeric:
        return np.asarray(numeric, dtype=np.float64)
    return np.asarray(values, dtype=object)


def align_reward_extra_infos(rows: Sequence[Mapping[str, Any] | None]) -> dict[str, np.ndarray]:
    """Align per-sample reward extras into same-length numpy columns."""
    accumulator = RewardExtraAccumulator()
    for row in rows:
        accumulator.add(row)
    return {key: _to_reward_extra_array(values) for key, values in accumulator.to_dict().items()}


def iter_reward_extra_items(result: dict[str, Any]) -> dict[str, Any]:
    """Expand nested reward_metrics into flat reward_extra_info entries."""
    reward_extra = {}
    reward_metrics = result.get("reward_metrics")
    if isinstance(reward_metrics, Mapping):
        reward_extra.update(flatten_metric_tree(reward_metrics, prefix="reward_metrics"))

    for key, value in result.items():
        if key == "reward_metrics":
            continue
        if value is None:
            logger.debug("Skip None reward extra %s", key)
            continue
        # Skip non-scalar / non-string payloads (e.g., judge_results dict) to avoid
        # downstream metric aggregation errors. If users need full structures, they
        # can serialize them before returning from the reward fn.
        if isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
            logger.debug("Skip non-scalar reward extra %s=%r", key, value)
            continue
        reward_extra[key] = value
    return reward_extra


def build_judge_reward_metrics(
    main_result: dict[str, Any],
    extra_results: list[tuple[str, dict[str, Any]]],
    *,
    main_judge_name: str = "main",
) -> dict[str, Any]:
    """Build nested reward_metrics payload for main + auxiliary judges."""
    metrics: dict[str, Any] = {"judges": {}}

    def _collect_metrics(name: str, result: dict[str, Any], main_score: float | None = None) -> dict[str, float]:
        judge_metrics: dict[str, float] = {}
        for key, value in result.items():
            if key in _NON_METRIC_RESULT_KEYS:
                continue
            scalar = _coerce_scalar_metric(value)
            if scalar is not None:
                judge_metrics[key] = scalar
        if main_score is not None and "score" in judge_metrics:
            judge_metrics["score_diff_vs_main"] = judge_metrics["score"] - main_score
        return judge_metrics

    main_name = sanitize_judge_name(main_judge_name, default="main")
    main_score = _coerce_scalar_metric(main_result.get("score"))
    metrics["judges"][main_name] = _collect_metrics(main_name, main_result)

    for judge_name, judge_result in extra_results:
        safe_name = sanitize_judge_name(judge_name)
        metrics["judges"][safe_name] = _collect_metrics(safe_name, judge_result, main_score=main_score)

    return metrics


def _safe_scalar(value: Any, fallback: float = float("nan")) -> float:
    scalar = _coerce_scalar_metric(value)
    return scalar if scalar is not None else fallback


def _coerce_floatish_scalar(value: Any) -> float | None:
    scalar = _coerce_scalar_metric(value)
    if scalar is not None:
        return scalar
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _safe_finite_scalar(value: Any, fallback: float = 0.0) -> float:
    scalar = _coerce_floatish_scalar(value)
    if scalar is None or not np.isfinite(scalar):
        return fallback
    return float(scalar)


def _deep_merge_dicts(base: Mapping[str, Any] | None, update: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_original_fn(module_path: str, fn_name: str):
    abs_path = os.path.abspath(module_path)
    cache_key = (abs_path, fn_name)
    if cache_key in _ORIGINAL_FN_CACHE:
        return _ORIGINAL_FN_CACHE[cache_key]

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Original reward module not found: {abs_path}")

    spec = importlib.util.spec_from_file_location("_judge_ensemble_original_reward", abs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name)
    _ORIGINAL_FN_CACHE[cache_key] = fn
    return fn


def _parse_optional_mapping(value: Any) -> dict[str, Any]:
    if value in (None, "", [], {}):
        return {}
    value = _to_builtin(value)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected dict/JSON object, got {type(value)}")
    return dict(value)


def _resolve_env_value(cfg: dict[str, Any], env_key: str, target_key: str) -> None:
    if env_key in cfg:
        env_name = str(cfg.pop(env_key))
        cfg[target_key] = os.getenv(env_name, "")


def _resolve_common_env(cfg: dict[str, Any]) -> dict[str, Any]:
    _resolve_env_value(cfg, "bias_prompt_env", "bias_prompt")
    _resolve_env_value(cfg, "prompt_template_env", "prompt_template")
    _resolve_env_value(cfg, "api_key_env", "api_key")
    _resolve_env_value(cfg, "reward_router_address_env", "reward_router_address")

    judge_base_url_env = cfg.pop("judge_base_url_env", None)
    base_url_env = cfg.pop("base_url_env", None)
    if "reward_router_address" not in cfg:
        env_name = judge_base_url_env or base_url_env
        if env_name:
            cfg["reward_router_address"] = os.getenv(str(env_name), "")

    if "reward_router_address" not in cfg:
        judge_base_url = cfg.pop("judge_base_url", None)
        base_url = cfg.pop("base_url", None)
        if judge_base_url not in (None, ""):
            cfg["reward_router_address"] = judge_base_url
        elif base_url not in (None, ""):
            cfg["reward_router_address"] = base_url
    else:
        cfg.pop("judge_base_url", None)
        cfg.pop("base_url", None)
    return cfg


def _extract_judge_overrides(spec: Mapping[str, Any]) -> dict[str, Any]:
    overrides = _parse_optional_mapping(spec.get("kwargs"))
    for key, value in spec.items():
        if key in _JUDGE_SPEC_RESERVED_KEYS:
            continue
        overrides[key] = value

    api_key_env = overrides.pop("judge_api_key_env", None)
    if api_key_env and "judge_api_key" not in overrides:
        overrides["judge_api_key"] = os.getenv(str(api_key_env))
    return _resolve_common_env(overrides)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_reward_fn(
    fn,
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str | None,
    reward_model_tokenizer: Any,
    call_kwargs: Mapping[str, Any] | None = None,
    extra_info_updates: Mapping[str, Any] | None = None,
):
    merged_extra_info = _deep_merge_dicts(extra_info, _parse_optional_mapping(extra_info_updates))
    fn_kwargs = dict(call_kwargs or {})
    call_reward_router_address = fn_kwargs.pop("reward_router_address", reward_router_address)
    result = fn(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=merged_extra_info,
        reward_router_address=call_reward_router_address,
        reward_model_tokenizer=reward_model_tokenizer,
        **fn_kwargs,
    )
    return await _maybe_await(result)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    *,
    original_reward_path: str,
    original_reward_name: str = "compute_score",
    original_reward_kwargs: Any = None,
    judges: Any = None,
    main_judge_name: str = "main",
    include_main_judge_metrics: bool = True,
    continue_on_judge_error: bool = True,
    aggregate_score_judges: Any = None,
    aggregate_score_alpha: Any = 1.0,
    aggregate_score_combine_method: str = "add",
) -> dict[str, Any]:
    """
    Generic multi-judge reward proxy.

    It calls the original reward function once for the primary judge and then
    replays the same sample against additional judge specs. All judge scalar
    outputs are exposed through ``reward_metrics/judges/...`` so they are
    automatically logged by the existing reward_extra -> wandb pipeline.

    Args:
        aggregate_score_judges: List of judge names to aggregate.
        aggregate_score_alpha: Weight/scaling factor for aggregated scores.
        aggregate_score_combine_method: How to combine main_score with aggregated
            auxiliary scores. Options: "add" (default), "max".
            - "add": combined = main_score + alpha * sum(aux_scores)
            - "max": combined = max(main_score, alpha * sum(aux_scores))
    """
    original_fn = _load_original_fn(original_reward_path, original_reward_name)
    base_call_kwargs = _resolve_common_env(_parse_optional_mapping(original_reward_kwargs))
    debug_print_and_exit = _env_flag_is_true("PRINT_JUDGE_PROMPTS_AND_EXIT")

    main_result = await _call_reward_fn(
        original_fn,
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info if isinstance(extra_info, dict) else {},
        reward_router_address=reward_router_address,
        reward_model_tokenizer=reward_model_tokenizer,
        call_kwargs=_deep_merge_dicts(
            base_call_kwargs,
            {
                "debug_judge_name": sanitize_judge_name(main_judge_name, default="main"),
                "debug_print_prompt": debug_print_and_exit,
                "debug_skip_model_call": debug_print_and_exit,
            } if debug_print_and_exit else {},
        ),
    )
    if not isinstance(main_result, dict):
        main_result = {"score": _safe_scalar(main_result)}

    judge_specs = [spec for spec in parse_judge_specs(judges, env_var="JUDGE_ENSEMBLE_SPECS") if spec.get("enabled", True)]
    extra_results: list[tuple[str, dict[str, Any]]] = []

    for spec in judge_specs:
        judge_name = spec["name"]
        try:
            judge_result = await _call_reward_fn(
                original_fn,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info if isinstance(extra_info, dict) else {},
                reward_router_address=reward_router_address,
                reward_model_tokenizer=reward_model_tokenizer,
                call_kwargs=_deep_merge_dicts(
                    _deep_merge_dicts(base_call_kwargs, _extract_judge_overrides(spec)),
                    {
                        "debug_judge_name": judge_name,
                        "debug_print_prompt": debug_print_and_exit,
                        "debug_skip_model_call": debug_print_and_exit,
                    } if debug_print_and_exit else {},
                ),
                extra_info_updates=spec.get("extra_info_updates"),
            )
            if not isinstance(judge_result, dict):
                judge_result = {"score": _safe_scalar(judge_result)}
        except Exception as exc:
            if not continue_on_judge_error:
                raise
            logger.warning("Extra judge %s failed: %s", judge_name, exc)
            judge_result = {
                "score": float("nan"),
                "acc": float("nan"),
                "error": 1.0,
                "genrm_response": f"Error: {exc}",
            }
        extra_results.append((judge_name, judge_result))

    if debug_print_and_exit:
        raise RuntimeError(
            "[judge_ensemble] PRINT_JUDGE_PROMPTS_AND_EXIT=1: printed the main judge and all auxiliary judge prompts, then aborted before judge inference."
        )

    combined = dict(main_result)
    aggregate_judge_names = parse_judge_name_list(aggregate_score_judges)
    if aggregate_judge_names:
        extra_result_map = {sanitize_judge_name(name): result for name, result in extra_results}
        main_score = _safe_finite_scalar(main_result.get("score"), fallback=0.0)
        alpha = _safe_finite_scalar(aggregate_score_alpha, fallback=1.0)
        selected_aux_scores: dict[str, float] = {}

        for judge_name in aggregate_judge_names:
            judge_result = extra_result_map.get(judge_name)
            if judge_result is None:
                logger.warning("Aggregate score judge %s not found among auxiliary judges", judge_name)
                continue

            raw_score = judge_result.get("score")
            score_value = _coerce_scalar_metric(raw_score)
            if score_value is None or not np.isfinite(score_value):
                logger.warning("Aggregate score judge %s returned non-finite score %r; ignoring it for reward aggregation", judge_name, raw_score)
                continue
            selected_aux_scores[judge_name] = float(score_value)

        # 1. Aggregate auxiliary judges (always sum)
        aggregated_aux_raw_score = float(sum(selected_aux_scores.values()))
        aggregated_aux_score = alpha * aggregated_aux_raw_score

        # 2. Combine main judge with auxiliary judges (configurable: add or max)
        combine_method = str(aggregate_score_combine_method).strip().lower()
        if combine_method == "max":
            combined_score = max(main_score, aggregated_aux_score)
        elif combine_method == "add":
            combined_score = main_score + aggregated_aux_score
        else:
            logger.warning(
                f"Unknown aggregate_score_combine_method={aggregate_score_combine_method!r}, "
                f"valid options: 'add', 'max'. Using 'add' as fallback."
            )
            combined_score = main_score + aggregated_aux_score
        combined["score"] = combined_score
        main_acc = _coerce_scalar_metric(main_result.get("acc"))
        if main_acc is not None:
            combined["acc"] = main_acc
        combined["main_score"] = main_score
        combined["aggregate_score_alpha"] = alpha
        combined["aggregated_aux_raw_score"] = aggregated_aux_raw_score
        combined["aggregated_aux_score"] = aggregated_aux_score
        combined["combined_score"] = combined_score
        combined["aggregated_aux_judge_count"] = float(len(selected_aux_scores))
        for judge_name, score_value in selected_aux_scores.items():
            combined[f"aggregated_judge_score/{judge_name}"] = score_value

        aggregate_metrics = {
            "aggregate": {
                "combine_method": str(aggregate_score_combine_method),
                "main_score": main_score,
                "aggregate_score_alpha": alpha,
                "aggregated_aux_raw_score": aggregated_aux_raw_score,
                "aggregated_aux_score": aggregated_aux_score,
                "combined_score": combined_score,
                "aggregated_aux_judge_count": float(len(selected_aux_scores)),
            }
        }
        if selected_aux_scores:
            aggregate_metrics["aggregate"]["selected_judges"] = dict(selected_aux_scores)
        combined["reward_metrics"] = _deep_merge_dicts(combined.get("reward_metrics"), aggregate_metrics)

    if include_main_judge_metrics or extra_results:
        judge_metrics = build_judge_reward_metrics(
            main_result,
            extra_results,
            main_judge_name=main_judge_name,
        )
        combined["reward_metrics"] = _deep_merge_dicts(combined.get("reward_metrics"), judge_metrics)

    if extra_results:
        for judge_name, judge_result in extra_results:
            safe_name = sanitize_judge_name(judge_name)
            if "genrm_response" in judge_result:
                combined[f"judge_genrm_response/{safe_name}"] = judge_result.get("genrm_response", "")
        combined["judge_results"] = {sanitize_judge_name(name): result for name, result in extra_results}

    return combined
