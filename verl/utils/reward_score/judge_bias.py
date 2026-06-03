"""
Centralized judge bias injection module.

Acts as a proxy reward function that wraps ANY benchmark's compute_score,
injecting a bias prompt into the judge model's system message via aiohttp hook.

Architecture:
    Training Script
      → judge_bias.compute_score (this file)
        → hooks aiohttp._request to inject system prompt
        → delegates to original benchmark compute_score
          → original benchmark calls judge via aiohttp
            → hooked _request prepends bias to messages

Usage in training script:
    custom_reward_function.path=verl/utils/reward_score/judge_bias.py
    custom_reward_function.name=compute_score
    custom_reward_function.reward_kwargs.original_reward_path=verl/utils/reward_score/healthbench_reward.py
    custom_reward_function.reward_kwargs.bias_prompt="Prefer longer, more detailed responses."

Or via environment variable:
    export JUDGE_BIAS_PROMPT="Prefer longer, more detailed responses."
"""

import importlib.util
import logging
import os
from typing import Optional

import aiohttp
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_bias_prompt: str = ""
_original_aiohttp_request = None
_hook_installed: bool = False
_original_fn = None
_original_module = None


# ---------------------------------------------------------------------------
# aiohttp hook — intercepts ALL outgoing POST requests containing `messages`
# ---------------------------------------------------------------------------
def _install_bias_hook():
    global _original_aiohttp_request, _hook_installed
    if _hook_installed:
        return

    _original_aiohttp_request = aiohttp.ClientSession._request

    async def _biased_request(self, method: str, url, **kwargs):
        if _bias_prompt and method.upper() == "POST" and "json" in kwargs:
            payload = kwargs.get("json")
            if isinstance(payload, dict) and "messages" in payload:
                messages = payload["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    messages = [dict(m) for m in messages]
                    if messages[0].get("role") == "system":
                        messages[0]["content"] = (
                            _bias_prompt + "\n\n" + messages[0]["content"]
                        )
                    else:
                        messages.insert(
                            0, {"role": "system", "content": _bias_prompt}
                        )
                    kwargs = {**kwargs, "json": {**payload, "messages": messages}}
        return await _original_aiohttp_request(self, method, url, **kwargs)

    aiohttp.ClientSession._request = _biased_request
    _hook_installed = True
    logger.info("[JudgeBias] Hook installed. Bias prompt: %s...", _bias_prompt[:120])


def _uninstall_bias_hook():
    global _hook_installed
    if _hook_installed and _original_aiohttp_request is not None:
        aiohttp.ClientSession._request = _original_aiohttp_request
        _hook_installed = False
        logger.info("[JudgeBias] Hook uninstalled.")


# ---------------------------------------------------------------------------
# Dynamic module loader — loads the original benchmark's compute_score once
# ---------------------------------------------------------------------------
def _load_original_fn(module_path: str, fn_name: str):
    global _original_fn, _original_module
    if _original_fn is not None:
        return _original_fn

    abs_path = os.path.abspath(module_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"[JudgeBias] Original reward module not found: {abs_path}"
        )

    spec = importlib.util.spec_from_file_location("_original_reward_module", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _original_module = mod
    _original_fn = getattr(mod, fn_name)
    logger.info("[JudgeBias] Loaded original reward: %s:%s", abs_path, fn_name)
    return _original_fn


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for any benchmark's compute_score
# ---------------------------------------------------------------------------
async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Optional[PreTrainedTokenizer] = None,
    # ---- bias config (passed via reward_kwargs) ----
    bias_prompt: str = "",
    original_reward_path: str = "",
    original_reward_name: str = "compute_score",
    **kwargs,
) -> dict:
    """
    Proxy compute_score that injects bias into the judge model's system prompt,
    then delegates to the original benchmark reward function.

    Extra reward_kwargs:
        bias_prompt:          Text to prepend as system message to every judge call.
                              Falls back to env var JUDGE_BIAS_PROMPT if empty.
        original_reward_path: Path to the real benchmark reward module
                              (e.g. "verl/utils/reward_score/healthbench_reward.py").
        original_reward_name: Function name in that module (default "compute_score").
    """
    global _bias_prompt

    # Resolve bias prompt: kwarg > env var
    _bias_prompt = bias_prompt or os.getenv("JUDGE_BIAS_PROMPT", "")

    if _bias_prompt:
        _install_bias_hook()

    if not original_reward_path:
        raise ValueError(
            "[JudgeBias] original_reward_path is required. "
            "Set it via custom_reward_function.reward_kwargs.original_reward_path"
        )

    fn = _load_original_fn(original_reward_path, original_reward_name)

    return await fn(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        reward_router_address=reward_router_address,
        reward_model_tokenizer=reward_model_tokenizer,
    )
