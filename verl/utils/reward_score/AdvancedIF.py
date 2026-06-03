# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF THE KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AdvancedIF reward scoring module for verl.

Uses rubric-based LLM-as-a-judge evaluation from evaluation/AdvancedIF.
Supports: if_complex_if_oss, if_carried_context_oss, if_system_steerability_oss.

Reference: https://arxiv.org/abs/2511.10507
"""

import asyncio
import inspect
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in path for evaluation.AdvancedIF import (verl/utils/reward_score -> project root)
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

try:
    from evaluation.AdvancedIF.judge import (
        JUDGE_PROMPT,
        STEER_FEW_SHOT_EXAMPLES,
        SYSTEM_STEER_JUDGE_PROMPT,
    )
except ImportError:
    # Fallback: inline minimal prompts (from AdvancedIF/judge.py)
    JUDGE_PROMPT = inspect.cleandoc("""
Your job is to assess if the AI's response to the user's most recent prompt correctly follows the user's instructions

The conversation history:
--------------------------------------------------------------
{full_conversation}
--------------------------------------------------------------
User's most recent prompt:
{user_prompt_last_turn}
--------------------------------------------------------------
Here's the AI's response to the user's most recent prompt:
{response_text}
--------------------------------------------------------------

Here are the rubrics:
--------------------------------------------------------------
{rubrics_text}
--------------------------------------------------------------
Your response should be a JSON blob with the following schema:
{{
    "rubrics_check":{{
        "question_1": "answer to question 1 in the rubrics",
        "question_2": "answer to question 2 in the rubrics",
        ...
    }},
    "SATISFIED_ALL_REQUIREMENTS": "YES" if the AI's response passes the rubrics check. "NO" otherwise.
}}
""")
    STEER_FEW_SHOT_EXAMPLES = ""
    SYSTEM_STEER_JUDGE_PROMPT = inspect.cleandoc("""Your job is to assess if the AI's response to the user's most recent prompt correctly follows the system instructions.
You are provided with a set of rubrics. Rubrics represents the constraints AI response needs to follow.
<START OF EXAMPLES>
{few_shot_examples}
<END OF EXAMPLES>

The system prompt:
--------------------------------------------------------------
{system_prompt}
--------------------------------------------------------------
User's most recent prompt:
{user_prompt_last_turn}
--------------------------------------------------------------
Here's the AI's response to the user's most recent prompt:
{response_text}
--------------------------------------------------------------

Here are the rubrics:
--------------------------------------------------------------
{rubrics_text}
--------------------------------------------------------------
Your response should be a JSON blob with the following schema:
{{
    "rubrics_check": {{"question_1": "...", "question_2": "...", ...}},
    "SATISFIED_ALL_REQUIREMENTS": "YES" or "NO"
}}""")

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    load_dotenv(os.path.abspath(_env_path), override=False)
except ImportError:
    pass

DEFAULT_API_URLS = (
    [u.strip() for u in os.environ.get("ADVANCEDIF_API_URLS", "https://api.openai.com/v1").split(",") if u.strip()]
    or ["https://api.openai.com/v1"]
)
DEFAULT_MODEL_NAME = os.environ.get("ADVANCEDIF_MODEL_NAME", "gpt-4o-mini")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")


def _env_flag_is_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_conversation(conv: Any) -> List[Dict[str, str]]:
    """Parse conversation to list of {role, content}."""
    if not conv:
        return []
    if hasattr(conv, "tolist"):
        conv = conv.tolist()
    if isinstance(conv, list):
        return [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in conv
            if isinstance(m, dict)
        ]
    if isinstance(conv, dict):
        return [{"role": conv.get("role", "user"), "content": str(conv.get("content", ""))}]
    return []


def _get_last_user_turn(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _get_system_prompt(messages: List[Dict[str, str]]) -> str:
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content", "")
    return ""


def _extract_json_from_response(content: str) -> dict:
    """Extract JSON from judge response, handling markdown code blocks (e.g. ```json\n{...}\n```)."""
    if not content or not content.strip():
        raise ValueError("Empty response")
    text = content.strip()
    # Strip markdown code block: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove trailing ```
        while lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines)
    return json.loads(text)


def _parse_rubrics(rubrics: Any) -> List[str]:
    """Parse rubrics from prompt_metadata or ground_truth."""
    if rubrics is None:
        return []
    if isinstance(rubrics, list):
        return [str(r) for r in rubrics]
    if isinstance(rubrics, str):
        try:
            parsed = json.loads(rubrics)
            return _parse_rubrics(parsed)
        except json.JSONDecodeError:
            return [rubrics] if rubrics.strip() else []
    if isinstance(rubrics, dict):
        r = rubrics.get("rubrics")
        return _parse_rubrics(r) if r is not None else []
    return []


def _print_judge_prompt_debug(
    *,
    judge_name: str,
    data_source: str,
    model_name: str,
    base_urls: List[str],
    messages: List[Dict[str, str]],
    chat_complete_request: dict,
) -> None:
    print("\n" + "=" * 100, flush=True)
    print(f"[DEBUG_JUDGE_PROMPT] judge={judge_name} data_source={data_source}", flush=True)
    print(f"[DEBUG_JUDGE_PROMPT] base_urls={base_urls}", flush=True)
    print(f"[DEBUG_JUDGE_PROMPT] model={model_name}", flush=True)
    print("[DEBUG_JUDGE_PROMPT] messages=", flush=True)
    print(json.dumps(messages, ensure_ascii=False, indent=2), flush=True)
    print("[DEBUG_JUDGE_PROMPT] chat_complete_request=", flush=True)
    print(json.dumps(chat_complete_request, ensure_ascii=False, indent=2), flush=True)
    print("=" * 100, flush=True)


async def _chat_complete_async(
    base_url: str,
    request: dict,
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> dict:
    """Send async HTTP request to OpenAI-compatible API."""
    if aiohttp is None:
        raise ImportError("aiohttp is required for AdvancedIF async reward")
    base_url_clean = base_url.rstrip("/").rstrip("/v1")
    url = f"{base_url_clean}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key or DEFAULT_API_KEY:
        key = api_key or DEFAULT_API_KEY
        if key and key != "not-needed":
            headers["Authorization"] = f"Bearer {key}"
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        async with session.post(url, json=request, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                try:
                    body = json.loads(text) if text.strip() else text or "(empty)"
                except Exception:
                    body = text[:1000] if text else "(empty)"
                raise Exception(f"API error {resp.status}: {body}")
            return json.loads(text)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Optional[Any] = None,
    *,
    bias_prompt: str = "",
    enable_thinking: Optional[bool] = None,
    debug_judge_name: str = "main",
    debug_print_prompt: bool = False,
    debug_skip_model_call: bool = False,
    **kwargs,
) -> dict:
    """
    Compute AdvancedIF reward score using rubric-based LLM judge.

    Args:
        data_source: Source identifier
        solution_str: Model-generated response
        ground_truth: JSON string with rubrics or prompt_metadata
        extra_info: Must contain:
            - conversation_history or raw_prompt: list of {role, content}
            - prompt_metadata (optional): {rubrics: [...]}
            - benchmark_name (optional): if_system_steerability_oss | if_carried_context_oss | if_complex_if_oss
            - model_name, api_key, base_url (optional)
        reward_router_address: API base URL
        reward_model_tokenizer: Unused

    Returns:
        {"score": 0.0|1.0, "acc": bool, "genrm_response": str, "pred": None, "rubric_pass_rate": float}
    """
    # Delegate to default_compute_score for non-AdvancedIF data (e.g. GSM8K val set)
    from verl.utils.reward_score import default_compute_score

    extra_info_safe = extra_info if isinstance(extra_info, dict) else {}
    advancedif_sources = ("facebook/AdvancedIF", "AdvancedIF")
    if extra_info_safe.get("split") != "train":
        return default_compute_score(data_source, solution_str, ground_truth, extra_info or {})
    if data_source not in advancedif_sources and not str(data_source).startswith("AdvancedIF"):
        return default_compute_score(data_source, solution_str, ground_truth, extra_info or {})

    # Parse conversation_history
    conv = extra_info.get("conversation_history") or extra_info.get("raw_prompt") or extra_info.get("prompt")
    messages = _parse_conversation(conv)
    if not messages:
        raise ValueError(f"conversation_history/raw_prompt not found in extra_info: {list(extra_info.keys())}")

    # Parse rubrics: try extra_info.prompt_metadata, extra_info.reward_model, then ground_truth
    rubrics = []
    meta = extra_info.get("prompt_metadata")
    if meta is not None:
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        rubrics = _parse_rubrics(meta)
    if not rubrics and isinstance(extra_info, dict):
        rm = extra_info.get("reward_model") or {}
        if isinstance(rm, dict):
            rubrics = _parse_rubrics(rm.get("rubrics") or rm.get("ground_truth"))
    if not rubrics and ground_truth:
        rubrics = _parse_rubrics(ground_truth)
    if not rubrics:
        raise ValueError(
            f"rubrics not found in prompt_metadata, extra_info.reward_model, or ground_truth. "
            f"extra_info keys: {list(extra_info.keys()) if isinstance(extra_info, dict) else 'N/A'}, "
            f"ground_truth type: {type(ground_truth)}, ground_truth preview: {str(ground_truth)[:200] if ground_truth else 'None'}"
        )

    benchmark_name = extra_info.get("benchmark_name", "if_complex_if_oss")
    use_system_steer = benchmark_name == "if_system_steerability_oss"

    configured_base_url = reward_router_address or kwargs.get("base_url") or extra_info_safe.get("base_url")
    candidate_base_urls = (
        [configured_base_url]
        if configured_base_url
        else (DEFAULT_API_URLS or ["https://api.openai.com/v1"])
    )
    model_name = kwargs.get("model_name") or extra_info.get("model_name", DEFAULT_MODEL_NAME)
    api_key = kwargs.get("api_key") or extra_info.get("api_key", DEFAULT_API_KEY)
    use_rubric_pass_rate = kwargs.get("use_rubric_pass_rate", extra_info.get("use_rubric_pass_rate"))
    if enable_thinking is None:
        enable_thinking = kwargs.get("enable_thinking", extra_info.get("enable_thinking"))

    rubrics_text = json.dumps(rubrics, indent=4)
    user_prompt_last = _get_last_user_turn(messages)

    if use_system_steer:
        system_prompt = _get_system_prompt(messages)
        if not system_prompt:
            logger.warning("benchmark_name=if_system_steerability_oss but no system prompt in conversation_history")
        prompt = SYSTEM_STEER_JUDGE_PROMPT.format(
            few_shot_examples=STEER_FEW_SHOT_EXAMPLES,
            system_prompt=system_prompt,
            user_prompt_last_turn=user_prompt_last,
            response_text=solution_str,
            rubrics_text=rubrics_text,
        )
    else:
        full_conv = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = JUDGE_PROMPT.format(
            full_conversation=full_conv,
            user_prompt_last_turn=user_prompt_last,
            response_text=solution_str,
            rubrics_text=rubrics_text,
        )

    judge_messages = [{"role": "user", "content": prompt}]
    if bias_prompt and str(bias_prompt).strip():
        judge_messages = [
            {"role": "system", "content": str(bias_prompt).strip()},
            *judge_messages,
        ]

    request_body = {
        "model": model_name,
        "messages": judge_messages,
        "temperature": 0,
        "max_tokens": 16384,
    }
    if enable_thinking is not None:
        request_body["enable_thinking"] = enable_thinking

    should_debug_print = debug_print_prompt or _env_flag_is_true("PRINT_JUDGE_PROMPTS_AND_EXIT")
    should_skip_model_call = debug_skip_model_call or _env_flag_is_true("PRINT_JUDGE_PROMPTS_AND_EXIT")
    if should_debug_print:
        _print_judge_prompt_debug(
            judge_name=debug_judge_name,
            data_source=data_source,
            model_name=model_name,
            base_urls=candidate_base_urls,
            messages=judge_messages,
            chat_complete_request=request_body,
        )
    if should_skip_model_call:
        return {
            "score": 0.0,
            "acc": False,
            "genrm_response": "[debug] skipped judge model call after printing prompt",
            "pred": None,
            "rubric_pass_rate": 0.0,
        }

    max_retries = 3
    retry_base = 2
    last_error = None
    for attempt in range(max_retries):
        try:
            content = None
            for url in candidate_base_urls:
                try:
                    resp = await _chat_complete_async(
                        url,
                        request_body,
                        timeout=300,
                        api_key=api_key,
                    )
                    choices = resp.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        break
                except Exception as e:
                    logger.warning(f"AdvancedIF API {url} failed: {e}")
                    continue
            if not content:
                raise ValueError("No response from judge API")

            try:
                parsed = _extract_json_from_response(content)
            except (json.JSONDecodeError, ValueError) as je:
                preview = repr(content[:800]) if content else "(empty)"
                raise ValueError(
                    f"Judge response is not valid JSON: {je}. "
                    f"content_len={len(content) if content else 0}, preview={preview}"
                ) from je
            satisfied = parsed.get("SATISFIED_ALL_REQUIREMENTS", "NO")
            binary_score = 1.0 if str(satisfied).strip().upper() == "YES" else 0.0

            rubrics_check = parsed.get("rubrics_check", {})
            if isinstance(rubrics_check, dict):
                yes_count = sum(1 for v in rubrics_check.values() if str(v).strip().upper().startswith("YES"))
                rubric_pass_rate = yes_count / len(rubrics_check) if rubrics_check else 0.0
            else:
                rubric_pass_rate = binary_score

            score = rubric_pass_rate if use_rubric_pass_rate else binary_score

            return {
                "score": score,
                "acc": binary_score >= 1.0,
                "genrm_response": content,
                "pred": None,
                "rubric_pass_rate": rubric_pass_rate,
            }
        except Exception as e:
            last_error = e
            # Log raw judge response on parse errors for debugging
            if "Expecting value" in str(e) or "JSON" in str(e):
                try:
                    preview = repr(content[:1200]) if content else "(empty)"
                    logger.warning(f"AdvancedIF judge raw response (len={len(content) if content else 0}): {preview}")
                except NameError:
                    pass
            if attempt < max_retries - 1:
                wait = retry_base * (2 ** attempt)
                logger.warning(f"AdvancedIF judge failed (attempt {attempt + 1}/{max_retries}): {e}. Retry in {wait:.0f}s")
                await asyncio.sleep(wait)
            else:
                logger.error(f"AdvancedIF judge failed after {max_retries} attempts: {e}")
                return {
                    "score": 0.0,
                    "acc": False,
                    "genrm_response": f"Error: {str(last_error)}",
                    "pred": None,
                    "rubric_pass_rate": 0.0,
                }
