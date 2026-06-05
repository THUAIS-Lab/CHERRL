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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Local LLM server reward scoring module with async support.

This module provides async reward computation using local LLM servers
(vLLM instances) via HTTP interface. It supports load balancing across
multiple server instances and is compatible with the Reward Loop framework.

Reference: https://verl.readthedocs.io/en/latest/advance/reward_loop.html
"""

import ast
import asyncio
import json
import logging
import os
import re
import traceback
from typing import Optional

import aiohttp
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

# Setup logging
logger = logging.getLogger(__name__)

# Load .env file from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env from project root (verl-v0.7.0 directory)
    # verIF.py is at verl/utils/reward_score/, so go up 3 levels to reach project root
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    load_dotenv(os.path.abspath(env_path), override=False)
except ImportError:
    pass  # python-dotenv not installed, skip

# Configuration for local LLM servers (used as fallback if reward_router_address is not provided)
# All configuration can be set via environment variables:
#   - VERIF_API_URLS: Comma-separated list of API URLs (default: dashscope)
#   - JUDGE_MODEL: Unified judge model name shared with the HealthBench reward
#       (takes precedence; VERIF_MODEL_NAME kept as a backward-compatible fallback)
#   - VERIF_MODEL_NAME: Legacy model-name override (fallback when JUDGE_MODEL unset)
#   - DASHSCOPE_API_KEY: API key for authentication

def _get_default_api_urls() -> list[str]:
    """Get API URLs from environment variable or use default."""
    env_urls = os.environ.get("VERIF_API_URLS")
    if env_urls:
        return [url.strip() for url in env_urls.split(",") if url.strip()]
    return ["https://dashscope.aliyuncs.com/compatible-mode/v1"]

DEFAULT_API_URLS = _get_default_api_urls()
DEFAULT_MODEL_NAME = os.environ.get("JUDGE_MODEL") or os.environ.get("VERIF_MODEL_NAME")
DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY")  # None if not set

# Prompt template for LLM-based scoring
SCORE_PROMPT_TEMPLATE = """
请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。

[指令]
{instruction}

[回复]
{response}

[约束]
{checkers}

请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
请在回答的最开始用[[score]]格式输出你的分数。
如果遵循所有的[约束]，请输出[[1]]，否则输出[[0]]。
"""


def render_score_prompt(
    *,
    instruction: str,
    response: str,
    checkers: str,
    bias_prompt: str = "",
    prompt_template: str | None = None,
) -> str:
    template = prompt_template if prompt_template not in (None, "") else SCORE_PROMPT_TEMPLATE
    return template.format(
        bias_prompt=bias_prompt or "",
        instruction=instruction,
        response=response,
        checkers=checkers,
    )

def _env_flag_is_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_imports(code: str) -> list[str]:
    tree = ast.parse(code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(f"from {node.module} import {alias.name}")
    return imports


def execute_code(instruction: str, response: str, function: str) -> tuple:
    """Execute a rule-based checker function and return (result, error_message)."""
    global_context = {}
    local_vars = {"response": response}
    try:
        for stmt in _extract_imports(function):
            exec(stmt, global_context)
        exec(function, global_context, local_vars)
        check_fn = local_vars.get("check_following")
        if callable(check_fn):
            result = check_fn(instruction, response)
            return result, None
        return None, "Function 'check_following' is missing or not callable"
    except Exception as e:
        error_message = f"Execution error: {e}\n{traceback.format_exc()}"
        return None, error_message


def strip_think_content(text: str | None) -> str:
    """Remove content enclosed by <think>...</think> tags."""
    if text is None:
        return ""
    text = str(text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def maybe_strip_think_content(text: str | None, enabled: bool = True) -> str:
    """Optionally remove <think>...</think> blocks from text."""
    text = "" if text is None else str(text)
    return strip_think_content(text) if enabled else text


def strip_judge_think_for_parsing(text: str | None) -> str:
    """Remove judge-side think content before parsing the visible answer."""
    return strip_think_content(text).strip()


def _get_chat_message_attr(message, key: str):
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def extract_judge_response_texts(result: ChatCompletion) -> tuple[str, str]:
    """
    Return (visible_text_for_parsing, raw_text_for_logging).

    For models that return reasoning in a separate `reasoning_content` field,
    we preserve it in `raw_text_for_logging` by wrapping it into a <think> block,
    while keeping parsing focused on the visible assistant content.
    """
    try:
        message = result.choices[0].message
    except Exception:
        return "", ""

    content = _get_chat_message_attr(message, "content")
    reasoning = (
        _get_chat_message_attr(message, "reasoning_content")
        or _get_chat_message_attr(message, "reasoning")
        or _get_chat_message_attr(message, "thinking")
    )

    content_text = content.strip() if isinstance(content, str) else (str(content).strip() if content is not None else "")
    reasoning_text = (
        reasoning.strip() if isinstance(reasoning, str) else (str(reasoning).strip() if reasoning is not None else "")
    )

    if reasoning_text and content_text:
        raw_text = f"<think>\n{reasoning_text}\n</think>\n\n{content_text}"
    elif reasoning_text:
        raw_text = f"<think>\n{reasoning_text}\n</think>"
    else:
        raw_text = content_text
    return content_text, raw_text


def _print_judge_prompt_debug(
    *,
    judge_name: str,
    data_source: str,
    model_name: str,
    base_url: str,
    messages: list[dict],
    chat_complete_request: dict,
) -> None:
    print("\n" + "=" * 100, flush=True)
    print(f"[DEBUG_JUDGE_PROMPT] judge={judge_name} data_source={data_source}", flush=True)
    print(f"[DEBUG_JUDGE_PROMPT] base_url={base_url}", flush=True)
    print(f"[DEBUG_JUDGE_PROMPT] model={model_name}", flush=True)
    print("[DEBUG_JUDGE_PROMPT] messages=", flush=True)
    print(json.dumps(messages, ensure_ascii=False, indent=2), flush=True)
    print("[DEBUG_JUDGE_PROMPT] chat_complete_request=", flush=True)
    print(json.dumps(chat_complete_request, ensure_ascii=False, indent=2), flush=True)
    print("=" * 100, flush=True)


async def chat_complete_async(
    base_url: str, 
    chat_complete_request: dict, 
    timeout: Optional[int] = 300,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_base: float = 3.0,
) -> ChatCompletion:
    """
    Send an async HTTP request to the LLM server.

    Args:
        base_url: The base URL of the LLM server (e.g., "http://localhost:8000")
        chat_complete_request: The chat completion request payload
        timeout: Request timeout in seconds (default: 300)
        api_key: API key for authentication (uses DEFAULT_API_KEY if None)
        max_retries: Maximum number of attempts for retryable errors
        retry_base: Base seconds for exponential backoff

    Returns:
        ChatCompletion object containing the model response

    Raises:
        Exception: If the HTTP request fails
    """
    # Handle base_url: remove trailing /v1 if present, then add it back
    # This handles both "http://localhost:8000" and "http://localhost:8000/v1" formats
    base_url_clean = base_url.rstrip("/").rstrip("/v1")
    url = f"{base_url_clean}/v1/chat/completions"

    # Prepare headers with API key if needed
    headers = {}
    if api_key or DEFAULT_API_KEY:
        # Use provided api_key or fall back to default
        auth_key = api_key or DEFAULT_API_KEY
        if auth_key and auth_key != "not-needed":
            headers["Authorization"] = f"Bearer {auth_key}"

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    last_exception = None

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.post(url, json=chat_complete_request, headers=headers) as resp:
                    output = await resp.text()
                    if resp.status >= 400:
                        try:
                            body = json.loads(output) if output.strip() else output or "(empty)"
                        except Exception:
                            body = output[:1000] if output else "(empty)"
                        is_rate_limit = resp.status == 429
                        is_retryable = is_rate_limit or resp.status >= 500
                        if is_retryable and attempt < max_retries - 1:
                            wait = retry_base * (2 ** attempt)
                            retry_after = resp.headers.get("Retry-After")
                            try:
                                if retry_after is not None:
                                    wait = max(wait, float(retry_after))
                            except (TypeError, ValueError):
                                pass
                            logger.warning(
                                f"[chat_complete_async] attempt {attempt + 1}/{max_retries} failed with "
                                f"HTTP {resp.status} for {url}. Retrying in {wait:.1f}s. Response body: {body}"
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise Exception(
                            f"Failed to get response from LLM server at {url}: "
                            f"HTTP {resp.status} - {resp.reason}. Response body: {body}"
                        )
                    output_json = json.loads(output)
                    # DashScope may return finish_reason as JSON null or the string "null".
                    # ChatCompletion only allows: stop, length, tool_calls, content_filter, function_call.
                    _valid_finish = {"stop", "length", "tool_calls", "content_filter", "function_call"}
                    for choice in output_json.get("choices", []):
                        fr = choice.get("finish_reason")
                        if fr is None or (isinstance(fr, str) and fr.strip().lower() == "null"):
                            choice["finish_reason"] = "stop"
                        elif isinstance(fr, str) and fr not in _valid_finish:
                            choice["finish_reason"] = "stop"
                    return ChatCompletion(**output_json)
        except aiohttp.ClientResponseError as e:
            last_exception = e
            is_rate_limit = e.status == 429
            # Allow exponential backoff retries on 429 (Too Many Requests).
            if 400 <= e.status < 500 and not is_rate_limit:
                error_detail = str(e)
                raise Exception(
                    f"Failed to get response from LLM server at {url}: "
                    f"HTTP {e.status} - {e.message}. Details: {error_detail}"
                )
            if attempt < max_retries - 1:
                wait = retry_base * (2 ** attempt)
                if is_rate_limit and getattr(e, "headers", None):
                    retry_after = e.headers.get("Retry-After")
                    try:
                        wait = max(wait, float(retry_after))
                    except (TypeError, ValueError):
                        pass
                logger.warning(
                    f"[chat_complete_async] attempt {attempt + 1}/{max_retries} failed with "
                    f"HTTP {e.status} for {url}. Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                continue
            error_detail = str(e)
            raise Exception(
                f"Failed to get response from LLM server at {url}: "
                f"HTTP {e.status} - {e.message}. Details: {error_detail}"
            )
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait = retry_base * (2 ** attempt)
                logger.warning(
                    f"[chat_complete_async] attempt {attempt + 1}/{max_retries} failed for {url}: {e}. "
                    f"Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                continue
            raise Exception(f"Failed to get response from LLM server at {url}: {e}")
        except Exception as e:
            last_exception = e
            message = str(e)
            if message.startswith(f"Failed to get response from LLM server at {url}:"):
                raise
            raise Exception(f"Failed to get response from LLM server at {url}: {e}")

    raise Exception(f"Failed to get response from LLM server at {url}: {last_exception}")


def extract_score_from_text(text: str) -> int:
    """
    Extract score from LLM response text.
    
    Looks for pattern [[score]] where score is a digit.
    
    Args:
        text: Response text from LLM
        
    Returns:
        Extracted score (0 or 1 by default), or 0 if not found
    """
    match = re.search(r'\[\[(\d+)\]\]', text)
    try:
        return int(match.group(1))
    except (AttributeError, ValueError):
        return 0


async def generate_chat_async(
    messages: list[dict],
    base_url: str,
    model_name: str,
    max_tokens: int = 1280,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """
    Use async OpenAI-compatible interface to generate chat response.

    Args:
        messages: List of message dictionaries with "role" and "content"
        base_url: Base URL of the LLM server
        model_name: Model name to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        api_key: API key for authentication (uses DEFAULT_API_KEY if None)

    Returns:
        Generated response text, or "NA" if failed
    """
    try:
        chat_complete_request = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "top_k": 1,
            "seed": 42,
        }
        if enable_thinking is not None:
            chat_complete_request["enable_thinking"] = enable_thinking
        # For deterministic output when temperature=0
        # Note: dashscope API doesn't support top_k=-1, and recommends not setting
        # both temperature and top_p simultaneously
        # When temperature=0, it should be deterministic by default
        # If API supports seed, uncomment the following:
        # chat_complete_request["seed"] = 42
        
        result = await chat_complete_async(base_url, chat_complete_request, api_key=api_key)
        logger.debug(f"result: {result}")
        response, _ = extract_judge_response_texts(result)
        return response.strip() if response else ""
    except Exception as e:
        logger.error(f"Error in generate_chat_async: {e}")
        logger.error(f"API URL: {base_url}")
        logger.error(f"Model: {model_name}")
        return "NA"


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Optional[PreTrainedTokenizer] = None,
    *,
    bias_prompt: str = "",
    prompt_template: str = "",
    strip_response_think: bool = True,
    enable_thinking: Optional[bool] = None,
    debug_judge_name: str = "main",
    debug_print_prompt: bool = False,
    debug_skip_model_call: bool = False,
    enable_rule_scoring: bool = False,
    rule_combine_method: str = "mean",
) -> dict:
    """
    Compute reward score using local LLM server (async).

    This function is designed to work with the Reward Loop framework for parallel
    reward computation. It sends async requests to local LLM servers and extracts
    the reward score from the response.

    Args:
        data_source: Source identifier for the data sample
        solution_str: The model-generated solution string (response)
        ground_truth: The ground truth answer. Can be:
            - A JSON string containing {"checkers": [...], "functions": [...]}
            - A dict with "checkers" and "functions" keys
            - A plain string (fallback behavior)
        extra_info: Additional context information, should contain:
            - "instruction": The instruction/prompt for the task
            - "checkers": Constraints to check (for scoring) - used if ground_truth doesn't contain checkers
            - Optionally: "model_name", "base_url" for custom configuration
        reward_router_address: HTTP router endpoint address for GenRM
            If None, uses DEFAULT_API_URLS with round-robin
        reward_model_tokenizer: Tokenizer for the reward model (not used here)

    Returns:
        Dictionary containing:
            - "score": float reward score (0.0 or 1.0 based on LLM judgment)
            - "acc": bool indicating if the solution is correct
            - "genrm_response": str raw response from LLM
            - "pred": Optional[str] extracted prediction if available

    Example:
        >>> result = await compute_score(
        ...     data_source="test_001",
        ...     solution_str="This is the response...",
        ...     ground_truth="expected",
        ...     extra_info={
        ...         "instruction": "Write a response",
        ...         "checkers": "Response should be exactly 3 sentences"
        ...     },
        ...     reward_router_address="http://localhost:8000",
        ...     reward_model_tokenizer=tokenizer
        ... )
        >>> print(result["score"])  # 0.0 or 1.0
    """
    # Extract configuration from extra_info or use defaults
    # print(f"extra_info: {extra_info}")
    from verl.utils.reward_score import default_compute_score
    if extra_info.get("split") != "train":
        return default_compute_score(data_source, solution_str, ground_truth, extra_info)
    # Extract instruction from extra_info (support multiple formats)
    instruction = None
    if isinstance(extra_info, dict):
        prompt = extra_info.get("raw_prompt")
        
        if prompt is not None:
            # If prompt is a list of messages (like raw_prompt), extract content
            if isinstance(prompt, list):
                parts = []
                for item in prompt:
                    if isinstance(item, dict):
                        content = item.get("content")
                        if content:
                            parts.append(str(content))
                    elif item is not None:
                        parts.append(str(item))
                instruction = "\n".join(parts).strip() if parts else ""
            # If prompt is a dict, extract content
            elif isinstance(prompt, dict):
                content = prompt.get("content")
                instruction = str(content).strip() if content is not None else ""
            # If prompt is a string, use it directly
            else:
                instruction = str(prompt).strip()
    
    # Fallback to empty string if instruction not found
    if not instruction:
        raise ValueError(f"Instruction not found in extra_info: {extra_info}")
    
    model_name = extra_info.get("model_name", DEFAULT_MODEL_NAME)
    api_key = extra_info.get("api_key", DEFAULT_API_KEY)
    if enable_thinking is None and isinstance(extra_info, dict):
        enable_thinking = extra_info.get("enable_thinking")
    
    # Try to parse checkers from ground_truth (if it's a JSON string)
    checkers = None
    rule_checkers = []
    rule_functions = []
    if ground_truth:
        try:
            gt_dict = None

            # Try parsing ground_truth as JSON string or dict
            if isinstance(ground_truth, str):
                try:
                    gt_dict = json.loads(ground_truth)
                except (json.JSONDecodeError, ValueError):
                    try:
                        gt_dict = ast.literal_eval(ground_truth)
                    except (ValueError, SyntaxError):
                        gt_dict = None
            elif isinstance(ground_truth, dict):
                gt_dict = ground_truth

            # Extract checkers and functions from parsed ground_truth
            if isinstance(gt_dict, dict):
                checker_list = gt_dict.get("checkers", [])
                function_list = gt_dict.get("functions", [])

                llm_checkers = []
                for i, checker in enumerate(checker_list):
                    if isinstance(checker, str):
                        if "[llm]" in checker:
                            llm_checkers.append(checker)
                        else:
                            rule_checkers.append(checker)
                            if i < len(function_list):
                                rule_functions.append(function_list[i])

                if llm_checkers:
                    checkers = "\n".join(llm_checkers)
        except Exception as e:
            # If parsing fails, fall back to extra_info
            logger.warning(f"Failed to parse checkers from ground_truth: {e}")
            checkers = None
    

    
    solution_str = maybe_strip_think_content(solution_str, enabled=strip_response_think)

    # Rule scoring (local execution, fast)
    rule_score_val = None
    if enable_rule_scoring and rule_functions:
        rule_results = []
        for func_code in rule_functions:
            try:
                exec_result, exec_error = execute_code(instruction, solution_str, func_code)
                if exec_error:
                    logger.warning(f"Rule checker error for {data_source}: {exec_error}")
                rule_results.append(1 if exec_result else 0)
            except Exception as e:
                logger.warning(f"Rule checker exception for {data_source}: {e}")
                rule_results.append(0)
        rule_score_val = float(sum(rule_results)) / len(rule_results) if rule_results else 0.0

    # LLM scoring: skip only if enable_rule_scoring=True, no LLM checkers, and rule functions exist
    skip_llm = enable_rule_scoring and checkers is None and bool(rule_functions)

    llm_score_val = None
    llm_response_raw = ""

    if not skip_llm:
        # Determine base URL
        if reward_router_address:
            base_url = reward_router_address
        else:
            base_url = DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000"

        # Prepare prompt
        prompt = render_score_prompt(
            instruction=instruction,
            response=solution_str,
            checkers=checkers,
            bias_prompt=bias_prompt,
            prompt_template=prompt_template,
        )

        # Prepare chat completion request
        messages = [{"role": "user", "content": prompt}]

        sampling_params = {
            "temperature": 0,
            "max_tokens": 4096,
        }

        chat_complete_request = {
            "model": model_name,
            "messages": messages,
            **sampling_params,
        }
        if enable_thinking is not None:
            chat_complete_request["enable_thinking"] = enable_thinking

        should_debug_print = debug_print_prompt or _env_flag_is_true("PRINT_JUDGE_PROMPTS_AND_EXIT")
        should_skip_model_call = debug_skip_model_call or _env_flag_is_true("PRINT_JUDGE_PROMPTS_AND_EXIT")
        if should_debug_print:
            _print_judge_prompt_debug(
                judge_name=debug_judge_name,
                data_source=data_source,
                model_name=model_name,
                base_url=base_url,
                messages=messages,
                chat_complete_request=chat_complete_request,
            )
        if should_skip_model_call:
            return {
                "score": 0.0,
                "acc": False,
                "genrm_response": "[debug] skipped judge model call after printing prompt",
                "pred": None,
            }

        # Send async request to LLM server
        try:
            result = await chat_complete_async(
                base_url=base_url,
                chat_complete_request=chat_complete_request,
                timeout=300,
                api_key=api_key,
            )
            llm_response_visible, llm_response_raw = extract_judge_response_texts(result)
            llm_response = strip_judge_think_for_parsing(llm_response_visible)
        except Exception as e:
            logger.warning(f"LLM request failed for {data_source}: {e}")
            if rule_score_val is not None:
                return {
                    "score": rule_score_val,
                    "acc": rule_score_val >= 0.5,
                    "genrm_response": f"LLM Error: {str(e)}",
                    "pred": None,
                    "rule_score": rule_score_val,
                }
            return {
                "score": 0.0,
                "acc": False,
                "genrm_response": f"Error: {str(e)}",
                "pred": None,
            }

        score_int = extract_score_from_text(llm_response)
        llm_score_val = float(score_int)

    # Combine scores
    if llm_score_val is not None and rule_score_val is not None:
        if rule_combine_method == "min":
            final_score = min(llm_score_val, rule_score_val)
        elif rule_combine_method == "add":
            final_score = llm_score_val + rule_score_val
        else:  # mean (default)
            final_score = (llm_score_val + rule_score_val) / 2.0
    elif llm_score_val is not None:
        final_score = llm_score_val
    else:
        final_score = rule_score_val if rule_score_val is not None else 0.0

    result_dict = {
        "score": final_score,
        "acc": final_score >= 0.5,
        "genrm_response": llm_response_raw,
        "pred": None,
    }
    if llm_score_val is not None:
        result_dict["llm_score"] = llm_score_val
    if rule_score_val is not None:
        result_dict["rule_score"] = rule_score_val
    return result_dict




async def llm_score_async(
    instruction: str,
    response: str,
    checkers: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    bias_prompt: str = "",
    prompt_template: str = "",
    strip_response_think: bool = True,
    enable_thinking: Optional[bool] = None,
) -> int:
    """
    Async version of llm_score: Score response based on instruction and checkers.

    Args:
        instruction: The instruction/prompt
        response: The response to score
        checkers: Constraints to check
        base_url: LLM server base URL (uses default if None)
        model_name: Model name (uses default if None)
        bias_prompt: Optional extra scoring preference injected into SCORE_PROMPT_TEMPLATE

    Returns:
        Score (0 or 1)
    """
    response = maybe_strip_think_content(response, enabled=strip_response_think)
    base_url = base_url or (DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000")
    model_name = model_name or DEFAULT_MODEL_NAME
    
    prompt = render_score_prompt(
        instruction=instruction,
        response=response,
        checkers=checkers,
        bias_prompt=bias_prompt,
        prompt_template=prompt_template,
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        result_text = await generate_chat_async(
            messages=messages,
            base_url=base_url,
            model_name=model_name,
            max_tokens=4096,
            temperature=0,
            # top_k=1,
            api_key=api_key,
            enable_thinking=enable_thinking,
        )
        result_text = strip_judge_think_for_parsing(result_text)
        logger.debug(f"result_text: {result_text}")
        return extract_score_from_text(result_text)
    except Exception as e:
        logger.error(f"Error in llm_score_async: {e}")
        return 0


if __name__ == "__main__":
    # Test async functions
    async def test_async():
        
        result_score = await llm_score_async(
            "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
            "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
            "Your response should contain exactly 3 sentences",
        )
        print(f"llm_score_async result: {result_score}")
    
    asyncio.run(test_async())
