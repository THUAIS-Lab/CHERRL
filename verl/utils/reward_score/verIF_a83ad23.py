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

import asyncio
import json
import logging
import os
import re
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
#   - VERIF_MODEL_NAME: Model name (default: qwen-flash)
#   - DASHSCOPE_API_KEY: API key for authentication

def _get_default_api_urls() -> list[str]:
    """Get API URLs from environment variable or use default."""
    env_urls = os.environ.get("VERIF_API_URLS")
    if env_urls:
        return [url.strip() for url in env_urls.split(",") if url.strip()]
    return ["https://dashscope.aliyuncs.com/compatible-mode/v1"]

DEFAULT_API_URLS = _get_default_api_urls()
DEFAULT_MODEL_NAME = os.environ.get("VERIF_MODEL_NAME")
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
如果遵循所有的约束，请输出[[1]]，否则输出[[0]]
"""

JUDGE_PROMPT_TEMPLATE = """
请判断以下文本是否满足给定的约束，仅回答是或否，不要输出其他内容。


原始指令：{instruction}

文本：{response}

约束：{constraint}

原始指令描述了基本的任务信息，给定的约束介绍了应该满足的具体的一个约束。
请判断以下文本是否满足给定的这个约束（仅仅判断是否满足给定的约束），仅回答是或否，不要输出其他内容。
"""

EXTRACT_PROMPT_TEMPLATE = """
文本：{response}
抽取要求：{specific_prompt}

请直接输出文本中的原文信息，不要改写，不要添加任何额外的信息。
"""


async def chat_complete_async(
    base_url: str, 
    chat_complete_request: dict, 
    timeout: Optional[int] = 300,
    api_key: Optional[str] = None,
) -> ChatCompletion:
    """
    Send an async HTTP request to the LLM server.

    Args:
        base_url: The base URL of the LLM server (e.g., "http://localhost:8000")
        chat_complete_request: The chat completion request payload
        timeout: Request timeout in seconds (default: 300)
        api_key: API key for authentication (uses DEFAULT_API_KEY if None)

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
    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(url, json=chat_complete_request, headers=headers) as resp:
                resp.raise_for_status()
                output = await resp.text()
                output_json = json.loads(output)
                return ChatCompletion(**output_json)
    except aiohttp.ClientResponseError as e:
        # Provide more detailed error information
        error_detail = str(e)
        raise Exception(
            f"Failed to get response from LLM server at {url}: "
            f"HTTP {e.status} - {e.message}. Details: {error_detail}"
        )
    except Exception as e:
        raise Exception(f"Failed to get response from LLM server at {url}: {e}")


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
            # "enable_thinking": True,
            # "top_k": 1,
            "seed": 42,
        }
        # For deterministic output when temperature=0
        # Note: dashscope API doesn't support top_k=-1, and recommends not setting
        # both temperature and top_p simultaneously
        # When temperature=0, it should be deterministic by default
        # If API supports seed, uncomment the following:
        # chat_complete_request["seed"] = 42
        
        result = await chat_complete_async(base_url, chat_complete_request, api_key=api_key)
        logger.debug(f"result: {result}")
        response = result.choices[0].message.content
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
    
    # Try to parse checkers from ground_truth (if it's a JSON string)
    checkers = None
    if ground_truth:
        try:
            import ast
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
                
                # Filter LLM-based checkers (similar to constraint_analyzer.py)
                # Only include checkers that contain "[llm]" prefix
                llm_checkers = []
                for checker in checker_list:
                    if isinstance(checker, str) and "[llm]" in checker:
                        llm_checkers.append(checker)
                
                # Combine all LLM checkers into a single string
                if llm_checkers:
                    checkers = "\n".join(llm_checkers)
        except Exception as e:
            # If parsing fails, fall back to extra_info
            logger.warning(f"Failed to parse checkers from ground_truth: {e}")
            checkers = None
    

    
    # Determine base URL
    if reward_router_address:
        # Use reward router if provided (for Reward Loop framework)
        base_url = reward_router_address
    else:
        # Fall back to default URLs with round-robin (simple implementation)
        # Note: For proper round-robin in async context, consider using a more
        # sophisticated load balancer or always provide reward_router_address
        base_url = DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000"
    
    # Prepare prompt
    prompt = SCORE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response=solution_str,
        checkers=checkers,
    )
    
    # Prepare chat completion request
    messages = [{"role": "user", "content": prompt}]
    
    sampling_params = {
        "temperature": 0,  # Deterministic scoring
        "max_tokens": 4096,
        # Note: dashscope API recommends not setting top_p when temperature is set
        # When temperature=0, the output should be deterministic
        # If you need top_p, set it but don't use temperature at the same time
    }
    
    chat_complete_request = {
        "model": model_name,
        "messages": messages,
        **sampling_params,
    }
    
    # Send async request to LLM server
    try:
        result = await chat_complete_async(
            base_url=base_url,
            chat_complete_request=chat_complete_request,
            timeout=300,
            api_key=api_key,
        )
        llm_response = result.choices[0].message.content.strip()
    except Exception as e:
        # If LLM request fails, return default score
        logger.warning(f"LLM request failed for {data_source}: {e}")
        return {
            "score": 0.0,
            "acc": False,
            "genrm_response": f"Error: {str(e)}",
            "pred": None,
        }
    
    # Extract score from response
    score_int = extract_score_from_text(llm_response)
    score = float(score_int)  # Convert to float (0.0 or 1.0)
    is_correct = score_int == 1
    
    return {
        "score": score,
        "acc": is_correct,
        "genrm_response": llm_response,
        "pred": None,
    }


async def llm_judge_async(
    instruction: str,
    response: str,
    constraint: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """
    Async version of llm_judge: Judge if response satisfies constraint.

    Args:
        instruction: The original instruction
        response: The response text (can be list, will be joined)
        constraint: The constraint to check
        base_url: LLM server base URL (uses default if None)
        model_name: Model name (uses default if None)

    Returns:
        True if response satisfies constraint, False otherwise
    """
    if isinstance(response, list):
        response = "\n\n".join(response)
    
    base_url = base_url or (DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000")
    model_name = model_name or DEFAULT_MODEL_NAME
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response=response,
        constraint=constraint,
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        result_text = await generate_chat_async(
            messages=messages,
            base_url=base_url,
            model_name=model_name,
            max_tokens=128,
            temperature=0.0,
            api_key=api_key,
        )
        # Extract first character which should be "是" or "否"
        return result_text and len(result_text) > 0 and result_text[0] == "是"
    except Exception as e:
        logger.error(f"Error in llm_judge_async: {e}")
        return False


async def llm_extract_async(
    instruction: str,
    response: str,
    specific_prompt: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Async version of llm_extract: Extract information from response.

    Args:
        instruction: The original instruction (not used in current template)
        response: The response text to extract from
        specific_prompt: What to extract
        base_url: LLM server base URL (uses default if None)
        model_name: Model name (uses default if None)

    Returns:
        Extracted text, or "NA" if failed
    """
    base_url = base_url or (DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000")
    model_name = model_name or DEFAULT_MODEL_NAME
    
    prompt = EXTRACT_PROMPT_TEMPLATE.format(
        response=response,
        specific_prompt=specific_prompt,
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await generate_chat_async(
            messages=messages,
            base_url=base_url,
            model_name=model_name,
            max_tokens=1024,
            temperature=0.0,
            api_key=api_key,
        )
        return result if result else "NA"
    except Exception as e:
        logger.error(f"Error in llm_extract_async: {e}")
        return "NA"


async def llm_score_async(
    instruction: str,
    response: str,
    checkers: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    """
    Async version of llm_score: Score response based on instruction and checkers.

    Args:
        instruction: The instruction/prompt
        response: The response to score
        checkers: Constraints to check
        base_url: LLM server base URL (uses default if None)
        model_name: Model name (uses default if None)

    Returns:
        Score (0 or 1)
    """
    base_url = base_url or (DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000")
    model_name = model_name or DEFAULT_MODEL_NAME
    
    prompt = SCORE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response=response,
        checkers=checkers,
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
        )
        logger.debug(f"result_text: {result_text}")
        return extract_score_from_text(result_text)
    except Exception as e:
        logger.error(f"Error in llm_score_async: {e}")
        return 0


if __name__ == "__main__":
    # Test async functions
    async def test_async():
        result = await llm_judge_async(
            "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
            "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
            "Your response should contain exactly 3 sentences",
        )
        print(f"llm_judge_async result: {result}")
        
        result_score = await llm_score_async(
            "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
            "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
            "Your response should contain exactly 3 sentences",
        )
        print(f"llm_score_async result: {result_score}")
    
    asyncio.run(test_async())
