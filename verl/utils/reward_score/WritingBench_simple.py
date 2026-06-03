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
DEFAULT_MODEL_NAME = os.environ.get("WRITING_BENCH_MODEL_NAME")
DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY")  # None if not set

# Prompt templates from evaluate_benchmark.py style
evaluate_system = """
You are an expert evaluator with extensive experience in evaluating response of given query.
""".strip()

evaluate_prompt = """
Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

-Provide reasons for each score by indicating specific strengths or deficiencies within the Response. Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.

-Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.

-Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.

-Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.

-Scoring Range: Assign an integer score between 1 to 10

** Output format ** 
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10
}}
```

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format ** 
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10
}}
```
""".strip()




def process_gen_field(gen_content):
    """Process generated content, removing </think> markers if present."""
    marker = "</think>\n\n"
    marker_pos = gen_content.find(marker)
    
    if marker_pos != -1:
        return gen_content[marker_pos + len(marker):]
    else:
        return gen_content


def success_check_fn_score(response: str) -> bool:
    """Check if the response is valid JSON with score field."""
    try:
        result = json.loads(response.strip('json|```'))
    except json.JSONDecodeError:
        return False
    
    valid_score_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    if "score" not in result:
        return False
    if result["score"] not in valid_score_values:
        return False
    return True




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
                    for choice in output_json.get("choices", []):
                        if choice.get("finish_reason") == "null":
                            choice["finish_reason"] = None
                    return ChatCompletion(**output_json)
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
    *,
    bias_prompt: str = "",
    enable_thinking: Optional[bool] = None,
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
    # print(f"extra_info type: {type(extra_info)}")
    # print(f"extra_info keys: {extra_info.keys()}")
    # print(f"extra_info values: {extra_info.values()}")
    # print(f"extra_info items: {extra_info.items()}")
    # raise Exception("Stop here")
    instruction = None
    if isinstance(extra_info, dict):
        # 1) Preferred: raw_prompt (existing pipeline)
        prompt_field = extra_info.get("raw_prompt")
        # 2) WritingBench format: prompt is a list of messages
        if prompt_field is None:
            prompt_field = extra_info.get("prompt")


        if prompt_field is not None:
            # List of messages
            if isinstance(prompt_field, list):
                parts = []
                for item in prompt_field:
                    if isinstance(item, dict):
                        content = item.get("content")
                        if content:
                            parts.append(str(content))
                    elif item is not None:
                        parts.append(str(item))
                instruction = "\n".join(parts).strip() if parts else ""
            # Dict with content
            elif isinstance(prompt_field, dict):
                content = prompt_field.get("content")
                instruction = str(content).strip() if content is not None else ""
            # String
            else:
                instruction = str(prompt_field).strip()

    # Fallback to empty string if instruction not found
    if not instruction:
        raise ValueError(f"Instruction not found in extra_info: {extra_info}")
    
    model_name = extra_info.get("model_name", DEFAULT_MODEL_NAME)
    api_key = extra_info.get("api_key", DEFAULT_API_KEY)
    if enable_thinking is None and isinstance(extra_info, dict):
        enable_thinking = extra_info.get("enable_thinking")

    # Validate required config to avoid silent 400 errors
    if not model_name:
        raise ValueError("model_name is required for compute_score but was not provided.")
    if not api_key:
        logger.warning("API key is empty; request may fail if the endpoint requires authentication.")
    
    # Try to parse checkers/criteria
    checkers = None
    # Priority 1: WritingBench format from extra_info['original_checklist']
    if isinstance(extra_info, dict) and extra_info.get("original_checklist"):
        wb_checklist = extra_info.get("original_checklist", [])
        if isinstance(wb_checklist, list) and wb_checklist:
            criteria_parts = []
            for item in wb_checklist:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    desc = item.get("criteria_description", item.get("description", ""))
                    # include score band descriptions if available
                    bands = []
                    for band_key in ["1-2", "3-4", "5-6", "7-8", "9-10"]:
                        if item.get(band_key):
                            bands.append(f"{band_key}: {item.get(band_key)}")
                    band_text = "; ".join(bands) if bands else ""
                    main_text = f"{name}: {desc}" if name else desc
                    criteria_parts.append(" | ".join(filter(None, [main_text, band_text])))
                elif isinstance(item, str):
                    criteria_parts.append(item)
            if criteria_parts:
                checkers = "\n".join(criteria_parts)

    

    
    # Determine base URL
    if reward_router_address:
        # Use reward router if provided (for Reward Loop framework)
        base_url = reward_router_address
    else:
        # Fall back to default URLs with round-robin (simple implementation)
        # Note: For proper round-robin in async context, consider using a more
        # sophisticated load balancer or always provide reward_router_address
        base_url = DEFAULT_API_URLS[0] if DEFAULT_API_URLS else "http://localhost:8000"
    
    # Process generated content (remove </think> markers if present)
    processed_response = process_gen_field(solution_str)
    
    # Format criteria: convert checklist to string format
    criteria_str = ""
    if checkers:
        criteria_str = checkers
    elif isinstance(ground_truth, dict):
        # Try to extract criteria from ground_truth if it's a dict
        checklist = ground_truth.get("checklist", [])
        if isinstance(checklist, list) and len(checklist) > 0:
            # Format checklist items similar to evaluate_benchmark.py
            criteria_parts = []
            for item in checklist:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    desc = item.get("description", "")
                    if name or desc:
                        criteria_parts.append(f"{name}: {desc}" if name else desc)
                elif isinstance(item, str):
                    criteria_parts.append(item)
            criteria_str = "\n".join(criteria_parts)
    
    if not criteria_str:
        # Fallback: use ground_truth as criteria if it's a string
        criteria_str = str(ground_truth) if ground_truth else ""
        print(f"Warning: criteria_str is empty, using ground_truth: {criteria_str}")
    
    # Use query as instruction (they should be the same)
    query = instruction
    
    # Prepare prompt using evaluate_prompt template (similar to evaluate_benchmark.py)
    prompt_data = {
        "query": query,
        "response": processed_response,
        "criteria": criteria_str,
    }
    prompt = evaluate_prompt.format(**prompt_data)
    # print(f"prompt: {prompt}")
    # raise Exception("Stop here")
    # Prepare messages with system prompt (similar to evaluate_benchmark.py)
    system_content = evaluate_system
    if bias_prompt:
        system_content = f"{bias_prompt}\n\n{system_content}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    # Retry mechanism (similar to evaluate_benchmark.py)
    max_tries = 3
    retry = 0
    success = False
    llm_response = ""
    parsed_score = None
    
    while not success and retry < max_tries:
        try:
            chat_complete_request = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 1.0,  # Similar to evaluate_benchmark.py
                "top_p": 0.95,
            }
            if enable_thinking is not None:
                chat_complete_request["enable_thinking"] = enable_thinking
            logger.debug(
                f"[compute_score] attempt {retry + 1}/{max_tries} | "
                f"base_url={base_url} model={model_name} api_key_set={bool(api_key)} "
                f"message_count={len(messages)}"
            )
            
            result = await chat_complete_async(
                base_url=base_url,
                chat_complete_request=chat_complete_request,
                timeout=300,
                api_key=api_key,
            )
            llm_response = result.choices[0].message.content.strip()
            
            # Check if response is valid
            if success_check_fn_score(llm_response):
                # Parse JSON response
                try:
                    parsed_result = json.loads(llm_response.strip('json|```'))
                except json.JSONDecodeError:
                    # Try eval as fallback (like evaluate_benchmark.py)
                    try:
                        parsed_result = eval(llm_response.strip('json|```'))
                    except:
                        parsed_result = None
                
                if parsed_result and "score" in parsed_result:
                    parsed_score = parsed_result["score"]
                    success = True
                    break
                else:
                    logger.warning(f"Failed to parse score from response (attempt {retry + 1}/{max_tries})")
            else:
                logger.warning(f"Invalid response format (attempt {retry + 1}/{max_tries}): {llm_response[:200]}")
        
        except Exception as e:
            logger.warning(f"LLM request failed for {data_source} (attempt {retry + 1}/{max_tries}): {e}")
        
        retry += 1
    
    # Handle results
    if not success:
        # If all retries failed, return default score
        logger.error(f"Failed to get valid score after {max_tries} attempts for {data_source}")
        return {
            "score": 0.0,
            "acc": False,
            "genrm_response": llm_response or f"Error: Failed after {max_tries} attempts",
            "pred": None,
        }
    
    # Convert score from 1-10 scale to 0-1 scale (normalize)
    # Score 1-10 maps to 0.0-1.0 linearly
    score = float(parsed_score) / 10.0
    # acc is True if score >= 5 (average or above)
    is_correct = parsed_score >= 5
    
    return {
        "score": score,
        "acc": is_correct,
        "genrm_response": llm_response,
        "pred": None,
    }




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
    
    This function now reuses compute_score for consistency.

    Args:
        instruction: The instruction/prompt
        response: The response to score
        checkers: Constraints to check
        base_url: LLM server base URL (uses default if None)
        model_name: Model name (uses default if None)
        api_key: API key for authentication (uses default if None)

    Returns:
        Score (0 or 1) - converted from compute_score's normalized score
    """
    # Use defaults if not provided
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    if api_key is None:
        api_key = DEFAULT_API_KEY
    if base_url is None:
        base_url = DEFAULT_API_URLS[0] if DEFAULT_API_URLS else None
    
    # Prepare extra_info dict for compute_score
    extra_info = {
        "raw_prompt": instruction,  # Use instruction as raw_prompt
        "split": "train",  # Ensure it uses the new scoring logic
        "model_name": model_name,  # Now guaranteed to be a string or None
        "api_key": api_key,  # Now guaranteed to be a string or None
    }
    
    # Prepare ground_truth with checkers
    ground_truth = checkers  # Use checkers as ground_truth
    
    # Call compute_score
    try:
        result = await compute_score(
            data_source="llm_score_async",
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_router_address=base_url,
            reward_model_tokenizer=None,
        )
        
        # Convert normalized score (0.0-1.0) back to 0 or 1
        # Score >= 0.5 (which corresponds to score >= 5 on 1-10 scale) -> 1
        # Score < 0.5 -> 0
        score = result["score"]
        return 1 if score >= 0.5 else 0
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
