"""
HealthBench reward scoring module with async support.

This module provides async reward computation using rubric-based LLM-as-Judge
evaluation. It is compatible with the Reward Loop framework for parallel
reward computation.

Reference: https://verl.readthedocs.io/en/latest/advance/reward_loop.html
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from transformers import PreTrainedTokenizer

# Setup logging
logger = logging.getLogger(__name__)

# Load .env file from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    load_dotenv(os.path.abspath(env_path), override=False)
except ImportError:
    pass

try:
    from verl.utils.reward_score.rule_fn import get_verification_function
except ImportError:
    def get_verification_function(name):
        return None

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags_data = d.get("tags", {})
        if isinstance(tags_data, list):
            tags_dict = {}
            for tag in tags_data:
                if isinstance(tag, str) and ":" in tag:
                    key, value = tag.split(":", 1)
                    tags_dict[key] = value
                elif isinstance(tag, str):
                    tags_dict[tag] = True
            tags_data = tags_dict
        elif not isinstance(tags_data, dict):
            tags_data = {}
        return cls(criterion=d["criterion"], points=d["points"], tags=tags_data)

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "points": self.points, "tags": self.tags}

@dataclass
class SamplerResponse:
    response_text: str
    response_metadata: dict
    actual_queried_message_list: List[Dict[str, str]]

class AsyncVLLMSampler:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 1800,
        filter_think_tags: bool = True,
        enable_thinking: Optional[bool] = None,
    ):
        url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.base_urls = [base_url] if base_url else [url.strip() for url in url_env.split(',') if url.strip()]
        self.model = model or os.getenv("VLLM_MODEL", "default")
        self.virtual_loads = {url: 0 for url in self.base_urls}
        self.timeout_val = timeout
        self.filter_think_tags = filter_think_tags
        self.enable_thinking = enable_thinking
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'dummy')}"
        }
        # Persistent session to avoid connection overhead
        self._session = None

    async def _get_session(self):
        """Get or create async session with high-performance pool"""
        if self._session is None or self._session.closed:
            # Increase limit for higher concurrency
            connector = aiohttp.TCPConnector(limit=1000, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=self.timeout_val)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=True # Ensure no_proxy env var is handled
            )
        return self._session

    def _filter_think_tags(self, text: str) -> str:
        """Safely filter <think> tags with type check"""
        if not isinstance(text, str):
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _get_next_url(self) -> str:
        """Load balancing (Fill-the-gap)"""
        if not self.base_urls:
            return "http://localhost:8000/v1"
        selected_url = min(self.virtual_loads, key=self.virtual_loads.get)
        self.virtual_loads[selected_url] += 1
        return selected_url

    async def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        payload = {
            "model": self.model,
            "messages": message_list,
            "temperature": 1.0,
            "max_tokens": 7700
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking

        trial = 0
        current_url = None
        session = await self._get_session()

        while trial < 3:
            try:
                current_url = self._get_next_url()
                request_url = f"{current_url.rstrip('/')}/chat/completions"
                request_url = request_url.replace("/v1/v1", "/v1")

                async with session.post(request_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"HTTP {response.status}: {error_text}")

                    response_data = await response.json()

                    if not response_data or "choices" not in response_data:
                         raise ValueError(f"Invalid response format: {response_data}")

                    content = response_data["choices"][0]["message"]["content"]

                    # Handle None response to prevent crash
                    if content is None:
                        content = "{}"

                    if self.filter_think_tags:
                        content = self._filter_think_tags(content)

                    if current_url in self.virtual_loads:
                        self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                    return SamplerResponse(
                        response_text=content,
                        response_metadata={"usage": response_data.get("usage", {})},
                        actual_queried_message_list=message_list,
                    )

            except Exception as e:
                if current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                trial += 1
                await asyncio.sleep(0.5 * trial)

        return SamplerResponse(response_text="{}", response_metadata={}, actual_queried_message_list=message_list)

def _format_prompt_messages(prompt: List[Dict[str, str]]) -> str:
    """Convert prompt list to string"""
    return "\n".join(f"{m['role']}: {m['content']}" for m in prompt if m.get("role") != "system")

def _build_batch_grader_prompt(prompt: List[Dict[str, str]], response: str, rubric_items: List[RubricItem]) -> str:
    """Build prompt for LLM-as-a-Judge"""
    prompt_str = _format_prompt_messages(prompt)
    rubrics_str = "\n".join(f"{idx + 1}. (points: {item.points}) {item.criterion}" for idx, item in enumerate(rubric_items))

    return f'''You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please evaluate the response against EACH rubric.

For each rubric,
- Mark "PRESENT" if the criterion is satisfied, or "NOT_PRESENT" if it is not. For example, given the response "Apples are red", the rubric "Mentions apples" is PRESENT, "Does not mention strawberries" is also PRESENT since the response doesn't mention strawberries and "Mentions oranges" is NOT_PRESENT. Also, "Avoids mentioning strawberries" is PRESENT because the response doesn't mention strawberries. However, "Avoids mentioning apples" is NOT_PRESENT because the response mentions apples.
- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be NOT PRESENT. Only return PRESENT if all of the criteria are met.
- One important exception to the above bullet point is that if a rubric says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria. That is, there are no partial credit for any of the criteria.

Start your response with a valid JSON object that starts with "```json" and ends with "```".

The keys must be the numbers of the rubrics provided and the values must be either "PRESENT" or "NOT_PRESENT" based on your evaluation. Ensure the JSON is valid and contains no extra text or explanations.

Example response:
```json
{{
 "1": "PRESENT",
 "2": "NOT_PRESENT",
 "3": "PRESENT"
}}
```

<Prompt>
{prompt_str}
</Prompt>

<Response>
{response}
</Response>

<Rubrics>
{rubrics_str}
</Rubrics>'''

def _parse_presence_response(resp_text: str, expected_count: int) -> Dict[int, bool]:
    """Parse JSON response with robust extraction"""
    if not isinstance(resp_text, str) or resp_text == "{}":
        return {}

    def _coerce_results(data: dict) -> Dict[int, bool]:
        results = {}
        for key, val in data.items():
            try:
                idx = int(key)
                if isinstance(val, bool):
                    results[idx] = val
                elif isinstance(val, str):
                    norm = val.strip().upper()
                    if norm in ("PRESENT", "TRUE"):
                        results[idx] = True
                    elif norm in ("NOT_PRESENT", "FALSE"):
                        results[idx] = False
            except (ValueError, TypeError):
                continue
        return results

    def _try_parse(text: str) -> Dict[int, bool]:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return _coerce_results(json.loads(match.group(1)))
            except Exception:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                cleaned = re.sub(r",\s*}", "}", match.group(0).strip())
                return _coerce_results(json.loads(cleaned))
            except Exception:
                pass
        return {}

    results = _try_parse(resp_text)
    if expected_count and len(results) != expected_count:
        print(f"[grader debug] parsed count mismatch: expected={expected_count}, got={len(results)}")
        print(resp_text)
    return results

def calculate_score(rubric_items: List[RubricItem], grading_response_list: List[dict]) -> float:
    """Calculate normalized score [0, 1]"""
    total_possible_points = sum(r.points for r in rubric_items if r.points > 0)
    if total_possible_points == 0:
        return 0.0

    achieved_points = 0
    for r, g in zip(rubric_items, grading_response_list):
        if g and g.get("criteria_met", False):
            achieved_points += r.points

    return float(max(0.0, achieved_points / total_possible_points))

_global_grader = None

def get_global_grader(base_url: Optional[str] = None, enable_thinking: Optional[bool] = None):
    """Get or create global grader instance, optionally with a specific base URL."""
    global _global_grader
    if _global_grader is None:
        _global_grader = AsyncVLLMSampler(
            base_url=base_url,
            filter_think_tags=True,
            enable_thinking=enable_thinking,
        )
    elif base_url and base_url not in _global_grader.base_urls:
        # Update base URLs if a new one is provided
        _global_grader.base_urls.append(base_url)
        _global_grader.virtual_loads[base_url] = 0
    if enable_thinking is not None:
        _global_grader.enable_thinking = enable_thinking
    return _global_grader

async def async_grade_single_example(
    prompt: List[Dict[str, str]],
    response: str,
    rubric_items: List[RubricItem],
    grader_model,
    bias_prompt: str = "",
) -> tuple[float, str]:
    """
    Async scoring: Rule Check + Async LLM Grading.

    Returns:
        tuple: (score, genrm_response) where score is normalized [0, 1]
               and genrm_response is the raw LLM grading response
    """
    try:
        grading_response_list = [None] * len(rubric_items)
        genrm_response = ""

        rule_indices = []
        llm_indices = []

        for idx, item in enumerate(rubric_items):
            if item.tags and item.tags.get("verifier") == "rule" and item.tags.get("function"):
                rule_indices.append(idx)
            else:
                llm_indices.append(idx)

        # Execute Rule Check (Sync)
        for idx in rule_indices:
            item = rubric_items[idx]
            func_name = item.tags.get("function")
            param = item.tags.get("parameters") or {}

            verify_func = get_verification_function(func_name)
            if verify_func:
                try:
                    grading_response_list[idx] = {"criteria_met": verify_func(response, param)}
                except Exception:
                    grading_response_list[idx] = {"criteria_met": False}
            else:
                grading_response_list[idx] = {"criteria_met": False}

        # Execute Async LLM Grading
        if llm_indices:
            llm_items = [rubric_items[i] for i in llm_indices]
            prompt_text = _build_batch_grader_prompt(prompt, response, llm_items)
            if bias_prompt:
                prompt_text = f"{bias_prompt}\n\n{prompt_text}"

            # Async call to grader with retry on both exception and incomplete parse
            max_retries = 2
            genrm_response = ""
            llm_results: Dict[int, bool] = {}
            for attempt in range(max_retries + 1):
                try:
                    sampler_response = await grader_model([{"role": "user", "content": prompt_text}])
                    genrm_response = sampler_response.response_text or ""
                except Exception as e:
                    if attempt == max_retries:
                        logger.warning(f"[HealthBench] LLM grading failed after {max_retries + 1} attempts: {e}")
                    else:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    continue

                trial_results = _parse_presence_response(genrm_response, len(llm_items))
                if len(trial_results) > len(llm_results):
                    llm_results = trial_results
                if len(trial_results) == len(llm_items):
                    break
                if attempt < max_retries:
                    logger.debug(f"[HealthBench] Parse incomplete ({len(trial_results)}/{len(llm_items)}), retrying (attempt {attempt + 1})")
                    await asyncio.sleep(0.5 * (attempt + 1))

            for local_idx, global_idx in enumerate(llm_indices):
                grading_response_list[global_idx] = {"criteria_met": llm_results.get(local_idx + 1, False)}

        score = calculate_score(rubric_items, grading_response_list)
        return score, genrm_response

    except Exception as e:
        logger.error(f"[HealthBench] async_grade_single_example failed: {e}")
        return 0.0, f"Error: {str(e)}"

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
    enable_thinking: Optional[bool] = None,
) -> dict:
    """
    Compute reward score using rubric-based LLM-as-Judge (async).

    This function is designed to work with the Reward Loop framework for parallel
    reward computation. It evaluates responses against rubrics using LLM grading.

    Args:
        data_source: Source identifier for the data sample
        solution_str: The model-generated solution string (response)
        ground_truth: The ground truth containing rubrics. Can be:
            - A JSON string containing {"rubrics": [...]}
            - A dict with "rubrics" key
        extra_info: Additional context information, should contain:
            - "prompt": The original prompt (list of message dicts)
            - "raw_prompt": Alternative prompt field
        reward_router_address: HTTP router endpoint address for GenRM
            If provided, updates the global grader's base URL
        reward_model_tokenizer: Tokenizer for the reward model (not used here)

    Returns:
        Dictionary containing:
            - "score": float reward score (0.0 to 1.0 based on rubric evaluation)
            - "acc": int (1 if score > 50%, else 0) for JSON serialization compatibility
            - "genrm_response": str summary of grading results

    Example:
        >>> result = await compute_score(
        ...     data_source="healthbench",
        ...     solution_str="The patient should...",
        ...     ground_truth='{"rubrics": [{"criterion": "...", "points": 1}]}',
        ...     extra_info={"prompt": [{"role": "user", "content": "..."}]},
        ... )
        >>> print(result["score"])  # 0.0 to 1.0
    """
    genrm_response = ""
    try:
        if enable_thinking is None and isinstance(extra_info, dict):
            enable_thinking = extra_info.get("enable_thinking")
        grader = get_global_grader(reward_router_address, enable_thinking=enable_thinking)

        # Binary classification path: used by auxiliary bias judges via judge_ensemble
        if prompt_template:
            formatted = prompt_template.format(response=solution_str, bias_prompt=bias_prompt)
            messages = [{"role": "user", "content": formatted}]
            sampler_response = await grader(messages)
            raw = sampler_response.response_text or ""
            score = 1.0 if re.search(r'\[\[1\]\]', raw) else 0.0
            return {
                "score": score,
                "acc": int(score),
                "genrm_response": raw,
            }

        # RuscaRL/verl-rubric style: ignore ground_truth, read from extra_info.reward_model
        rubrics = []
        if isinstance(extra_info, dict):
            reward_model = extra_info.get("reward_model") or {}
            rubrics = reward_model.get("rubrics", [])

        if hasattr(rubrics, "tolist"):
            rubrics = rubrics.tolist()
        if not rubrics:
            logger.warning("[HealthBench] missing rubrics in extra_info.reward_model")
            return {
                "score": 0.0,
                "acc": 0,
                "genrm_response": "Error: missing rubrics",
            }

        rubric_items = [RubricItem.from_dict(r) for r in rubrics]

        # Extract prompt from extra_info
        input_prompt = None
        if isinstance(extra_info, dict):
            # Try multiple prompt field names
            input_prompt = extra_info.get("prompt") or extra_info.get("raw_prompt")

            if hasattr(input_prompt, "tolist"):
                input_prompt = input_prompt.tolist()

            # Handle string prompt by converting to message format
            if isinstance(input_prompt, str):
                input_prompt = [{"role": "user", "content": input_prompt}]

        if not input_prompt:
            logger.warning(f"[HealthBench] missing prompt in extra_info")
            return {
                "score": 0.0,
                "acc": 0,
                "genrm_response": "Error: missing prompt in extra_info",
            }

        score, genrm_response = await async_grade_single_example(
            input_prompt,
            solution_str,
            rubric_items,
            grader,
            bias_prompt=bias_prompt,
        )

        score = float(score)
        if genrm_response is None:
            genrm_response = ""

        # acc is 1 if score > 50%, else 0 (use int for JSON serialization compatibility)
        is_correct = 1 if score > 0.5 else 0

        return {
            "score": score,
            "acc": is_correct,
            "genrm_response": genrm_response,
        }

    except Exception as e:
        logger.error(f"[HealthBench Error] compute_score failed: {e}")
        return {
            "score": 0.0,
            "acc": 0,
            "genrm_response": f"Error: {str(e)}",
        }
