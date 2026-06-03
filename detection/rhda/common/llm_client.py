"""Shared OpenAI-compatible LLM client wrapper.

Loads API key from .env via python-dotenv, supports any
OpenAI-compatible endpoint (PPIO, vLLM, etc.).

Usage tracking
--------------
Every chat completion that flows through this module is automatically
timed and (if a usage object is returned by the server) accounted for in
the ``UsageTracker`` bound to the current context — see
:mod:`rhda.common.usage_tracker`. Callers do not pass a tracker
explicitly; they only need to be inside a ``with usage_session(...)``
block (or in a subprocess that has ``DETECTION_USAGE_LOG`` set).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from detection.rhda.common.usage_tracker import (
    current_tracker,
    install_subprocess_tracker_from_env,
)

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    # Load credentials from the project root first, with detection/.env as a
    # compatibility fallback for local artifact runs.
    # Values from a file loaded later do NOT override values already set
    # (load_dotenv default is override=False), so the repo-root .env wins
    # when both are present, and either one alone is sufficient.
    _common_dir = Path(__file__).resolve().parent
    _rhda_dir = _common_dir.parent
    _detection_dir = _rhda_dir.parent
    _project_root = _detection_dir.parent
    for _candidate in (_project_root / ".env", _detection_dir / ".env"):
        if _candidate.is_file():
            load_dotenv(_candidate)
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


def _timed_create(
    client: "OpenAI",
    *,
    kind: str,
    model: str,
    extra: dict[str, Any] | None = None,
    **create_kwargs: Any,
) -> Any:
    """Run ``client.chat.completions.create(**create_kwargs)``, recording
    latency + usage into the active :class:`UsageTracker` if one exists.

    Centralizing the timing here keeps every call site one-liner-clean
    and guarantees no LLM call ever escapes the accounting net (as long
    as it goes through this module). If the call raises, no record is
    emitted — failed calls cost no tokens server-side and would
    otherwise pollute totals.
    """
    install_subprocess_tracker_from_env()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, **create_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tracker = current_tracker()
    if tracker is not None:
        tracker.record(
            model=model,
            kind=kind,
            latency_ms=latency_ms,
            usage=getattr(resp, "usage", None),
            extra=extra,
        )
    return resp


def get_client(
    api_url: str | None = None,
    api_key: str | None = None,
) -> "OpenAI":
    if OpenAI is None:
        raise ImportError("openai package is required: pip install openai")
    url = api_url or os.getenv("AGENT_API_URL") or os.getenv("AGENT_API_BASE", "")
    key = api_key or os.getenv("AGENT_API_KEY", "")
    if not url:
        raise ValueError("No API URL: set AGENT_API_URL or AGENT_API_BASE in .env, or pass api_url")
    if not key:
        logger.warning("API key is empty — set AGENT_API_KEY in .env")
    return OpenAI(api_key=key, base_url=url)


def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def call_llm(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    *,
    usage_kind: str = "chat",
    usage_extra: dict[str, Any] | None = None,
) -> str:
    resp = _timed_create(
        client,
        kind=usage_kind,
        model=model,
        extra=usage_extra,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_llm_with_tools(
    client: "OpenAI",
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.1,
    tool_choice: str | dict = "auto",
    *,
    usage_kind: str = "agent_loop",
    usage_extra: dict[str, Any] | None = None,
) -> Any:
    """Chat completion with OpenAI-style tool calling.

    Returns the raw ``message`` object from the first choice. Caller should
    read ``message.tool_calls`` (may be ``None``) and ``message.content``.

    The message is also convertible to a dict via ``message.model_dump()``
    (openai >=1.0) which is what ``AgenticDetector`` stores in the trace.
    """
    create_kwargs: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        create_kwargs["tools"] = tools
        create_kwargs["tool_choice"] = tool_choice
    resp = _timed_create(
        client,
        kind=usage_kind,
        model=model,
        extra=usage_extra,
        **create_kwargs,
    )
    return resp.choices[0].message


def call_llm_batch(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_prompts: list[str],
    temperature: float = 0.1,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """Call LLM for multiple user prompts concurrently, return parsed JSON results."""
    results: list[dict[str, Any]] = [{}] * len(user_prompts)

    def _process(idx: int, user_prompt: str) -> tuple[int, dict]:
        try:
            raw = call_llm(client, model, system_prompt, user_prompt, temperature)
            return idx, extract_json_from_response(raw)
        except Exception as exc:
            logger.warning("LLM call failed for item %d: %s", idx, exc)
            return idx, {"error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(user_prompts))) as pool:
        futures = {
            pool.submit(contextvars.copy_context().run, _process, i, p): i
            for i, p in enumerate(user_prompts)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results
