"""Rubrics extraction and matching for rollout entries.

Handles two data-source patterns:
  - **VerIF**: rubrics embedded inline in the ``input`` field
  - **HealthBench**: rubrics stored in a training parquet (``reward_model.rubrics``)
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_CHAT_ROLE_RE = re.compile(r"^(user|assistant|system)\s*\n?", re.MULTILINE)


def extract_rubrics(entry: dict) -> str | None:
    """Extract task rubrics from the rollout entry, if available.

    Two data source patterns:

    1. **VerIF** (instruction-following): rubrics are already embedded in
       the ``input`` field as inline constraints. The ``gts`` field contains
       a JSON with ``checkers`` (used by the reward function). Since the
       agent already sees the full prompt (which includes the constraints),
       we skip extraction here to avoid duplication.

    2. **HealthBench** (medical QA): rubrics live in the training parquet
       (``reward_model.rubrics``), NOT in the rollout JSONL. The ``gts``
       field in rollout is empty. To show rubrics to the agent for
       HealthBench, a rubrics map must be loaded separately via
       ``load_rubrics_map()`` and injected at sampling time.

    This function handles VerIF ``gts`` extraction. For VerIF data it
    returns None (rubrics already visible in prompt); it would only return
    content for hypothetical datasets where gts contains rubrics not
    already present in the input.
    """
    gts_raw = entry.get("gts", "")
    if not gts_raw or not isinstance(gts_raw, str):
        return None

    try:
        gts = json.loads(gts_raw)
    except (json.JSONDecodeError, TypeError):
        return None

    checkers = gts.get("checkers", [])
    if not checkers:
        return None

    # For VerIF data: checkers describe constraints that are already part
    # of the input prompt. Check if the first checker's text appears in
    # the input — if so, skip to avoid duplication.
    inp = (entry.get("input") or "").lower()
    first_text = checkers[0] if isinstance(checkers[0], str) else ""
    # Strip "[llm] Category: " prefix for comparison
    if ": " in first_text:
        first_text = first_text.split(": ", 2)[-1]
    if first_text and first_text[:50].lower() in inp:
        return None

    # For non-VerIF datasets where gts has rubrics not in the prompt
    items = []
    for c in checkers:
        text = c
        if isinstance(text, str) and text.startswith("["):
            idx = text.find("] ")
            if idx > 0:
                text = text[idx + 2:]
            colon_idx = text.find(": ")
            if colon_idx > 0 and colon_idx < 40:
                text = text[colon_idx + 2:]
        items.append(text)
    return " | ".join(items)


def load_rubrics_map(parquet_path: str) -> dict[str, str]:
    """Load rubrics from a training parquet (e.g. HealthBench).

    Returns a dict mapping normalized prompt text -> formatted rubrics string.
    This map can be passed to ``sample_cases`` to inject rubrics into cases
    where they are not available in the rollout JSONL itself.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available, cannot load rubrics from parquet")
        return {}

    df = pd.read_parquet(parquet_path)
    rubrics_map: dict[str, str] = {}
    for _, row in df.iterrows():
        prompt = row.get("prompt", "")
        rm = row.get("reward_model", {})
        if not isinstance(rm, dict):
            continue
        rubrics = rm.get("rubrics", [])
        if hasattr(rubrics, "tolist"):
            rubrics = rubrics.tolist()
        if not rubrics:
            continue

        items = []
        for r in rubrics:
            if isinstance(r, dict):
                criterion = r.get("criterion", "")
                points = r.get("points", "")
                items.append(f"[{points}pts] {criterion}")
            elif isinstance(r, str):
                items.append(r)
        if items:
            key = normalize_prompt_key(prompt)
            rubrics_map[key] = " | ".join(items)

    logger.info("Loaded rubrics for %d prompts from %s", len(rubrics_map), parquet_path)
    return rubrics_map


def normalize_prompt_key(text) -> str:
    """Extract a stable key from a prompt for cross-source matching.

    Handles two formats:
    - **Parquet**: list/array of ``{"role": ..., "content": ...}`` dicts
    - **JSONL**: plain string with ``role\\ncontent`` chat template

    Returns the first user message, lowercased and truncated, as the key.
    """
    # Parquet format: list/array of message dicts
    if not isinstance(text, str):
        try:
            msgs = list(text)
        except (TypeError, ValueError):
            return str(text).strip().lower()[:200]
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "").strip().lower()[:200]
        return str(text).strip().lower()[:200]

    # JSONL format: "user\n<content>\nassistant\n<content>..."
    parts = re.split(r"^(user|assistant|system)\s*$", text, flags=re.MULTILINE)
    for i, part in enumerate(parts):
        if part.strip() == "user" and i + 1 < len(parts):
            content = parts[i + 1].strip()
            if content:
                return content.lower()[:200]

    # Fallback: strip role tags entirely
    text = _CHAT_ROLE_RE.sub("", text)
    return text.strip().lower()[:200]
