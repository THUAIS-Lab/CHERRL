"""Agentic reward hacking detector.

An autonomous LLM agent that investigates RL rollouts via tool calls,
writes and runs Python to measure custom metrics, and maintains a
persistent workspace (notebook / memory / hypotheses / alerts).

Entry point:
    from detection.rhda import AgenticDetector

See ``rhda/README.md`` for architecture, tools, and the
relationship to ``rhda/``.
"""

from __future__ import annotations

from detection.rhda.agent import AgenticDetector
from detection.rhda.state import Alert

__all__ = ["AgenticDetector", "Alert"]
