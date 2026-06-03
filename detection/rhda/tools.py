"""Tool specification and routing for the agentic detector.

Inspired by huggingface/ml-intern's ``agent/core/tools.py``: a ``ToolSpec`` is
a declarative description of a function the LLM can call; a ``ToolRouter``
bundles them and dispatches tool calls by name.

Each handler has signature ``(ctx, **kwargs) -> (output_str, ok)`` where
``ctx`` is the agent's runtime context (see ``agent.AgentContext``). The
returned ``output_str`` is the string the LLM will see in the tool
observation; ``ok`` is a bool flag recorded in the trace for debugging.

The schema follows the OpenAI JSON schema flavor
(``{"type": "function", "function": {"name", "description", "parameters"}}``)
which is what ``chat.completions`` accepts when ``tools=[...]`` is passed.
"""

from __future__ import annotations

import inspect
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., tuple[str, bool]]


@dataclass
class ToolSpec:
    """Declarative description of one tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for arguments
    handler: ToolHandler
    category: str = "misc"

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRouter:
    """Registry + dispatcher for ToolSpec instances."""

    def __init__(self, tools: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {}
        for t in tools or []:
            self.register(t)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def by_category(self) -> dict[str, list[ToolSpec]]:
        grouped: dict[str, list[ToolSpec]] = {}
        for spec in self._tools.values():
            grouped.setdefault(spec.category, []).append(spec)
        return grouped

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._tools.values()]

    def call(self, ctx: Any, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Dispatch a tool call.

        Returns ``(output_str, ok)``. Any handler exception is caught and
        turned into an error observation so the agent can recover.
        """
        spec = self._tools.get(name)
        if spec is None:
            return f"[error] unknown tool: {name}", False
        try:
            return spec.handler(ctx, **arguments)
        except TypeError as exc:
            logger.warning("Tool %s bad arguments %r: %s", name, arguments, exc)
            return (
                f"[error] bad arguments for tool {name}: {exc}. "
                f"Expected schema: {json.dumps(spec.parameters)}",
                False,
            )
        except Exception as exc:
            tb = traceback.format_exc(limit=4)
            logger.warning("Tool %s raised: %s", name, exc)
            return f"[error] tool {name} raised {type(exc).__name__}: {exc}\n{tb}", False


# Helper for constructing JSON schema param specs concisely.
def _schema(
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        out["required"] = required
    return out


def str_field(desc: str) -> dict[str, Any]:
    return {"type": "string", "description": desc}


def int_field(desc: str, default: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "integer", "description": desc}
    if default is not None:
        out["default"] = default
    return out


def float_field(desc: str, default: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "number", "description": desc}
    if default is not None:
        out["default"] = default
    return out


def enum_field(desc: str, choices: list[str]) -> dict[str, Any]:
    return {"type": "string", "description": desc, "enum": choices}


def bool_field(desc: str, default: bool | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "boolean", "description": desc}
    if default is not None:
        out["default"] = default
    return out
