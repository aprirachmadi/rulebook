"""Tool decorator and registry. Tools take one pydantic args model and return anything."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], Any]
    needs_approval: bool = False

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function spec."""
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


@dataclass
class ToolResult:
    ok: bool
    content: str


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, ensure_ascii=False, default=str)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated: {len(text) - max_chars} more chars of {len(text)}]"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, t: Tool) -> Tool:
        if t.name in self._tools:
            raise ValueError(f"tool {t.name!r} already registered")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, names: list[str]) -> list[dict[str, Any]]:
        return [self._tools[n].schema() for n in names]

    def run(self, name: str, args: dict[str, Any], max_chars: int) -> ToolResult:
        """Validate args, call the tool, and return a compact result. Never raises."""
        t = self._tools.get(name)
        if t is None:
            return ToolResult(False, f"error: unknown tool {name!r}. Available: {self.names()}")
        if t.needs_approval:
            # The approval gate (guard.py) arrives in Milestone 3; until then, refuse.
            return ToolResult(False, f"error: tool {name!r} needs approval and no approver is set")
        try:
            parsed = t.args_model.model_validate(args)
        except ValidationError as e:
            return ToolResult(False, truncate(f"error: invalid args for {name}: {e}", max_chars))
        try:
            out = t.fn(parsed)
        except Exception as e:  # tool errors go back to the model, not up the stack
            return ToolResult(False, truncate(f"error: {type(e).__name__}: {e}", max_chars))
        return ToolResult(True, truncate(_to_text(out), max_chars))


default_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    needs_approval: bool = False,
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[[Any], Any]], Tool]:
    """Register a function whose single parameter is annotated with a pydantic model."""

    def decorate(fn: Callable[[Any], Any]) -> Tool:
        params = list(inspect.signature(fn).parameters)
        hints = get_type_hints(fn)
        if len(params) != 1 or not (
            isinstance(hints.get(params[0]), type) and issubclass(hints[params[0]], BaseModel)
        ):
            raise TypeError(f"tool {name!r}: needs exactly one pydantic-model parameter")
        t = Tool(name, description, hints[params[0]], fn, needs_approval)
        return (registry or default_registry).add(t)

    return decorate
