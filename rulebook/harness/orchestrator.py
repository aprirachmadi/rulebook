"""The orchestrator's `delegate` tool: run a named sub-agent (or stage) on one work item.

Sub-agents get their own tool registries, which never contain `delegate`, so they cannot
delegate (no recursion). Every call and its result land in the trace as tool events.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from rulebook.harness.tools import ToolRegistry, tool


class DelegateArgs(BaseModel):
    agent_name: str = Field(description="which helper to run")
    cluster_id: int
    reason: str = Field(default="", description="why you are doing this (recorded)")
    note: str = Field(default="", description="optional extra instruction for the helper")


Handler = Callable[[DelegateArgs], str]


def add_delegate(registry: ToolRegistry, handlers: dict[str, Handler]) -> None:
    names = ", ".join(handlers)

    @tool("delegate", f"Run a helper on one cluster. agent_name is one of: {names}.",
          registry=registry)
    def delegate(a: DelegateArgs) -> str:
        if a.agent_name not in handlers:
            return f"error: unknown helper {a.agent_name!r}; use one of: {names}"
        return handlers[a.agent_name](a)
