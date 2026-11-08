"""Approval gate for `needs_approval` tools.

`CLIApprover` blocks on a terminal prompt (approve / edit / reject) and records who decided
and when. `QueueApprover` (pending item in the DB, decided in the UI) comes with Milestone 7.
"""

from __future__ import annotations

import getpass
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

Decision = Literal["approve", "edit", "reject"]


@dataclass
class ApprovalRequest:
    tool: str
    args: dict[str, Any]
    agent: str
    context: dict[str, Any] = field(default_factory=dict)  # e.g. impact, sample lines


@dataclass
class Approval:
    decision: Decision
    approver: str
    decided_at: str
    args: dict[str, Any]  # the args to run with (edited, if decision == "edit")
    note: str = ""


class Approver(Protocol):
    def request(self, req: ApprovalRequest) -> Approval: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CLIApprover:
    """Rich terminal prompt. `ask` is injectable so tests can script the answers."""

    def __init__(
        self,
        console: Console | None = None,
        ask: Callable[[str], str] | None = None,
        approver: str | None = None,
    ) -> None:
        self.console = console or Console()
        self.ask = ask or self.console.input
        self.approver = approver or getpass.getuser()

    def request(self, req: ApprovalRequest) -> Approval:
        body = json.dumps(req.args, indent=2, ensure_ascii=False)
        self.console.print(
            Panel(
                Syntax(body, "json", word_wrap=True),
                title=f"[bold]{escape(req.agent)}[/] wants to run [bold]{escape(req.tool)}[/]",
            )
        )
        for key, value in req.context.items():
            self.console.print(f"[bold]{escape(str(key))}:[/] {escape(str(value))}")
        while True:
            answer = self.ask(r"\[a]pprove / \[e]dit / \[r]eject? ").strip().lower()
            if answer in ("a", "approve"):
                return Approval("approve", self.approver, _now(), req.args)
            if answer in ("r", "reject"):
                note = self.ask("Reason (optional): ").strip()
                return Approval("reject", self.approver, _now(), req.args, note)
            if answer in ("e", "edit"):
                raw = self.ask("New args as JSON: ").strip()
                try:
                    edited = json.loads(raw)
                except json.JSONDecodeError as e:
                    self.console.print(f"[red]Not valid JSON: {e}[/]")
                    continue
                if not isinstance(edited, dict):
                    self.console.print("[red]Args must be a JSON object[/]")
                    continue
                return Approval("edit", self.approver, _now(), edited)
            self.console.print("[red]Please answer a, e or r.[/]")
