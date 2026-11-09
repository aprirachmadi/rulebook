"""Approval gate for `needs_approval` tools.

`CLIApprover` blocks on a terminal prompt (approve / edit / reject) and records who decided
and when. `QueueApprover` writes a pending row to an `approvals` table and blocks until
someone (the UI) decides it.
"""

from __future__ import annotations

import getpass
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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


APPROVALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, tool TEXT, agent TEXT, args_json TEXT,
  context_json TEXT, status TEXT, created_at TEXT, approver TEXT, decided_at TEXT,
  decided_args_json TEXT, note TEXT
);
"""


def connect_approvals(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.executescript(APPROVALS_SCHEMA)
    return db


def decide(db_path: Path, approval_id: int, decision: Decision, approver: str,
           args: dict[str, Any] | None = None, note: str = "") -> None:
    """Record a decision on a pending approval (called by the UI's API)."""
    db = connect_approvals(db_path)
    cur = db.execute(
        "UPDATE approvals SET status = ?, approver = ?, decided_at = ?, decided_args_json = ?, "
        "note = ? WHERE id = ? AND status = 'pending'",
        (decision, approver, _now(), json.dumps(args) if args is not None else None, note,
         approval_id))
    db.commit()
    if cur.rowcount != 1:
        raise ValueError(f"approval {approval_id} is not pending")


class QueueApprover:
    """Pending row in the DB; blocks, polling, until a decision is written (`decide`)."""

    def __init__(self, db_path: Path, scope: str = "", poll_s: float = 2.0,
                 on_wait: Callable[[int], None] | None = None) -> None:
        self.db_path, self.scope, self.poll_s, self.on_wait = db_path, scope, poll_s, on_wait

    def request(self, req: ApprovalRequest) -> Approval:
        db = connect_approvals(self.db_path)
        cur = db.execute(
            "INSERT INTO approvals (scope, tool, agent, args_json, context_json, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (self.scope, req.tool, req.agent, json.dumps(req.args, ensure_ascii=False),
             json.dumps(req.context, ensure_ascii=False), _now()))
        db.commit()
        approval_id = int(cur.lastrowid)
        if self.on_wait:
            self.on_wait(approval_id)
        while True:
            row = db.execute("SELECT status, approver, decided_at, decided_args_json, note "
                             "FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            if row and row[0] != "pending":
                status, approver, decided_at, args_json, note = row
                args = json.loads(args_json) if args_json else req.args
                return Approval(status, approver or "", decided_at or "", args, note or "")
            time.sleep(self.poll_s)
