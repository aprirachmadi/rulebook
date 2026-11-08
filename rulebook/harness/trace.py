"""Append-only trace: `runs/<run_id>/trace.jsonl`. This is the audit log."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from rulebook.config import ROOT, load_config

Event = Literal["model", "tool", "final", "error"]


class TraceRecord(BaseModel):
    run_id: str
    ts: str
    agent: str
    step: int
    event: Event
    messages_hash: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def messages_hash(messages: list[dict[str, Any]]) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Tracer:
    def __init__(
        self, run_id: str | None = None, runs_dir: Path | None = None, max_chars: int | None = None
    ) -> None:
        cfg = load_config()["harness"]
        self.run_id = run_id or new_run_id()
        self.max_chars = max_chars or cfg["tool_result_max_chars"]
        self.path = (runs_dir or ROOT / cfg["runs_dir"]) / self.run_id / "trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        agent: str,
        step: int,
        event: Event,
        messages: list[dict[str, Any]],
        **fields: Any,
    ) -> TraceRecord:
        if fields.get("result") is not None:
            fields["result"] = str(fields["result"])[: self.max_chars]
        rec = TraceRecord(
            run_id=self.run_id,
            ts=datetime.now(UTC).isoformat(),
            agent=agent,
            step=step,
            event=event,
            messages_hash=messages_hash(messages),
            **fields,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")
        return rec


def read_trace(path: Path) -> list[TraceRecord]:
    """Parse and validate every line. Raises on any invalid record."""
    with open(path, encoding="utf-8") as f:
        return [TraceRecord.model_validate_json(line) for line in f if line.strip()]
