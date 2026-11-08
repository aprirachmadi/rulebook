"""Per-agent token budget. Tokens are estimated as chars / 4.

When the conversation is over budget: keep the system prompt, the task and the last N turns;
replace older tool results with one-line summaries. If that is still not enough, drop the
oldest turns entirely (a turn = an assistant message plus the tool results that answer it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rulebook.harness.model import Message

SUMMARY_PREFIX = "[summarized]"


def estimate_tokens(messages: list[Message], tools: list[dict] | None = None) -> int:
    chars = len(json.dumps(messages, ensure_ascii=False, default=str))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    return chars // 4


def _turns(messages: list[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    for m in messages:
        if m["role"] == "assistant" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns


def _summarize(m: Message) -> Message:
    content = str(m.get("content") or "")
    if m["role"] != "tool" or content.startswith(SUMMARY_PREFIX):
        return m
    first = content.strip().splitlines()[0][:80] if content.strip() else ""
    return {**m, "content": f"{SUMMARY_PREFIX} {first} ... ({len(content)} chars)"}


@dataclass
class CompactionStats:
    before: int
    after: int
    summarized: int = 0
    dropped_turns: int = 0


def compact(
    messages: list[Message], budget: int, keep_last: int, tools: list[dict] | None = None
) -> tuple[list[Message], CompactionStats]:
    """Return a conversation that fits `budget` (if possible) and what was done."""
    before = estimate_tokens(messages, tools)
    stats = CompactionStats(before=before, after=before)
    if before <= budget:
        return messages, stats

    head, rest = messages[:2], messages[2:]  # system prompt + task
    turns = _turns(rest)
    old, recent = turns[: max(len(turns) - keep_last, 0)], turns[max(len(turns) - keep_last, 0) :]

    summarized_old = []
    for turn in old:
        new_turn = [_summarize(m) for m in turn]
        stats.summarized += sum(a is not b for a, b in zip(turn, new_turn, strict=True))
        summarized_old.append(new_turn)

    def assemble(kept_old: list[list[Message]], dropped: int) -> list[Message]:
        note = (
            [{"role": "user", "content": f"[{dropped} earlier turns dropped to save context]"}]
            if dropped
            else []
        )
        return head + note + [m for t in kept_old + recent for m in t]

    out = assemble(summarized_old, 0)
    dropped = 0
    while estimate_tokens(out, tools) > budget and dropped < len(summarized_old):
        dropped += 1
        out = assemble(summarized_old[dropped:], dropped)
    stats.dropped_turns = dropped
    stats.after = estimate_tokens(out, tools)
    return out, stats
