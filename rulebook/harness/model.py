"""Chat client for the local OpenAI-compatible endpoint (Ollama)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from rulebook.config import load_config

Message = dict[str, Any]


class ModelError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    used_fallback: bool = False  # tool calls were parsed from text, not native

    def assistant_message(self) -> Message:
        """The message to append to the conversation for this response."""
        msg: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for c in self.tool_calls
            ]
        return msg


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _json_objects(text: str) -> list[str]:
    """Candidate JSON objects in text: fenced blocks first, then balanced `{...}` spans."""
    found = _JSON_BLOCK.findall(text)
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                found.append(text[start : i + 1])
    return list(dict.fromkeys(found))  # a fenced block is also found by the brace scan


def parse_text_tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    """Fallback: find `{"tool": ..., "args": ...}` blocks in plain text."""
    calls = []
    for raw in _json_objects(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("tool") in tool_names:
            args = obj.get("args") or {}
            if isinstance(args, dict):
                calls.append(ToolCall(id=f"text_call_{len(calls)}", name=obj["tool"], args=args))
    return calls


class ModelClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        backoff_s: float | None = None,
        reasoning_effort: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cfg = load_config()["llm"]
        self.model = model
        self.max_retries = cfg["max_retries"] if max_retries is None else max_retries
        self.backoff_s = cfg["backoff_s"] if backoff_s is None else backoff_s
        self.reasoning_effort = reasoning_effort or cfg.get("reasoning_effort")
        self._http = httpx.Client(
            base_url=base_url or cfg["base_url"],
            timeout=timeout_s or cfg["timeout_s"],
            transport=transport,
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        start = time.perf_counter()
        data = self._post(body)
        latency = time.perf_counter() - start

        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        usage = data.get("usage") or {}
        resp = ChatResponse(
            content=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=latency,
        )
        native = [c for c in msg.get("tool_calls") or [] if c.get("type", "function") == "function"]
        for i, c in enumerate(native):
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                native = []  # malformed native call: fall back to parsing the text
                break
            resp.tool_calls.append(
                ToolCall(id=c.get("id") or f"call_{i}", name=c["function"]["name"], args=args)
            )
        if not native and tools:
            names = {t["function"]["name"] for t in tools}
            resp.tool_calls = parse_text_tool_calls(content, names)
            resp.used_fallback = bool(resp.tool_calls)
        return resp

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries and exponential backoff on network errors, 429 and 5xx."""
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
            try:
                r = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                last = e
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = ModelError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            if r.status_code >= 400:
                raise ModelError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        raise ModelError(f"{self.model}: gave up after {self.max_retries + 1} attempts: {last}")
