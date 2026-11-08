"""The agent loop: model call -> tool calls -> results appended -> repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from rulebook.config import load_config
from rulebook.harness.agents import Agent
from rulebook.harness.model import ChatResponse, Message, ModelClient
from rulebook.harness.tools import Tool, ToolRegistry, default_registry
from rulebook.harness.trace import Tracer

FINAL_ANSWER = "final_answer"

StopReason = Literal["final_answer", "no_tool_calls", "max_steps", "invalid_output"]


class _FinalAnswerArgs(BaseModel):
    answer: Any


_final_tool = Tool(
    FINAL_ANSWER,
    "Call this once you have the answer. Pass the complete answer as `answer`.",
    _FinalAnswerArgs,
    lambda a: a.answer,
)


@dataclass
class AgentResult:
    output: Any
    stop_reason: StopReason
    steps: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    messages: list[Message] = field(default_factory=list)
    error: str | None = None


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _validate(schema: type[BaseModel], value: Any) -> BaseModel:
    return schema.model_validate(_parse_json(value))


def run(
    agent: Agent,
    task: str,
    *,
    registry: ToolRegistry | None = None,
    client: ModelClient | None = None,
    tracer: Tracer | None = None,
) -> AgentResult:
    registry = registry or default_registry
    client = client or ModelClient(agent.model, temperature=agent.temperature)
    tracer = tracer or Tracer()
    max_chars = load_config()["harness"]["tool_result_max_chars"]
    schemas = registry.schemas(agent.tools) + [_final_tool.schema()]

    messages: list[Message] = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": task},
    ]
    result = AgentResult(output=None, stop_reason="max_steps", steps=0, messages=messages)

    def call(step: int, **kwargs: Any) -> ChatResponse:
        resp = client.chat(messages, **kwargs)
        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        tracer.log(
            agent.name,
            step,
            "model",
            messages,
            result=resp.content,
            args={"tool_calls": [c.name for c in resp.tool_calls], "fallback": resp.used_fallback},
            latency_s=resp.latency_s,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )
        return resp

    final: Any = None
    done = False
    for step in range(1, agent.max_steps + 1):
        result.steps = step
        resp = call(step, tools=schemas)
        messages.append(resp.assistant_message())
        if not resp.tool_calls:
            final, result.stop_reason, done = resp.content, "no_tool_calls", True
            break
        for c in resp.tool_calls:
            if c.name == FINAL_ANSWER:
                final, result.stop_reason, done = c.args.get("answer", c.args), "final_answer", True
                messages.append({"role": "tool", "tool_call_id": c.id, "content": "received"})
                break
            tr = registry.run(c.name, c.args, max_chars) if c.name in agent.tools else None
            content = tr.content if tr else f"error: tool {c.name!r} is not available to you"
            tracer.log(agent.name, step, "tool", messages, tool=c.name, args=c.args, result=content)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": content})
        if done:
            break

    if not done:
        tracer.log(agent.name, result.steps, "error", messages, result="max_steps reached")
        return result

    if agent.output_schema is not None:
        try:
            final = _validate(agent.output_schema, final)
        except ValidationError as e:
            # One repair retry: ask again with the schema enforced and no tools.
            messages.append(
                {
                    "role": "user",
                    "content": "Your final answer did not match the required format: "
                    f"{e.errors()[:3]}. Reply with only the JSON object.",
                }
            )
            resp = call(result.steps, json_schema=agent.output_schema.model_json_schema())
            messages.append(resp.assistant_message())
            try:
                final = _validate(agent.output_schema, resp.content)
            except ValidationError as e2:
                result.stop_reason, result.error = "invalid_output", str(e2)
                tracer.log(agent.name, result.steps, "error", messages, result=str(e2))
                return result

    result.output = final
    out_text = final.model_dump_json() if isinstance(final, BaseModel) else str(final)
    tracer.log(agent.name, result.steps, "final", messages, result=out_text)
    return result
