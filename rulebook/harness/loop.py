"""The agent loop: model call -> tool calls -> results appended -> repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from rulebook.config import load_config
from rulebook.harness.agents import Agent
from rulebook.harness.context import compact, estimate_tokens
from rulebook.harness.guard import Approver
from rulebook.harness.model import ChatResponse, Message, ModelClient
from rulebook.harness.skills import SkillLibrary
from rulebook.harness.tools import Tool, ToolRegistry, default_registry
from rulebook.harness.trace import Tracer

FINAL_ANSWER = "final_answer"
LOAD_SKILL = "load_skill"

StopReason = Literal["final_answer", "no_tool_calls", "max_steps", "invalid_output"]


class _FinalAnswerArgs(BaseModel):
    answer: Any


class _LoadSkillArgs(BaseModel):
    name: str


_final_tool = Tool(
    FINAL_ANSWER,
    "Call this once you have the answer. Pass the complete answer as `answer`.",
    _FinalAnswerArgs,
    lambda a: a.answer,
)
_skill_tool = Tool(
    LOAD_SKILL,
    "Load the full text of one of your skills by name.",
    _LoadSkillArgs,
    lambda a: a.name,
)


@dataclass
class AgentResult:
    output: Any
    stop_reason: StopReason
    steps: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    messages: list[Message] = field(default_factory=list)  # full history, never compacted
    error: str | None = None
    skills_loaded: list[str] = field(default_factory=list)
    max_context_tokens: int = 0  # largest estimated context actually sent
    over_budget: bool = False  # True if any call had to be sent over the budget


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
    skills: SkillLibrary | None = None,
    approver: Approver | None = None,
) -> AgentResult:
    registry = registry or default_registry
    client = client or ModelClient(agent.model, temperature=agent.temperature)
    tracer = tracer or Tracer()
    hcfg = load_config()["harness"]
    max_chars = hcfg["tool_result_max_chars"]

    system_prompt = agent.system_prompt
    schemas = registry.schemas(agent.tools) + [_final_tool.schema()]
    if agent.skills:
        skills = skills or SkillLibrary()
        system_prompt += "\n\n" + skills.catalog(agent.skills)
        schemas.append(_skill_tool.schema())

    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    result = AgentResult(output=None, stop_reason="max_steps", steps=0, messages=messages)

    def call(step: int, **kwargs: Any) -> ChatResponse:
        sent = messages
        if agent.budget:
            sent, stats = compact(messages, agent.budget, hcfg["context_keep_turns"],
                                  kwargs.get("tools"))
            result.max_context_tokens = max(result.max_context_tokens, stats.after)
            result.over_budget |= stats.after > agent.budget
        resp = client.chat(sent, **kwargs)
        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        tracer.log(
            agent.name,
            step,
            "model",
            sent,
            result=resp.content,
            args={"tool_calls": [c.name for c in resp.tool_calls], "fallback": resp.used_fallback},
            latency_s=resp.latency_s,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            context_tokens=estimate_tokens(sent, kwargs.get("tools")),
        )
        return resp

    def run_tool(step: int, name: str, args: dict[str, Any]) -> str:
        if name == LOAD_SKILL and agent.skills and skills is not None:
            skill = str(args.get("name", ""))
            content = skills.load(skill, agent.skills)
            if not content.startswith("error:"):
                result.skills_loaded.append(skill)
        elif name in agent.tools:
            tr = registry.run(name, args, max_chars, approver=approver, agent=agent.name)
            content = tr.content
            if tr.approval:
                a = tr.approval
                decided = f"{a.decision} by {a.approver} at {a.decided_at} {a.note}".strip()
                tracer.log(agent.name, step, "approval", messages, tool=name, args=a.args,
                           result=decided)
        else:
            content = f"error: tool {name!r} is not available to you"
        tracer.log(agent.name, step, "tool", messages, tool=name, args=args, result=content)
        return content

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
            content = run_tool(step, c.name, c.args)
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
