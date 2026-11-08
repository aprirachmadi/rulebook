"""Harness unit tests against a scripted fake endpoint (no live model)."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from rulebook.harness.agents import Agent
from rulebook.harness.loop import run
from rulebook.harness.model import ModelClient, ModelError, parse_text_tool_calls
from rulebook.harness.tools import ToolRegistry, tool, truncate
from rulebook.harness.trace import Tracer, read_trace


def _reply(content: str = "", tool_calls: list[dict] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": n, "arguments": json.dumps(a)},
            }
            for i, (n, a) in enumerate((c["name"], c["args"]) for c in tool_calls)
        ]
    return {
        "choices": [{"message": msg}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class FakeServer:
    """Returns scripted responses in order; records request bodies."""

    def __init__(self, *responses: dict[str, Any] | int) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        r = self.responses.pop(0)
        if isinstance(r, int):
            return httpx.Response(r, text="boom")
        return httpx.Response(200, json=r)

    def client(self, **kw: Any) -> ModelClient:
        return ModelClient("fake", transport=httpx.MockTransport(self), backoff_s=0, **kw)


class AddArgs(BaseModel):
    a: float
    b: float


class Answer(BaseModel):
    value: float


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @tool("add", "Add two numbers.", registry=reg)
    def add(args: AddArgs) -> float:
        return args.a + args.b

    @tool("explode", "Always fails.", registry=reg)
    def explode(args: AddArgs) -> float:
        raise ZeroDivisionError("nope")

    @tool("big", "Huge output.", registry=reg)
    def big(args: AddArgs) -> str:
        return "x" * 5000

    @tool("dangerous", "Needs approval.", needs_approval=True, registry=reg)
    def dangerous(args: AddArgs) -> str:
        return "did it"

    return reg


def _agent(**kw: Any) -> Agent:
    defaults: dict[str, Any] = {
        "name": "toy",
        "system_prompt": "sys",
        "model": "fake",
        "tools": ["add", "explode", "big", "dangerous"],
        "max_steps": 5,
    }
    return Agent(**{**defaults, **kw})


# --- model.py ---


def test_native_tool_calls_parsed() -> None:
    server = FakeServer(_reply(tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}}]))
    resp = server.client().chat([], tools=[{"type": "function", "function": {"name": "add"}}])
    assert [(c.name, c.args) for c in resp.tool_calls] == [("add", {"a": 1, "b": 2})]
    assert not resp.used_fallback
    assert resp.prompt_tokens == 10


def test_fallback_parses_json_block_from_text() -> None:
    text = 'Let me add.\n```json\n{"tool": "add", "args": {"a": 1, "b": 2}}\n```'
    server = FakeServer(_reply(content=text))
    resp = server.client().chat([], tools=[{"type": "function", "function": {"name": "add"}}])
    assert [(c.name, c.args) for c in resp.tool_calls] == [("add", {"a": 1, "b": 2})]
    assert resp.used_fallback


def test_fallback_ignores_unknown_tools_and_bare_json() -> None:
    text = 'x {"tool": "rm_rf", "args": {}} y {"a": 1} z {"tool": "add", "args": {"a": 1}}'
    assert [c.name for c in parse_text_tool_calls(text, {"add"})] == ["add"]


def test_retries_then_succeeds() -> None:
    server = FakeServer(500, 503, _reply(content="ok"))
    assert server.client(max_retries=3).chat([]).content == "ok"
    assert len(server.requests) == 3


def test_gives_up_after_retries() -> None:
    server = FakeServer(500, 500)
    with pytest.raises(ModelError):
        server.client(max_retries=1).chat([])


def test_client_error_is_not_retried() -> None:
    server = FakeServer(400, _reply(content="never"))
    with pytest.raises(ModelError):
        server.client(max_retries=3).chat([])
    assert len(server.requests) == 1


def test_request_body_carries_schema_and_reasoning_effort() -> None:
    server = FakeServer(_reply(content="{}"))
    server.client(reasoning_effort="none", temperature=0).chat([], json_schema={"type": "object"})
    body = server.requests[0]
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["reasoning_effort"] == "none"


# --- tools.py ---


def test_tool_run_validates_catches_and_truncates(registry: ToolRegistry) -> None:
    assert registry.run("add", {"a": 1, "b": 2}, 2000).content == "3.0"
    bad = registry.run("add", {"a": "one"}, 2000)
    assert not bad.ok and "invalid args" in bad.content
    err = registry.run("explode", {"a": 1, "b": 2}, 2000)
    assert not err.ok and "ZeroDivisionError" in err.content
    big = registry.run("big", {"a": 1, "b": 2}, 2000)
    assert big.ok and len(big.content) < 2100 and "[truncated: 3000 more chars" in big.content
    assert not registry.run("nope", {}, 2000).ok


def test_needs_approval_tool_is_refused_without_approver(registry: ToolRegistry) -> None:
    res = registry.run("dangerous", {"a": 1, "b": 2}, 2000)
    assert not res.ok and "needs approval" in res.content


def test_tool_schema_comes_from_pydantic(registry: ToolRegistry) -> None:
    fn = registry.get("add").schema()["function"]
    assert fn["name"] == "add"
    assert set(fn["parameters"]["properties"]) == {"a", "b"}


def test_tool_decorator_rejects_non_pydantic_param() -> None:
    with pytest.raises(TypeError):

        @tool("bad", "bad", registry=ToolRegistry())
        def bad(x: int) -> int:
            return x


def test_truncate_leaves_short_text() -> None:
    assert truncate("abc", 10) == "abc"


# --- loop.py + trace.py ---


def test_loop_runs_tools_until_final_answer(registry: ToolRegistry, tmp_path: Path) -> None:
    server = FakeServer(
        _reply(tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}}]),
        _reply(tool_calls=[{"name": "final_answer", "args": {"answer": {"value": 5}}}]),
    )
    tracer = Tracer(runs_dir=tmp_path)
    res = run(
        _agent(output_schema=Answer),
        "2+3?",
        registry=registry,
        client=server.client(),
        tracer=tracer,
    )
    assert res.stop_reason == "final_answer"
    assert res.output == Answer(value=5)
    assert res.steps == 2
    # the tool result went back to the model
    tool_msgs = [m for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "5.0"

    records = read_trace(tracer.path)
    assert [r.event for r in records] == ["model", "tool", "model", "final"]
    assert {r.run_id for r in records} == {tracer.run_id}
    assert records[1].tool == "add" and records[1].args == {"a": 2, "b": 3}


def test_loop_stops_on_plain_text(registry: ToolRegistry, tmp_path: Path) -> None:
    server = FakeServer(_reply(content="It is 5."))
    res = run(_agent(), "2+3?", registry=registry, client=server.client(),
              tracer=Tracer(runs_dir=tmp_path))
    assert res.stop_reason == "no_tool_calls" and res.output == "It is 5."


def test_loop_stops_at_max_steps(registry: ToolRegistry, tmp_path: Path) -> None:
    calls = [_reply(tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}}]) for _ in range(3)]
    server = FakeServer(*calls)
    tracer = Tracer(runs_dir=tmp_path)
    res = run(_agent(max_steps=3), "loop", registry=registry, client=server.client(),
              tracer=tracer)
    assert res.stop_reason == "max_steps" and res.output is None
    assert read_trace(tracer.path)[-1].event == "error"


def test_loop_refuses_tools_outside_allowlist(registry: ToolRegistry, tmp_path: Path) -> None:
    server = FakeServer(
        _reply(tool_calls=[{"name": "big", "args": {"a": 1, "b": 1}}]),
        _reply(content="done"),
    )
    run(_agent(tools=["add"]), "x", registry=registry, client=server.client(),
        tracer=Tracer(runs_dir=tmp_path))
    tool_msg = [m for m in server.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "not available" in tool_msg["content"]


def test_output_schema_repair_retry(registry: ToolRegistry, tmp_path: Path) -> None:
    server = FakeServer(_reply(content="five"), _reply(content='{"value": 5}'))
    res = run(_agent(output_schema=Answer), "2+3?", registry=registry,
              client=server.client(), tracer=Tracer(runs_dir=tmp_path))
    assert res.output == Answer(value=5)
    assert "response_format" in server.requests[1] and "tools" not in server.requests[1]


def test_output_schema_gives_up_after_one_repair(registry: ToolRegistry, tmp_path: Path) -> None:
    server = FakeServer(_reply(content="five"), _reply(content="still five"))
    res = run(_agent(output_schema=Answer), "2+3?", registry=registry,
              client=server.client(), tracer=Tracer(runs_dir=tmp_path))
    assert res.stop_reason == "invalid_output" and res.output is None
    assert len(server.requests) == 2


def test_read_trace_rejects_invalid_lines(tmp_path: Path) -> None:
    bad = tmp_path / "trace.jsonl"
    bad.write_text('{"run_id": "x"}\n')
    with pytest.raises(ValueError):
        read_trace(bad)


def test_toy_outcome_requires_grounded_answer() -> None:
    from rulebook.harness.loop import AgentResult
    from rulebook.pipeline.toy import Answer as ToyAnswer
    from rulebook.pipeline.toy import Task, TaskOutcome, safe_eval

    task = Task("q", 41250.0)
    ungrounded = TaskOutcome(task, AgentResult(ToyAnswer(value=41250), "final_answer", 2), [75.0])
    assert ungrounded.correct and not ungrounded.grounded and not ungrounded.solved
    grounded = TaskOutcome(task, AgentResult(ToyAnswer(value=41250), "final_answer", 2), [41250.0])
    assert grounded.solved
    off_by_one = TaskOutcome(task, AgentResult(ToyAnswer(value=41251), "final_answer", 2), [41251])
    assert not off_by_one.correct
    assert safe_eval("(2 + 3) * 4 / 5") == 4.0
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")


def test_temperature_omitted_when_unset() -> None:
    server = FakeServer(_reply(content="x"))
    server.client().chat([])
    assert "temperature" not in server.requests[0]


def test_agent_temperatures_from_config() -> None:
    from rulebook.config import agent_temperature

    assert [agent_temperature(f"labeler_{x}") for x in "abc"] == [0, 0, 0]
    assert agent_temperature("verifier") == 0
    assert {agent_temperature(n) for n in ("rule_writer", "diagnoser", "orchestrator")} == {0.2}
    assert agent_temperature("unknown_agent") is None


def test_embed_uses_embeddings_endpoint() -> None:
    server = FakeServer({"data": [{"index": 1, "embedding": [0.0, 1.0]},
                                  {"index": 0, "embedding": [1.0, 0.0]}]})
    vecs = server.client().embed(["a", "b"])
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    assert server.requests[0] == {"model": "fake", "input": ["a", "b"]}
