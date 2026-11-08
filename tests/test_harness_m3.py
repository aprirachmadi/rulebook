"""Milestone 3: skills, context budget, approval gate, agents.yaml."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rulebook.harness.agents import Agent, AgentConfigError, build_agent, load_specs, resolve_model
from rulebook.harness.context import SUMMARY_PREFIX, compact, estimate_tokens
from rulebook.harness.guard import ApprovalRequest, CLIApprover
from rulebook.harness.loop import run
from rulebook.harness.skills import SkillLibrary, parse_skill
from rulebook.harness.tools import ToolRegistry, tool
from rulebook.harness.trace import Tracer, read_trace
from tests.test_harness import FakeServer, _reply

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


def _agent(**kw: Any) -> Agent:
    base: dict[str, Any] = {"name": "t", "system_prompt": "sys", "model": "fake", "max_steps": 5}
    return Agent(**{**base, **kw})


# --- skills ---


def test_skill_frontmatter_and_allowlist() -> None:
    lib = SkillLibrary(FIXTURES)
    assert set(lib.skills) == {"pemula", "rahasia"}
    catalog = lib.catalog(["pemula"])
    assert "pemula: How to answer" in catalog and "rahasia" not in catalog
    assert lib.load("pemula", ["pemula"]).startswith("The vault code")
    assert lib.load("rahasia", ["pemula"]).startswith("error:")
    with pytest.raises(KeyError):
        lib.catalog(["missing"])


def test_skill_name_must_match_folder(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "SKILL.md").write_text("---\nname: y\ndescription: d\n---\nbody")
    with pytest.raises(ValueError):
        parse_skill(tmp_path / "x" / "SKILL.md")


def test_loop_loads_skill_on_demand(tmp_path: Path) -> None:
    server = FakeServer(
        _reply(tool_calls=[{"name": "load_skill", "args": {"name": "pemula"}}]),
        _reply(tool_calls=[{"name": "load_skill", "args": {"name": "rahasia"}}]),
        _reply(content="done"),
    )
    res = run(_agent(skills=["pemula"]), "q", registry=ToolRegistry(), client=server.client(),
              tracer=Tracer(runs_dir=tmp_path), skills=SkillLibrary(FIXTURES))
    assert res.skills_loaded == ["pemula"]
    first = server.requests[0]
    assert "pemula: How to answer" in first["messages"][0]["content"]  # catalog in system prompt
    assert "The vault code" not in json.dumps(first)  # full text only after loading
    assert "load_skill" in {t["function"]["name"] for t in first["tools"]}
    tool_msgs = [m["content"] for m in server.requests[2]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("The vault code") and tool_msgs[1].startswith("error:")


# --- context ---


def _conversation(turns: int, chars: int) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        content = f"page {i}\n" + "x" * chars
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": content})
    return msgs


def test_compact_is_noop_under_budget() -> None:
    msgs = _conversation(2, 10)
    out, stats = compact(msgs, budget=10_000, keep_last=4)
    assert out is msgs and stats.summarized == 0


def test_compact_summarizes_old_tool_results_and_keeps_recent() -> None:
    msgs = _conversation(10, 2000)
    out, stats = compact(msgs, budget=3000, keep_last=2)
    assert stats.after <= 3000 and stats.summarized == 8 and stats.dropped_turns == 0
    assert out[:2] == msgs[:2]
    assert out[-4:] == msgs[-4:]  # last 2 turns untouched
    old_tools = [m for m in out[2:-4] if m["role"] == "tool"]
    assert all(m["content"].startswith(SUMMARY_PREFIX) for m in old_tools)
    assert old_tools[0]["content"].startswith(f"{SUMMARY_PREFIX} page 0")


def test_compact_drops_oldest_turns_when_summaries_are_not_enough() -> None:
    msgs = _conversation(200, 400)
    out, stats = compact(msgs, budget=3000, keep_last=2)
    assert stats.after <= 3000 and stats.dropped_turns > 0
    assert "earlier turns dropped" in out[2]["content"]
    # every tool result still follows the assistant call it answers
    ids = set()
    for m in out:
        ids |= {c["id"] for c in m.get("tool_calls", [])}
        if m["role"] == "tool":
            assert m["tool_call_id"] in ids


class PageArgs(BaseModel):
    page: int


def test_fifty_step_run_stays_within_budget(tmp_path: Path) -> None:
    reg = ToolRegistry()

    @tool("read_page", "Read one page.", registry=reg)
    def read_page(a: PageArgs) -> str:
        return f"page {a.page}\n" + "lorem " * 300

    calls = [_reply(tool_calls=[{"name": "read_page", "args": {"page": i}}]) for i in range(1, 51)]
    server = FakeServer(*calls, _reply(content="finished"))
    tracer = Tracer(runs_dir=tmp_path)
    res = run(_agent(tools=["read_page"], max_steps=60, budget=3000), "read", registry=reg,
              client=server.client(), tracer=tracer)
    assert res.steps == 51 and res.stop_reason == "no_tool_calls"
    assert not res.over_budget and res.max_context_tokens <= 3000
    assert estimate_tokens(res.messages) > 3000 * 5  # the full history is far bigger
    sent = [r.context_tokens for r in read_trace(tracer.path) if r.event == "model"]
    assert len(sent) == 51 and max(sent) <= 3000


# --- guard ---


class Patch(BaseModel):
    rule: str


def _approval_registry(log: list[str]) -> ToolRegistry:
    reg = ToolRegistry()

    @tool("apply_patch", "Apply a guideline patch.", needs_approval=True, registry=reg)
    def apply_patch(a: Patch) -> str:
        log.append(f"ran {a.rule}")
        return f"applied {a.rule}"

    return reg


def _scripted(answers: list[str], log: list[str]) -> CLIApprover:
    def ask(prompt: str) -> str:
        log.append(f"asked: {prompt.split('?')[0]}")
        return answers.pop(0)

    return CLIApprover(ask=ask, approver="tester")


def test_approval_blocks_until_decided_then_runs() -> None:
    log: list[str] = []
    res = _approval_registry(log).run("apply_patch", {"rule": "7.2"}, 2000,
                                      approver=_scripted(["a"], log), agent="orch")
    assert log == [r"asked: \[a]pprove / \[e]dit / \[r]eject", "ran 7.2"]  # asked before running
    assert res.ok and res.approval and res.approval.approver == "tester"
    assert res.approval.decided_at


def test_reject_does_not_run() -> None:
    log: list[str] = []
    res = _approval_registry(log).run("apply_patch", {"rule": "7.2"}, 2000,
                                      approver=_scripted(["r", "too broad"], log))
    assert not res.ok and res.content == "rejected by tester: too broad"
    assert not any(e.startswith("ran") for e in log)


def test_edit_runs_with_edited_args_after_bad_json() -> None:
    log: list[str] = []
    answers = ["e", "{bad", "e", '{"rule": "7.3"}']
    res = _approval_registry(log).run("apply_patch", {"rule": "7.2"}, 2000,
                                      approver=_scripted(answers, log))
    assert res.ok and res.content == "applied 7.3" and res.approval.decision == "edit"


def test_cli_approver_shows_context() -> None:
    from rich.console import Console

    console = Console(record=True, width=100)
    approver = CLIApprover(console=console, ask=lambda _: "a", approver="tester")
    approver.request(ApprovalRequest("apply_patch", {"rule": "7.2"}, "orch",
                                     {"impact": "12 lines", "sample": "DISC 10% [x2]"}))
    text = console.export_text()
    assert "[a]pprove" in console.render_str(r"\[a]pprove").plain
    assert "orch" in text and "apply_patch" in text and "12 lines" in text
    assert "DISC 10% [x2]" in text  # brackets in receipt text are not eaten as markup


def test_loop_traces_approvals(tmp_path: Path) -> None:
    log: list[str] = []
    server = FakeServer(
        _reply(tool_calls=[{"name": "apply_patch", "args": {"rule": "7.2"}}]),
        _reply(content="ok"),
    )
    tracer = Tracer(runs_dir=tmp_path)
    run(_agent(tools=["apply_patch"]), "go", registry=_approval_registry(log),
        client=server.client(), tracer=tracer, approver=_scripted(["a"], log))
    events = [(r.event, r.tool) for r in read_trace(tracer.path)]
    assert ("approval", "apply_patch") in events
    approval = next(r for r in read_trace(tracer.path) if r.event == "approval")
    assert approval.result.startswith("approve by tester at ")


# --- agents.yaml ---


def test_every_agent_resolves_a_model_and_fits_its_window() -> None:
    from rulebook.config import context_window

    specs = load_specs()
    assert set(specs) == {"ocr_agent", "labeler_a", "labeler_b", "labeler_c", "diagnoser",
                          "rule_writer", "verifier", "orchestrator"}
    for name, spec in specs.items():
        model = resolve_model(spec["model"])
        assert spec["budget"] < context_window(model), name
    assert len({resolve_model(specs[f"labeler_{x}"]["model"]) for x in "abc"}) == 3


def test_build_agent_needs_prompt_and_schema() -> None:
    class LabelBatch(BaseModel):
        labels: list[str]

    with pytest.raises(AgentConfigError):
        build_agent("diagnoser")  # no system prompt yet
    with pytest.raises(AgentConfigError):
        build_agent("labeler_a", system_prompt="label")  # schema missing
    a = build_agent("labeler_a", system_prompt="label", output_schema=LabelBatch)
    assert a.model == "qwen3.5:4b" and a.temperature == 0 and a.skills == ["cord-guideline"]


def test_build_agent_rejects_budget_over_window(tmp_path: Path) -> None:
    p = tmp_path / "agents.yaml"
    p.write_text("defaults: {max_steps: 3, budget: 5000}\nagents:\n  x: {model: toy_agent}\n")
    with pytest.raises(AgentConfigError, match="context window"):
        build_agent("x", system_prompt="p", path=p)
