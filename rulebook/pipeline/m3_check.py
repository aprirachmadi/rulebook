"""Milestone 3 live checks: skill on demand, a 50-step run within budget, approval gate."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from rulebook.config import agent_temperature, load_config
from rulebook.harness.agents import Agent
from rulebook.harness.guard import Approver
from rulebook.harness.loop import AgentResult, run
from rulebook.harness.skills import SkillLibrary
from rulebook.harness.tools import ToolRegistry, tool
from rulebook.harness.trace import Tracer

DEMO_SKILLS = Path(__file__).parent / "demo_skills"
PAGES = 50
SECRET = "MANGGA"


def _agent(name: str, prompt: str, **kw: object) -> Agent:
    return Agent(
        name=name,
        system_prompt=prompt,
        model=load_config()["models"]["toy_agent"],
        temperature=agent_temperature(name),
        **kw,  # type: ignore[arg-type]
    )


def check_skill(tracer: Tracer) -> AgentResult:
    agent = _agent(
        "toy_skill",
        "Answer the user's question. Use your skills when they apply.",
        skills=["vault-code"],
        max_steps=5,
    )
    return run(agent, "What is the vault code for 7 guests?", registry=ToolRegistry(),
               tracer=tracer, skills=SkillLibrary(DEMO_SKILLS))


class NoArgs(BaseModel):
    pass


def check_long_run(tracer: Tracer, budget: int) -> AgentResult:
    reg = ToolRegistry()
    state = {"page": 0}
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 25  # ~1,400 chars

    @tool("read_next_page", "Read the next page of the book.", registry=reg)
    def read_next_page(_: NoArgs) -> str:
        state["page"] += 1
        p = state["page"]
        if p >= PAGES:
            return f"Page {PAGES} of {PAGES} (last page). The secret word is {SECRET}."
        return f"Page {p} of {PAGES}. {filler}\nNot finished: call read_next_page again."

    agent = _agent(
        "toy_reader",
        "You read a book one page at a time with read_next_page. Keep calling it until a page "
        "says it is the last page. Then call final_answer with the secret word from that page.",
        tools=["read_next_page"],
        max_steps=PAGES + 10,
        budget=budget,
    )
    res = run(agent, "Read the whole book and tell me the secret word.", registry=reg,
              tracer=tracer)
    res.error = res.error or f"pages read: {state['page']}"
    return res


class PatchArgs(BaseModel):
    rule: str
    text: str


def check_approval(tracer: Tracer, approver: Approver) -> tuple[AgentResult, list[str]]:
    reg = ToolRegistry()
    applied: list[str] = []

    @tool("apply_patch", "Apply a guideline rule patch.", needs_approval=True, registry=reg)
    def apply_patch(a: PatchArgs) -> str:
        applied.append(a.rule)
        return f"patch {a.rule} applied"

    agent = _agent(
        "toy_approval",
        "You apply guideline patches with apply_patch. Report what happened, then stop.",
        tools=["apply_patch"],
        max_steps=4,
    )
    res = run(agent, 'Apply rule 7.2 with text "DISC lines are menu.discountprice".',
              registry=reg, tracer=tracer, approver=approver)
    return res, applied
