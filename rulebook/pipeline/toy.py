"""Milestone 2 toy agent: a calculator tool and five arithmetic tasks."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rulebook.config import agent_temperature, load_config
from rulebook.harness.agents import Agent
from rulebook.harness.loop import AgentResult, run
from rulebook.harness.tools import ToolRegistry, tool
from rulebook.harness.trace import Tracer, read_trace

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic. No names, calls or attributes."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")

    return ev(ast.parse(expression, mode="eval").body)


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic like '(2 + 3) * 4 / 5'. Plain numbers only.")


class Answer(BaseModel):
    value: float


registry = ToolRegistry()


@tool("calculator", "Evaluate an arithmetic expression and return the number.", registry=registry)
def calculator(args: CalculatorArgs) -> float:
    return safe_eval(args.expression)


SYSTEM_PROMPT = (
    "You solve arithmetic word problems. Use the calculator tool for every calculation; "
    "do not compute in your head. Numbers like 12.500 or 100.000 use '.' as a thousands "
    "separator (Indonesian format), so 12.500 means 12500. When you have the result, call "
    'final_answer with {"answer": {"value": <number>}}.'
)


@dataclass
class Task:
    question: str
    expected: float  # computed by Python below, not typed in by hand


TASKS = [
    Task("What is 1234 multiplied by 5678?", 1234 * 5678),
    Task("What is (17.5 + 2.25) * 4 - 3?", (17.5 + 2.25) * 4 - 3),
    Task("What is 2 to the power of 20, divided by 64?", 2**20 / 64),
    Task(
        "A receipt has 3 items at Rp 12.500 each, plus 10% tax on the total. "
        "How much is the bill?",
        3 * 12500 * 1.10,
    ),
    Task("The bill is Rp 87.450 and the customer pays Rp 100.000 in cash. How much change?",
         100000 - 87450),
]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


@dataclass
class TaskOutcome:
    task: Task
    result: AgentResult
    calculator_results: list[float]

    @property
    def used_calculator(self) -> bool:
        return bool(self.calculator_results)

    @property
    def correct(self) -> bool:
        out = self.result.output
        return isinstance(out, Answer) and _close(out.value, self.task.expected)

    @property
    def grounded(self) -> bool:
        """The final answer equals a number the calculator actually returned."""
        out = self.result.output
        if not isinstance(out, Answer):
            return False
        return any(_close(out.value, r) for r in self.calculator_results)

    @property
    def solved(self) -> bool:
        return self.correct and self.grounded


def _calculator_results(messages: list[dict]) -> list[float]:
    ids = {
        c["id"]
        for m in messages
        if m["role"] == "assistant"
        for c in m.get("tool_calls", [])
        if c["function"]["name"] == "calculator"
    }
    results = []
    for m in messages:
        if m["role"] == "tool" and m["tool_call_id"] in ids:
            try:
                results.append(float(m["content"]))
            except ValueError:
                pass  # an error message, not a number
    return results


def run_toy(model: str | None = None) -> tuple[Tracer, list[TaskOutcome]]:
    agent = Agent(
        name="toy_calculator",
        system_prompt=SYSTEM_PROMPT,
        model=model or load_config()["models"]["toy_agent"],
        tools=["calculator"],
        max_steps=8,
        output_schema=Answer,
        temperature=agent_temperature("toy_calculator"),
    )
    tracer = Tracer()
    outcomes = []
    for t in TASKS:
        res = run(agent, t.question, registry=registry, tracer=tracer)
        outcomes.append(TaskOutcome(t, res, _calculator_results(res.messages)))
    read_trace(tracer.path)  # raises if any record is invalid
    return tracer, outcomes
