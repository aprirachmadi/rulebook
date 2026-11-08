"""CLI entry point: `rulebook ...`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def toy_agent(model: str = typer.Option(None, help="Override models.toy_agent")) -> None:
    """Milestone 2 check: calculator agent on 5 tasks, then validate the trace."""
    from rulebook.harness.trace import read_trace
    from rulebook.pipeline.toy import run_toy

    tracer, outcomes = run_toy(model)
    table = Table(
        "#", "question", "expected", "answer", "correct", "grounded", "calc results", "steps",
        "stop", "tokens",
    )
    for i, o in enumerate(outcomes, 1):
        out = o.result.output
        table.add_row(
            str(i),
            o.task.question[:48],
            f"{o.task.expected:.10g}",
            f"{out.value:.10g}" if out is not None else "-",
            "yes" if o.correct else "NO",
            "yes" if o.grounded else "NO",
            ", ".join(f"{r:g}" for r in o.calculator_results) or "-",
            str(o.result.steps),
            o.result.stop_reason,
            str(o.result.prompt_tokens + o.result.completion_tokens),
        )
    console.print(table)
    records = read_trace(tracer.path)
    solved = sum(o.solved for o in outcomes)
    console.print(
        f"solved (correct and grounded) {solved}/{len(outcomes)} | "
        f"trace valid: {len(records)} records -> {tracer.path}"
    )
    if solved < len(outcomes):
        raise typer.Exit(1)


@app.command()
def run() -> None:
    """Labeling pipeline (Milestone 4+)."""
    console.print("not implemented yet")
    raise typer.Exit(1)
