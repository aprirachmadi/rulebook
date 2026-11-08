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


@app.command()
def m3_check(
    part: str = typer.Argument("all", help="skill | budget | approval | all"),
    budget: int = typer.Option(3000, help="Token budget for the long run"),
) -> None:
    """Milestone 3 live checks on the toy model."""
    from rulebook.harness.guard import CLIApprover
    from rulebook.harness.trace import Tracer, read_trace
    from rulebook.pipeline import m3_check as m3

    tracer = Tracer()
    if part in ("skill", "all"):
        r = m3.check_skill(tracer)
        console.print(f"[bold]skill[/]: loaded={r.skills_loaded} answer={r.output!r} "
                      f"steps={r.steps} stop={r.stop_reason}")
    if part in ("budget", "all"):
        r = m3.check_long_run(tracer, budget)
        console.print(f"[bold]budget[/]: steps={r.steps} stop={r.stop_reason} answer={r.output!r} "
                      f"{r.error} | max context sent={r.max_context_tokens} budget={budget} "
                      f"over_budget={r.over_budget} | full history≈"
                      f"{sum(len(str(m.get('content') or '')) for m in r.messages) // 4} tokens")
    if part in ("approval", "all"):
        r, applied = m3.check_approval(tracer, CLIApprover(console=console))
        console.print(f"[bold]approval[/]: applied={applied} answer={str(r.output)[:80]!r}")
    console.print(f"trace valid: {len(read_trace(tracer.path))} records -> {tracer.path}")
