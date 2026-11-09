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
def run(
    batch_size: int = typer.Option(20, help="Receipts in the batch (first N by id)"),
    text_source: str = typer.Option("ocr", help="ocr | gold"),
    offset: int = typer.Option(0, help="Skip this many receipts first"),
    labelers: str = typer.Option("", help="Comma-separated subset, e.g. labeler_a (default: all)"),
    improve: bool = typer.Option(False, help="Then cluster, diagnose, write and verify (M5)"),
    guideline_version: str = typer.Option("", help="Label with this version, e.g. v1 "
                                                   "(default: current)"),
) -> None:
    """Label a batch with the 3 labelers and report agreement (M4); optionally improve (M5)."""
    from rulebook.pipeline.labeling import run_batch

    if text_source not in ("ocr", "gold"):
        raise typer.BadParameter("text-source must be ocr or gold")
    subset = [x.strip() for x in labelers.split(",") if x.strip()] or None
    summary = run_batch(text_source, batch_size, offset, subset,  # type: ignore[arg-type]
                        guideline_version or None)
    if improve:
        from rulebook.pipeline.improve import improve as run_improve

        run_improve(summary["run_id"])


@app.command()
def improve(
    run_id: str,
    max_clusters: int = typer.Option(0, help="Override clustering.max_clusters_per_run"),
) -> None:
    """Cluster, diagnose, write and verify on an already-labeled run (M5)."""
    from rulebook.config import load_config
    from rulebook.pipeline.improve import improve as run_improve

    if max_clusters:
        load_config()["clustering"]["max_clusters_per_run"] = max_clusters
    run_improve(run_id)


@app.command()
def loop(
    batch_size: int = typer.Option(10, help="Receipts per batch (first N by id)"),
    batches: int = typer.Option(1, help="Number of batches"),
    text_source: str = typer.Option("ocr", help="ocr | gold"),
    offset: int = typer.Option(0, help="Start at this receipt"),
    max_clusters: int = typer.Option(0, help="Override clustering.max_clusters_per_run"),
    approver: str = typer.Option("queue", help="queue (UI) | cli"),
    loop_id: str = typer.Option("", help="Loop id (default: new)"),
    orchestrator: str = typer.Option("agent", help="agent (LLM orchestrator) | code (fallback)"),
) -> None:
    """M6: label -> improve -> approval -> apply -> relabel -> exports, batch by batch."""
    from rulebook.pipeline.orchestrate import run_loop

    if text_source not in ("ocr", "gold") or approver not in ("queue", "cli") \
            or orchestrator not in ("agent", "code"):
        raise typer.BadParameter("use ocr|gold, queue|cli, agent|code")
    run_loop(text_source, batch_size, batches, offset, max_clusters, approver,  # type: ignore
             loop_id or None, orchestrator)  # type: ignore


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


if __name__ == "__main__":
    app()
