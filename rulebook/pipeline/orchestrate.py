"""M6: the end-to-end loop, batch by batch.

label -> cluster -> diagnose -> write -> verify (advisory) -> approval -> apply -> relabel ->
exports -> next batch. `apply_patch` and `relabel` are `needs_approval` tools (SPEC section 3),
so every change waits for a human: `QueueApprover` (the UI) or `CLIApprover` (terminal).

The loop itself is plain code that calls the agents' stages; the SPEC's LLM orchestrator
agent with `delegate` is logged under "improve later" (decision 2026-11-09: end to end first).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rich.console import Console

from rulebook.config import ROOT, load_config
from rulebook.data.store import connect
from rulebook.harness.guard import Approver, CLIApprover, QueueApprover
from rulebook.harness.tools import ToolRegistry, tool
from rulebook.pipeline import exports
from rulebook.pipeline.improve import improve
from rulebook.pipeline.labeling import LABELERS, run_batch
from rulebook.tools.guideline import GUIDELINES, current_version, publish_version, render_patch

console = Console()
AGENT = "orchestrator"


class PatchArgs(BaseModel):
    patch_id: str
    rule_text: str


class RelabelArgs(BaseModel):
    run_id: str
    guideline_version: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def new_loop_id() -> str:
    return "loop-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def db_path() -> Path:
    return ROOT / load_config()["storage"]["db_path"]


def _patch_words(items_json: str) -> list[tuple[str, int, int]]:
    """Cluster item ids 'rid:line:w0-wn' -> word keys."""
    out = []
    for item in json.loads(items_json):
        rid, li, span = item.rsplit(":", 2)
        w0, wn = (int(x) for x in span.split("-"))
        out += [(rid, int(li), w) for w in range(w0, wn + 1)]
    return out


def run_loop(text_source: str, batch_size: int, batches: int, offset: int = 0,
             max_clusters: int = 0, approver_kind: Literal["queue", "cli"] = "queue",
             loop_id: str | None = None) -> str:
    cfg = load_config()
    if max_clusters:
        cfg["clustering"]["max_clusters_per_run"] = max_clusters
    loop_id = loop_id or new_loop_id()
    db = connect()
    settings = {"text_source": text_source, "batch_size": batch_size, "batches": batches,
                "offset": offset, "max_clusters": cfg["clustering"]["max_clusters_per_run"],
                "approver": approver_kind}
    db.execute("INSERT OR REPLACE INTO loops VALUES (?, ?, ?, 'running', 'starting', ?)",
               (loop_id, _now(), json.dumps(settings), _now()))
    db.commit()

    def step(text: str) -> None:
        db.execute("UPDATE loops SET step = ?, updated_at = ? WHERE loop_id = ?",
                   (text, _now(), loop_id))
        db.commit()
        console.print(f"[bold]{loop_id}[/]: {text}")

    approver: Approver = (QueueApprover(db_path(), scope=loop_id,
                                        on_wait=lambda i: step(f"waiting for approval #{i}"))
                          if approver_kind == "queue" else CLIApprover())
    out = exports.EXPORTS
    state: dict = {}
    reg = ToolRegistry()

    @tool("apply_patch", "Add an approved rule to the guideline as a new version.",
          needs_approval=True, registry=reg)
    def apply_patch(a: PatchArgs) -> str:
        num, section, examples = db.execute(
            "SELECT rule_number, section, examples_json FROM patches WHERE patch_id = ?",
            (a.patch_id,)).fetchone()
        block = render_patch(num, a.rule_text, json.loads(examples))
        version = publish_version(section, block, f"rule {num} added ({a.patch_id})")
        pdir = GUIDELINES / "patches"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{a.patch_id}.json").write_text(json.dumps(
            {"patch_id": a.patch_id, "rule_number": num, "section": section,
             "rule_text": a.rule_text, "examples": json.loads(examples),
             "guideline_version": version}, indent=1, ensure_ascii=False))
        return version

    @tool("relabel", "Relabel a batch's receipts under a new guideline version.",
          needs_approval=True, registry=reg)
    def relabel(a: RelabelArgs) -> str:
        if a.guideline_version != current_version():
            raise ValueError(f"current guideline is {current_version()}")
        return run_batch(text_source, batch_size, state["offset"])["run_id"]  # type: ignore

    max_chars = cfg["harness"]["tool_result_max_chars"]
    try:
        for b in range(batches):
            state["offset"] = offset + b * batch_size
            timings: dict[str, float] = {}
            step(f"batch {b}: labeling {batch_size} receipts from offset {state['offset']}")
            t0 = datetime.now(UTC)
            summary = run_batch(text_source, batch_size, state["offset"])  # type: ignore
            run_id, v_before = summary["run_id"], summary["guideline_version"]
            timings["label"] = (datetime.now(UTC) - t0).total_seconds()

            step(f"batch {b}: cluster, diagnose, write, verify ({run_id})")
            res = improve(run_id)
            timings |= res["timings_s"]

            decided, rules = [], {}
            for p in res["patches"]:
                pid = p["patch_id"]
                rule_text, items_json = db.execute(
                    "SELECT p.rule_text, c.items_json FROM patches p JOIN clusters c ON "
                    "c.run_id = p.run_id AND c.cluster_id = p.cluster_id WHERE p.patch_id = ?",
                    (pid,)).fetchone()
                step(f"batch {b}: patch {p['rule_number']} waiting for a decision")
                r = reg.run("apply_patch", {"patch_id": pid, "rule_text": rule_text}, max_chars,
                            approver, AGENT, context={"verification": p, "run_id": run_id})
                ap = r.approval
                decision = ("rejected" if not r.ok else
                            "edited and approved" if ap and ap.decision == "edit" else "approved")
                final_text = ap.args.get("rule_text", rule_text) if ap else rule_text
                db.execute("UPDATE patches SET status = ?, approver = ?, decided_at = ?, "
                           "rule_text = ? WHERE patch_id = ?",
                           ("approved" if r.ok else "rejected", ap.approver if ap else None,
                            ap.decided_at if ap else None, final_text, pid))
                db.commit()
                if r.ok:
                    rules |= {k: p["rule_number"] for k in _patch_words(items_json)}
                elif ap is None or ap.decision != "reject":
                    console.print(f"[red]apply_patch failed: {r.content}[/]")
                decided.append({"patch_id": pid, "rule_number": p["rule_number"],
                                "rule_text": final_text, "decision": decision,
                                "approver": ap.approver if ap else None, "verification": p,
                                "result": r.content})

            relabel_id, v_after, alpha_after = None, current_version(), None
            final_run = run_id
            if v_after != v_before:
                step(f"batch {b}: relabel under {v_after} waiting for a decision")
                t0 = datetime.now(UTC)
                r = reg.run("relabel", {"run_id": run_id, "guideline_version": v_after},
                            max_chars, approver, AGENT,
                            context={"receipts": batch_size, "from": v_before, "to": v_after})
                if r.ok:
                    relabel_id = final_run = r.content
                    alpha_after = json.loads((ROOT / cfg["harness"]["runs_dir"] / relabel_id /
                                              "summary.json").read_text())["alpha_word_level"]
                timings["relabel"] = (datetime.now(UTC) - t0).total_seconds()

            step(f"batch {b}: exports")
            labels = exports.final_labels(db, final_run, summary.get("labelers_run", LABELERS),
                                          run_id if relabel_id else None, rules)
            exports.export_receipts(db, final_run, current_version(), labels, out / "receipts")
            exports.export_guideline(current_version(), out / "guidelines")
            exports.export_traces([run_id, f"{run_id}-improve"] + ([relabel_id] if relabel_id
                                                                   else []), out / "traces")
            db.execute("INSERT OR REPLACE INTO loop_batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (loop_id, b, run_id, relabel_id, v_before, current_version(),
                        summary["alpha_word_level"], alpha_after, json.dumps(decided),
                        json.dumps({k: round(v, 1) for k, v in timings.items()})))
            db.commit()
            exports.loop_report(db, loop_id, out / "reports")
        db.execute("UPDATE loops SET status = 'done', step = 'done', updated_at = ? "
                   "WHERE loop_id = ?", (_now(), loop_id))
        db.commit()
        exports.loop_report(db, loop_id, out / "reports")
    except Exception as e:
        db.execute("UPDATE loops SET status = 'failed', step = ?, updated_at = ? "
                   "WHERE loop_id = ?", (f"failed: {type(e).__name__}: {e}"[:500], _now(),
                                         loop_id))
        db.commit()
        raise
    return loop_id
