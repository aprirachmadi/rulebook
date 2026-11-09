"""The orchestrator as an LLM agent (decision 2026-11-09; the plain-code `improve()` stays as
a fallback). It picks which clusters to handle, delegates to the diagnoser, rule writer and
verifier, and submits patches to the review queue. Its tool calls (with a reason each) and its
final report are in the run's improve trace, agent "orchestrator".

Approval, apply and relabel stay in the loop after this round: they need a human.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from rulebook.harness.agents import build_agent
from rulebook.harness.loop import run as run_agent
from rulebook.harness.orchestrator import DelegateArgs, add_delegate
from rulebook.harness.tools import ToolRegistry, tool
from rulebook.pipeline.improve import ImproveSession

HELPERS = ("diagnoser", "rule_writer", "verifier")
SHOWN_FACTOR = 2  # the agent sees 2x as many clusters as it may handle, and picks


class OrchestratorReport(BaseModel):
    summary: str
    submitted: list[int] = Field(default_factory=list)


class ClusterArg(BaseModel):
    cluster_id: int
    reason: str = Field(default="", description="why (recorded for the human)")


class NoArgs(BaseModel):
    pass


def orchestrator_registry(s: ImproveSession, max_clusters: int) -> ToolRegistry:
    reg = ToolRegistry()
    handled: set[int] = set()
    decisions: list[dict] = []
    s.decisions = decisions  # type: ignore[attr-defined]

    def state(cid: int) -> str:
        d = s.diagnoses.get(cid)
        cand = s.candidates.get(cid)
        if cand and cand.submitted:
            return "patch submitted"
        if cand and cand.verification:
            return f"rule {cand.number} verified ({cand.verification['verdict']})"
        if cand:
            return f"rule {cand.number} drafted, not verified"
        if cid in s.diagnoses:
            return f"diagnosed {d.diagnosis if d else 'FAILED'}"
        if any(x["cluster_id"] == cid and x["action"] == "skip" for x in decisions):
            return "skipped"
        return "open"

    @tool("list_clusters", "The open clusters with their size, labels and status.", registry=reg)
    def list_clusters(_: NoArgs) -> str:
        return json.dumps({
            "may_handle": max_clusters, "handled": sorted(handled),
            "clusters": [{"cluster_id": c.cluster_id, "items": len(c.items),
                          "labels_voted": list(c.labels), "text_pattern": c.pattern,
                          "status": state(c.cluster_id)} for c in s.clusters]},
            ensure_ascii=False)

    def diagnose(a: DelegateArgs) -> str:
        c = s.get(a.cluster_id)
        if a.cluster_id not in handled and len(handled) >= max_clusters:
            return (f"error: you may handle at most {max_clusters} clusters "
                    f"(handled {sorted(handled)}); skip the rest")
        handled.add(a.cluster_id)
        d = s.diagnose(c, a.note)
        return json.dumps({"cluster_id": c.cluster_id,
                           "diagnosis": d.diagnosis if d else "FAILED",
                           "reason": d.reason if d else "the diagnoser gave no answer"})

    def write(a: DelegateArgs) -> str:
        cand = s.write(s.get(a.cluster_id), a.note)
        if cand is None:
            return "error: the rule writer gave no valid rule"
        return json.dumps({"cluster_id": a.cluster_id, "rule_number": cand.number,
                           "section": cand.draft.section, "rule_text": cand.draft.rule_text,
                           "rationale": cand.draft.rationale}, ensure_ascii=False)

    def verify(a: DelegateArgs) -> str:
        cand = s.candidates.get(a.cluster_id)
        if cand is None:
            return f"error: cluster {a.cluster_id} has no drafted rule; run rule_writer first"
        s.verify([cand])
        v = cand.verification
        return json.dumps({"cluster_id": a.cluster_id, "rule_number": cand.number,
                           "verdict": v["verdict"], "fail_reasons": v["fail_reasons"],
                           "gold_accuracy": [v["gold_accuracy_before"],
                                             v["gold_accuracy_after"]],
                           "cluster_alpha": [v["cluster_alpha_before"],
                                             v["cluster_alpha_after"]]})

    add_delegate(reg, {"diagnoser": diagnose, "rule_writer": write, "verifier": verify})

    @tool("submit_patch", "Send a cluster's verified rule to the human review queue.",
          registry=reg)
    def submit_patch(a: ClusterArg) -> str:
        cand = s.candidates.get(a.cluster_id)
        if cand is None or not cand.verification:
            return f"error: cluster {a.cluster_id} has no verified rule (delegate verifier first)"
        if cand.submitted:
            return f"rule {cand.number} is already in the review queue"
        s.submit(cand)
        decisions.append({"cluster_id": a.cluster_id, "action": "submit", "reason": a.reason})
        return f"rule {cand.number} sent to the review queue"

    @tool("skip_cluster", "Leave a cluster for human review without writing a rule.",
          registry=reg)
    def skip_cluster(a: ClusterArg) -> str:
        s.get(a.cluster_id)
        decisions.append({"cluster_id": a.cluster_id, "action": "skip", "reason": a.reason})
        s.db.execute("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                     (s.run_id, "cluster", json.dumps({"cluster_id": a.cluster_id}),
                      f"skipped by orchestrator: {a.reason}"))
        s.db.commit()
        return f"cluster {a.cluster_id} skipped"

    return reg


def improve_with_agent(run_id: str, client=None) -> dict:
    s = ImproveSession(run_id)
    max_clusters = s.ccfg["max_clusters_per_run"]
    s.cluster(cap=max_clusters * SHOWN_FACTOR)
    agent = build_agent("orchestrator", output_schema=OrchestratorReport)
    agent.system_prompt = agent.system_prompt.replace("{MAX_CLUSTERS}", str(max_clusters))
    res = run_agent(agent,
                    f"Run the improvement round for batch run {run_id}: {len(s.clusters)} open "
                    f"clusters, guideline {s.version}. Start with list_clusters.",
                    registry=orchestrator_registry(s, max_clusters), tracer=s.tracer,
                    client=client)
    report = res.output
    acted = set(s.diagnoses) | {d["cluster_id"] for d in getattr(s, "decisions", [])}
    for c in s.clusters:  # clusters the agent never touched still reach a human
        if c.cluster_id not in acted:
            s.db.execute("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                         (s.run_id, "cluster", json.dumps({"cluster_id": c.cluster_id}),
                          "not handled by the orchestrator"))
    s.db.commit()
    return s.finish({
        "orchestrator": "agent",
        "orchestrator_stop": res.stop_reason, "orchestrator_steps": res.steps,
        "orchestrator_report": report.model_dump() if report else None,
        "orchestrator_decisions": getattr(s, "decisions", []),
    })
