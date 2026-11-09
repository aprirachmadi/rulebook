"""The orchestrator agent: scripted model, fake improve session. Checks the tools enforce the
order (diagnose -> write -> verify -> submit), the cluster cap, and that decisions are traced."""

import json
from types import SimpleNamespace
from typing import Any

import httpx

from rulebook.harness.agents import build_agent
from rulebook.harness.loop import run as run_agent
from rulebook.harness.model import ModelClient
from rulebook.harness.trace import Tracer, read_trace
from rulebook.pipeline.orchestrator_agent import OrchestratorReport, orchestrator_registry


def _call(name: str, args: dict, i: int) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class FakeSession:
    def __init__(self) -> None:
        self.clusters = [SimpleNamespace(cluster_id=i, items=[1] * (9 - i), labels=("a", "b"),
                                         pattern="total") for i in range(3)]
        self.diagnoses: dict = {}
        self.candidates: dict = {}
        self.run_id = "r"
        self.queue: list = []
        self.db = SimpleNamespace(execute=lambda sql, args: self.queue.append(args),
                                  commit=lambda: None)

    def get(self, cid: int):
        return next(c for c in self.clusters if c.cluster_id == cid)

    def diagnose(self, c, note=""):
        d = SimpleNamespace(diagnosis="GUIDELINE_GAP", reason="no rule")
        self.diagnoses[c.cluster_id] = d
        return d

    def write(self, c, note=""):
        cand = SimpleNamespace(number="3.8", draft=SimpleNamespace(
            section="3", rule_text="CASH -> total.cashprice", rationale="r"),
            verification={}, submitted=False)
        self.candidates[c.cluster_id] = cand
        return cand

    def verify(self, cands):
        for c in cands:
            c.verification = {"verdict": "pass", "fail_reasons": [], "gold_accuracy_before": .5,
                              "gold_accuracy_after": .6, "cluster_alpha_before": 0,
                              "cluster_alpha_after": 1}

    def submit(self, cand):
        cand.submitted = True


def test_orchestrator_drives_one_cluster_and_respects_the_cap(tmp_path) -> None:
    script = [("list_clusters", {}),
              ("submit_patch", {"cluster_id": 0}),  # too early: no verified rule
              ("delegate", {"agent_name": "diagnoser", "cluster_id": 0, "reason": "biggest"}),
              ("delegate", {"agent_name": "rule_writer", "cluster_id": 0, "reason": "gap"}),
              ("delegate", {"agent_name": "verifier", "cluster_id": 0, "reason": "check"}),
              ("submit_patch", {"cluster_id": 0, "reason": "verified"}),
              ("delegate", {"agent_name": "diagnoser", "cluster_id": 1, "reason": "next"}),
              ("skip_cluster", {"cluster_id": 1, "reason": "over budget"}),
              ("final_answer", {"answer": {"summary": "1 rule submitted", "submitted": [0]}})]
    responses = [_call(n, a, i) for i, (n, a) in enumerate(script)]
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    s = FakeSession()
    agent = build_agent("orchestrator", output_schema=OrchestratorReport)
    tracer = Tracer(run_id="orch", runs_dir=tmp_path)
    res = run_agent(agent, "go", registry=orchestrator_registry(s, max_clusters=1),
                    tracer=tracer, client=ModelClient("fake", backoff_s=0,
                                                      transport=httpx.MockTransport(handler)))
    assert res.output == OrchestratorReport(summary="1 rule submitted", submitted=[0])
    assert s.candidates[0].submitted and 1 not in s.diagnoses  # cap held
    assert s.decisions == [{"cluster_id": 0, "action": "submit", "reason": "verified"},
                           {"cluster_id": 1, "action": "skip", "reason": "over budget"}]
    tools = [(r.tool, r.result) for r in read_trace(tracer.path) if r.event == "tool"]
    assert "no verified rule" in tools[1][1]
    assert '"verdict": "pass"' in tools[4][1]
    assert "at most 1 clusters" in tools[6][1]
    assert {t["function"]["name"] for t in seen[0]["tools"]} >= {
        "list_clusters", "delegate", "submit_patch", "skip_cluster"}
