"""M5 on a labeled run: cluster -> diagnose -> write -> verify, with timings per stage.

Clustering, diagnosis and rule writing read only labeler votes (never v2). Verification reads
gold labels only through `rulebook.tools.verify` (the sanctioned gold-set route). The verifier is
advisory (decision 2026-11-09): every patch is "proposed" with its numbers and goes to the
review queue, where a human approves, edits or rejects it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

from rulebook.config import ROOT, load_config
from rulebook.data.store import connect, load_docs, load_votes_full
from rulebook.harness.agents import build_agent
from rulebook.harness.loop import run as run_agent
from rulebook.harness.memory import default_guard
from rulebook.harness.model import ModelClient
from rulebook.harness.skills import SkillLibrary
from rulebook.harness.trace import Tracer
from rulebook.pipeline.labeling import (
    LABELERS,
    LabelerStats,
    label_doc,
    labeler_agent,
    labeler_labels,
)
from rulebook.tools.agent_tools import RunContext, build_registry, cluster_view
from rulebook.tools.clusters import (
    Cluster,
    cluster_items,
    disagreement_spans,
    make_clusters,
    split_low_confidence,
)
from rulebook.tools.guideline import (
    current_version,
    guideline_text,
    next_rule_number,
    render_patch,
    sections,
    with_patches,
    write_skill_text,
)
from rulebook.tools.labels import LabelDoc
from rulebook.tools.verify import (
    WordLabels,
    as_word_labels,
    gold_accuracy,
    gold_docs,
    gold_label_set,
    thresholds,
    words_alpha,
)

console = Console()
# No-op control patch: changes no labeling decision. Its gold-accuracy change is the noise
# floor of "any edit to the guideline text" (decision 2026-11-09). Reported, not used to judge.
CONTROL_SECTION = "4"
CONTROL_RULE = "Read each line in full before you choose its labels."
Diag = Literal["OCR_ERROR", "MODEL_ERROR", "AMBIGUOUS", "GUIDELINE_GAP"]


class Diagnosis(BaseModel):
    diagnosis: Diag
    reason: str


class PatchDraft(BaseModel):
    section: str
    rule_text: str
    example_item_ids: list[str] = Field(min_length=3, max_length=5)
    rationale: str


def draft_model(visible_ids: set[str], allowed_sections: list[str]) -> type[PatchDraft]:
    """Examples must come from the cluster part the rule writer was shown."""

    class Draft(PatchDraft):
        @field_validator("example_item_ids")
        @classmethod
        def _visible(cls, v: list[str]) -> list[str]:
            bad = [i for i in v if i not in visible_ids]
            if bad:
                raise ValueError(f"example ids not in the cluster you were shown: {bad}")
            return v

        @field_validator("section")
        @classmethod
        def _section(cls, v: str) -> str:
            v = v.strip().rstrip(".")
            if v not in allowed_sections:
                raise ValueError(f"section must be one of {allowed_sections}")
            return v

    return Draft


@dataclass
class Candidate:
    patch_id: str
    cluster: Cluster
    draft: PatchDraft
    number: str
    block: str
    skills: SkillLibrary  # cord-guideline = current guideline + this patch
    verification: dict = field(default_factory=dict)
    submitted: bool = False


def _held_receipts(docs: dict[str, LabelDoc],
                   words: list[tuple[str, int, int]]) -> list[LabelDoc]:
    """Whole receipts that hold these words. They are relabeled in full (as in the run) and
    only the held words are scored (decision 2026-11-09)."""
    return [docs[rid] for rid in sorted({rid for rid, _, _ in words})]


def labeler_instructions(name: str) -> str:
    """The system prompt a labeler follows (label list filled in), shown to the rule writer."""
    return labeler_agent(name, labeler_labels()).system_prompt


def _item_words(items: list) -> list[tuple[str, int, int]]:
    return [(it.receipt_id, it.line_idx, w) for it in items for w in it.words]


def _label_docs(agent, docs: list[LabelDoc], labels, tracer, skills, st) -> WordLabels:
    out: WordLabels = {}
    for d in docs:
        got, _ = label_doc(agent, d, labels, load_config()["labeling"]["chunk_lines"], tracer,
                           skills, st)
        out |= as_word_labels(d, got)
    return out


def _cache_key(text_source: str, agents: list[str]) -> str:
    from rulebook.harness.agents import load_specs

    specs = load_specs()
    blob = json.dumps({
        "guideline": guideline_text(current_version()), "text_source": text_source,
        "models": [load_config()["models"]["labelers"][a] for a in agents],
        "prompt": specs["labeler_a"]["system_prompt"],
        "chunk": load_config()["labeling"]["chunk_lines"],
        "gold": load_config()["data"]["gold_receipt_ids"],
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ImproveSession:
    """One run's improve stages. `improve()` runs them in a fixed order (plain-code path);
    the orchestrator agent calls them one cluster at a time through its tools."""

    def __init__(self, run_id: str, tracer: Tracer | None = None) -> None:
        cfg = load_config()
        self.cfg, self.ccfg, self.run_id = cfg, cfg["clustering"], run_id
        self.db = connect()
        self.text_source, self.version = self.db.execute(
            "SELECT text_source, guideline_version FROM runs WHERE run_id = ?",
            (run_id,)).fetchone()
        self.docs = {d.receipt_id: d for d in load_docs(self.db, run_id)}
        self.votes = load_votes_full(self.db, run_id)
        summary = json.loads((ROOT / cfg["harness"]["runs_dir"] / run_id /
                              "summary.json").read_text())
        self.agents = summary.get("labelers_run", LABELERS)
        self.tracer = tracer or Tracer(run_id=f"{run_id}-improve")
        self.timings: dict[str, float] = {"cluster": 0.0, "diagnose": 0.0, "write": 0.0,
                                          "verify": 0.0}
        self.out_dir = ROOT / cfg["harness"]["runs_dir"] / run_id / "improve"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.skills = SkillLibrary()
        self.base_text = guideline_text(self.version)
        self.taken: set[str] = set()
        self.clusters: list[Cluster] = []
        self.diagnoses: dict[int, Diagnosis | None] = {}
        self.candidates: dict[int, Candidate] = {}
        self.forced = 0  # diagnoses that hit max_steps and were forced
        self.control: dict = {}
        self.stats: dict = {}

    def _time(self, stage: str, t0: float) -> None:
        self.timings[stage] += time.perf_counter() - t0

    def cluster(self, cap: int | None = None) -> list[Cluster]:
        """Spans -> clusters; the top `cap` (default: config) are open for the agents, the rest
        go to the review queue."""
        t0 = time.perf_counter()
        ccfg = {**self.ccfg, "max_clusters_per_run": cap or self.ccfg["max_clusters_per_run"]}
        embedder = ModelClient(self.cfg["models"]["embedding"])
        embed = lambda texts: np.asarray(embedder.embed(texts), dtype=float)  # noqa: E731
        accepted, items = disagreement_spans(list(self.docs.values()), self.votes, self.agents,
                                             self.cfg["pipeline"]["accept_min_confidence"])
        low, items = split_low_confidence(items, self.agents)
        groups, cstats = cluster_items(items, embed, ccfg)
        self.clusters, rest = make_clusters(groups, ccfg)
        db, run_id = self.db, self.run_id
        db.execute("DELETE FROM clusters WHERE run_id = ?", (run_id,))
        db.execute("DELETE FROM review_queue WHERE run_id = ?", (run_id,))
        for c in self.clusters:
            db.execute("INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (run_id, c.cluster_id, json.dumps(c.labels), c.pattern, len(c.items),
                        json.dumps([it.item_id for it in c.items]), "open", None, None))
        for g in rest:
            reason = ("below cluster size" if len(g) < ccfg["min_cluster_for_rule"]
                      else f"beyond top {ccfg['max_clusters_per_run']} clusters")
            db.executemany("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                           [(run_id, "item", json.dumps({"item_id": it.item_id,
                                                         "text": it.text}), reason) for it in g])
        db.executemany("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                       [(run_id, "item", json.dumps({"item_id": it.item_id, "text": it.text}),
                         "low confidence") for it in low])
        db.commit()
        self.ctx = RunContext(run_id, self.text_source, self.docs, self.votes, self.agents,
                              {c.cluster_id: c for c in self.clusters}, embed, self.version)
        self.stats = {"accepted_words": accepted, "low_confidence_spans": len(low),
                      "disagreement_spans": len(items), "cluster_stats": cstats}
        self._time("cluster", t0)
        console.print(f"cluster: {accepted} accepted words, {len(low)} low-confidence spans, "
                      f"{len(items)} disagreement spans, {len(groups)} groups, "
                      f"{len(self.clusters)} clusters open, "
                      f"{len(low) + sum(len(g) for g in rest)} items to review queue")
        return self.clusters

    def get(self, cluster_id: int) -> Cluster:
        c = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
        if c is None:
            raise ValueError(f"no cluster {cluster_id}; open clusters: "
                             f"{[c.cluster_id for c in self.clusters]}")
        return c

    def diagnose(self, c: Cluster, note: str = "") -> Diagnosis | None:
        t0 = time.perf_counter()
        self.ctx.view = "all"
        res = run_agent(build_agent("diagnoser", output_schema=Diagnosis),
                        f"Diagnose cluster {c.cluster_id}: {len(c.items)} items, labels voted "
                        f"{list(c.labels)}, text pattern '{c.pattern}'."
                        + (f" Note from the orchestrator: {note}" if note else "")
                        + f"\n\nThe cluster:\n{cluster_view(self.ctx, c.cluster_id)}",
                        registry=build_registry(self.ctx), tracer=self.tracer, skills=self.skills)
        d = res.output
        self.diagnoses[c.cluster_id] = d
        self.forced += res.stop_reason == "forced_final"
        self.db.execute("UPDATE clusters SET diagnosis = ?, diagnosis_reason = ?, status = ? "
                        "WHERE run_id = ? AND cluster_id = ?",
                        (d.diagnosis if d else "FAILED", d.reason if d else res.error,
                         "diagnosed" if d else "diagnosis_failed", self.run_id, c.cluster_id))
        if d is None or d.diagnosis != "GUIDELINE_GAP":
            self.db.execute("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                            (self.run_id, "cluster", json.dumps({"cluster_id": c.cluster_id}),
                             d.diagnosis if d else "diagnosis failed"))
        self.db.commit()
        self._time("diagnose", t0)
        console.print(f"  cluster {c.cluster_id} ({len(c.items)} items, {c.pattern!r}, "
                      f"{list(c.labels)}): {d.diagnosis if d else 'FAILED'}"
                      + (" (forced at max_steps)" if res.stop_reason == "forced_final" else ""))
        return d

    def write(self, c: Cluster, note: str = "") -> Candidate | None:
        d = self.diagnoses.get(c.cluster_id)
        if d is None or d.diagnosis != "GUIDELINE_GAP":
            raise ValueError(f"cluster {c.cluster_id} is not diagnosed GUIDELINE_GAP "
                             f"({d.diagnosis if d else 'not diagnosed'})")
        t0 = time.perf_counter()
        self.ctx.view = "visible"
        visible = {it.item_id: it for it in c.visible}
        writer = build_agent("rule_writer", output_schema=draft_model(set(visible),
                                                                      sections(self.base_text)))
        res = run_agent(writer,
                        f"Write a rule for cluster {c.cluster_id} (diagnosed GUIDELINE_GAP: "
                        f"{d.reason}). Labels voted {list(c.labels)}, text pattern "
                        f"'{c.pattern}'." + (f" Note from the orchestrator: {note}" if note else "")
                        + f"\n\nThe cluster (the part you may use):\n"
                        f"{cluster_view(self.ctx, c.cluster_id)}\n\nThe labelers' instructions "
                        f"(your rule must not contradict them):\n"
                        f"{labeler_instructions(self.agents[0])}",
                        registry=build_registry(self.ctx), tracer=self.tracer, skills=self.skills)
        self._time("write", t0)
        if res.output is None:
            console.print(f"  cluster {c.cluster_id}: rule writer failed ({res.stop_reason})")
            return None
        draft: PatchDraft = res.output
        number = next_rule_number(self.base_text, draft.section, self.taken)
        self.taken.add(number)
        examples = [visible[i].text for i in draft.example_item_ids]
        block = render_patch(number, draft.rule_text, examples)
        pid = f"{self.run_id}-c{c.cluster_id}"
        skill_root = write_skill_text(with_patches(self.base_text, [(draft.section, block)]),
                                      f"{self.version}+{number}", self.out_dir / pid / "skills")
        cand = Candidate(pid, c, draft, number, block, SkillLibrary(skill_root))
        self.candidates[c.cluster_id] = cand
        console.print(f"  patch {number} for cluster {c.cluster_id}: {draft.rule_text[:100]}")
        return cand

    def verify(self, cands: list[Candidate]) -> None:
        """Advisory verification. The no-op control runs once per session."""
        if not cands:
            return
        t0 = time.perf_counter()
        control_skills = None
        if not self.control:
            cnum = next_rule_number(self.base_text, CONTROL_SECTION, self.taken)
            cblock = f"### {cnum} Rule\n{CONTROL_RULE}\n"
            croot = write_skill_text(with_patches(self.base_text, [(CONTROL_SECTION, cblock)]),
                                     f"{self.version}+control", self.out_dir / "control" / "skills")
            control_skills = SkillLibrary(croot)
        control = verify(cands, self.docs, self.text_source, self.agents, self.tracer,
                         control_skills)
        if control_skills is not None:
            self.control = {**control, "rule": cblock}
        self._time("verify", t0)

    def submit(self, cand: Candidate) -> None:
        """Record a patch as "proposed" and put it in the review queue."""
        now = datetime.now(UTC).isoformat()
        v = cand.verification
        record = {
            "patch_id": cand.patch_id, "rule_number": cand.number, "section": cand.draft.section,
            "rule_text": cand.draft.rule_text,
            "examples": [next(it.text for it in cand.cluster.visible if it.item_id == i)
                         for i in cand.draft.example_item_ids],
            "rationale": cand.draft.rationale,
            "affected_count": sum(len(it.words) for it in cand.cluster.items),
            "held_out": cand.cluster.held_out, "status": "proposed", "verification": v,
            "guideline_version": self.version, "created_at": now,
        }
        self.db.execute("INSERT OR REPLACE INTO patches VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (cand.patch_id, self.run_id, cand.cluster.cluster_id, cand.number,
                         cand.draft.section, cand.draft.rule_text,
                         json.dumps(record["examples"]), cand.draft.rationale,
                         record["affected_count"], int(cand.cluster.held_out), "proposed",
                         json.dumps(v), now, None, None))
        self.db.execute("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                        (self.run_id, "patch", json.dumps({"patch_id": cand.patch_id}),
                         f"proposed rule {cand.number}; verifier {v.get('verdict', 'not run')}"))
        self.db.commit()
        (self.out_dir / f"{cand.patch_id}.json").write_text(json.dumps(record, indent=1,
                                                                       ensure_ascii=False))
        cand.submitted = True

    def finish(self, extra: dict | None = None) -> dict:
        guard = default_guard()
        for m in guard.loaded():
            guard.unload(m)
        submitted = [c for c in self.candidates.values() if c.submitted]
        result = {
            "run_id": self.run_id,
            "timings_s": {k: round(v, 1) for k, v in self.timings.items()},
            **self.stats,
            "clusters": [{"cluster_id": c.cluster_id, "size": len(c.items), "pattern": c.pattern,
                          "labels": list(c.labels), "held_out": c.held_out,
                          "diagnosis": d.diagnosis if (d := self.diagnoses.get(c.cluster_id))
                          else None} for c in self.clusters],
            "diagnosed": len(self.diagnoses),
            "diagnoses_forced_at_max_steps": self.forced,
            "control": self.control,
            "patches": [{"patch_id": c.patch_id, "rule_number": c.number, **c.verification}
                        for c in submitted],
            **(extra or {}),
        }
        (self.out_dir / "improve.json").write_text(json.dumps(result, indent=1,
                                                              ensure_ascii=False))
        console.print(f"timings (s): {result['timings_s']}; diagnoses forced at max_steps: "
                      f"{self.forced}/{len(self.diagnoses)}")
        return result


def improve(run_id: str) -> dict:
    """Plain-code path (fallback): diagnose every open cluster, write rules for the gaps,
    verify them all, submit them all."""
    s = ImproveSession(run_id)
    for c in s.cluster():
        s.diagnose(c)
    cands = [cand for c in s.clusters
             if (d := s.diagnoses.get(c.cluster_id)) and d.diagnosis == "GUIDELINE_GAP"
             and (cand := s.write(c)) is not None]
    s.verify(cands)
    for cand in cands:
        s.submit(cand)
    return s.finish({"orchestrator": "code"})


def verify(candidates: list[Candidate], docs: dict[str, LabelDoc], text_source: str,
           agents: list[str], tracer: Tracer, control_skills: SkillLibrary | None) -> dict:
    """Relabel the gold set (baseline cached) and each hold-out under the current guideline and
    under guideline + patch. Labelers are the outer loop: each model loads once."""
    th = thresholds()
    labels = labeler_labels()
    gdocs = gold_docs(text_source)  # type: ignore[arg-type]
    base_skills = SkillLibrary()
    cache = ROOT / load_config()["harness"]["runs_dir"] / "gold_cache" / \
        f"{_cache_key(text_source, agents)}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    held = {c.patch_id: _item_words(c.cluster.held) for c in candidates}
    held_docs = {pid: _held_receipts(docs, w) for pid, w in held.items()}

    gold_before: dict[str, WordLabels] = {}
    gold_after: dict[str, dict[str, WordLabels]] = {c.patch_id: {} for c in candidates}
    held_before: dict[str, dict[str, WordLabels]] = {c.patch_id: {} for c in candidates}
    held_after: dict[str, dict[str, WordLabels]] = {c.patch_id: {} for c in candidates}
    gold_control: dict[str, WordLabels] = {}
    for name in agents:  # one model load per labeler for the whole stage
        agent = labeler_agent(name, labels)
        st = LabelerStats(name, agent.model)
        if name in cached:
            gold_before[name] = {tuple(json.loads(k)): v for k, v in cached[name].items()}
        else:
            gold_before[name] = _label_docs(agent, gdocs, labels, tracer, base_skills, st)
            cached[name] = {json.dumps(list(k)): v for k, v in gold_before[name].items()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached))
        if control_skills is not None:
            gold_control[name] = _label_docs(agent, gdocs, labels, tracer, control_skills, st)
        for c in candidates:
            gold_after[c.patch_id][name] = _label_docs(agent, gdocs, labels, tracer, c.skills, st)
            held_before[c.patch_id][name] = _label_docs(agent, held_docs[c.patch_id], labels,
                                                        tracer, base_skills, st)
            held_after[c.patch_id][name] = _label_docs(agent, held_docs[c.patch_id], labels,
                                                       tracer, c.skills, st)
        console.print(f"  verify: {name} done ({st.calls} calls, {st.seconds:.0f}s)")

    gold_classes = gold_label_set()
    acc_before = {a: gold_accuracy(gdocs, gold_before[a])[0] for a in agents}
    control: dict = {}
    if gold_control:
        acc_control = {a: gold_accuracy(gdocs, gold_control[a])[0] for a in agents}
        mean_c = sum(acc_control.values()) / len(agents)
        mean_b0 = sum(acc_before.values()) / len(agents)
        control = {"gold_accuracy_before": round(mean_b0, 4),
                   "gold_accuracy_after": round(mean_c, 4),
                   "gold_accuracy_by_labeler": {a: [round(acc_before[a], 4),
                                                    round(acc_control[a], 4)] for a in agents}}
        console.print(f"  control (no-op rule): gold {mean_b0:.3f} -> {mean_c:.3f} "
                      f"({ {a: round(acc_control[a] - acc_before[a], 3) for a in agents} })")
    for c in candidates:
        acc_after = {a: gold_accuracy(gdocs, gold_after[c.patch_id][a])[0] for a in agents}
        mean_b = sum(acc_before.values()) / len(agents)
        mean_a = sum(acc_after.values()) / len(agents)
        a_before = words_alpha(held[c.patch_id], held_before[c.patch_id])
        a_after = words_alpha(held[c.patch_id], held_after[c.patch_id])
        reasons = []
        if mean_b - mean_a > th["max_gold_accuracy_drop"]:
            reasons.append(f"gold accuracy dropped {mean_b - mean_a:.3f}")
        if a_after - a_before < th["min_cluster_alpha_gain"]:
            reasons.append(f"cluster alpha gain {a_after - a_before:+.3f} < "
                           f"{th['min_cluster_alpha_gain']}")
        c.verification = {  # advisory (decision 2026-11-09): the human decides
            "verdict": "fail" if reasons else "pass",
            "fail_reasons": reasons,
            "gold_accuracy_before": round(mean_b, 4), "gold_accuracy_after": round(mean_a, 4),
            "gold_accuracy_by_labeler": {a: [round(acc_before[a], 4), round(acc_after[a], 4)]
                                         for a in agents},
            "cluster_alpha_before": round(a_before, 4), "cluster_alpha_after": round(a_after, 4),
            "scored_on": "held-out part" if c.cluster.held_out else "whole cluster (not held out)",
            "held_out_words": len(held[c.patch_id]),
            "gold_verified": bool((set(c.cluster.labels) - {"other"}) & gold_classes),
        }
        console.print(f"  patch {c.number}: verifier {c.verification['verdict']} "
                      f"(gold {mean_b:.3f} -> {mean_a:.3f}, cluster alpha "
                      f"{a_before:.3f} -> {a_after:.3f}) {'; '.join(reasons)}")
    return control
