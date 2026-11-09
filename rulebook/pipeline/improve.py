"""M5 on a labeled run: cluster -> diagnose -> write -> verify, with timings per stage.

Clustering, diagnosis and rule writing read only labeler votes (never v2). Verification reads
gold labels only through `rulebook.tools.verify` (the sanctioned gold-set route). Patches stop
at "verified" / "rejected"; approval and applying them is Milestone 6.
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


def improve(run_id: str) -> dict:
    cfg = load_config()
    ccfg = cfg["clustering"]
    db = connect()
    text_source, version = db.execute(
        "SELECT text_source, guideline_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    docs = {d.receipt_id: d for d in load_docs(db, run_id)}
    votes = load_votes_full(db, run_id)
    summary = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())
    agents = summary.get("labelers_run", LABELERS)
    tracer = Tracer(run_id=f"{run_id}-improve")
    timings: dict[str, float] = {}
    out_dir = ROOT / cfg["harness"]["runs_dir"] / run_id / "improve"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- cluster ---
    t0 = time.perf_counter()
    embedder = ModelClient(cfg["models"]["embedding"])
    embed = lambda texts: np.asarray(embedder.embed(texts), dtype=float)  # noqa: E731
    accepted, items = disagreement_spans(list(docs.values()), votes, agents,
                                         cfg["pipeline"]["accept_min_confidence"])
    low, items = split_low_confidence(items, agents)
    groups, cstats = cluster_items(items, embed, ccfg)
    clusters, rest = make_clusters(groups, ccfg)
    db.execute("DELETE FROM clusters WHERE run_id = ?", (run_id,))
    db.execute("DELETE FROM review_queue WHERE run_id = ?", (run_id,))
    for c in clusters:
        db.execute("INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (run_id, c.cluster_id, json.dumps(c.labels), c.pattern, len(c.items),
                    json.dumps([it.item_id for it in c.items]), "open", None, None))
    for g in rest:
        reason = ("below cluster size" if len(g) < ccfg["min_cluster_for_rule"]
                  else f"beyond top {ccfg['max_clusters_per_run']} clusters")
        db.executemany("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                       [(run_id, "item", json.dumps({"item_id": it.item_id, "text": it.text}),
                         reason) for it in g])
    db.executemany("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                   [(run_id, "item", json.dumps({"item_id": it.item_id, "text": it.text}),
                     "low confidence") for it in low])
    db.commit()
    timings["cluster"] = time.perf_counter() - t0
    console.print(f"cluster: {accepted} accepted words, {len(low)} low-confidence spans, "
                  f"{len(items)} disagreement spans, {len(groups)} groups, "
                  f"{len(clusters)} clusters processed, {len(low) + sum(len(g) for g in rest)} "
                  f"items to review queue ({timings['cluster']:.0f}s)")

    ctx = RunContext(run_id, text_source, docs, votes, agents,
                     {c.cluster_id: c for c in clusters}, embed, version)
    skills = SkillLibrary()

    # --- diagnose ---
    t0 = time.perf_counter()
    diagnoser = build_agent("diagnoser", output_schema=Diagnosis)
    diagnoses: dict[int, Diagnosis | None] = {}
    forced = 0  # diagnoses that hit max_steps and were forced
    for c in clusters:
        ctx.view = "all"
        res = run_agent(diagnoser,
                        f"Diagnose cluster {c.cluster_id}: {len(c.items)} items, labels voted "
                        f"{list(c.labels)}, text pattern '{c.pattern}'.\n\nThe cluster:\n"
                        f"{cluster_view(ctx, c.cluster_id)}",
                        registry=build_registry(ctx), tracer=tracer, skills=skills)
        diagnoses[c.cluster_id] = res.output
        forced += res.stop_reason == "forced_final"
        d = res.output
        db.execute("UPDATE clusters SET diagnosis = ?, diagnosis_reason = ?, status = ? "
                   "WHERE run_id = ? AND cluster_id = ?",
                   (d.diagnosis if d else "FAILED", d.reason if d else res.error,
                    "diagnosed" if d else "diagnosis_failed", run_id, c.cluster_id))
        if d is None or d.diagnosis != "GUIDELINE_GAP":
            db.execute("INSERT INTO review_queue VALUES (?, ?, ?, ?)",
                       (run_id, "cluster", json.dumps({"cluster_id": c.cluster_id}),
                        d.diagnosis if d else "diagnosis failed"))
        db.commit()
        console.print(f"  cluster {c.cluster_id} ({len(c.items)} items, {c.pattern!r}, "
                      f"{list(c.labels)}): {d.diagnosis if d else 'FAILED'}")
    timings["diagnose"] = time.perf_counter() - t0
    console.print(f"  diagnoses forced at max_steps: {forced}/{len(clusters)}")

    # --- write ---
    t0 = time.perf_counter()
    base_text = guideline_text(version)
    taken: set[str] = set()
    candidates: list[Candidate] = []
    gaps = [c for c in clusters
            if (d := diagnoses[c.cluster_id]) and d.diagnosis == "GUIDELINE_GAP"]
    labeler_rules = labeler_instructions(agents[0])
    for c in gaps:
        ctx.view = "visible"
        visible = {it.item_id: it for it in c.visible}
        writer = build_agent("rule_writer", output_schema=draft_model(set(visible),
                                                                      sections(base_text)))
        res = run_agent(writer,
                        f"Write a rule for cluster {c.cluster_id} (diagnosed GUIDELINE_GAP: "
                        f"{diagnoses[c.cluster_id].reason}). Labels voted {list(c.labels)}, "
                        f"text pattern '{c.pattern}'.\n\nThe cluster (the part you may use):\n"
                        f"{cluster_view(ctx, c.cluster_id)}\n\nThe labelers' instructions "
                        f"(your rule must not contradict them):\n{labeler_rules}",
                        registry=build_registry(ctx), tracer=tracer, skills=skills)
        if res.output is None:
            console.print(f"  cluster {c.cluster_id}: rule writer failed ({res.stop_reason})")
            continue
        draft: PatchDraft = res.output
        number = next_rule_number(base_text, draft.section, taken)
        taken.add(number)
        examples = [visible[i].text for i in draft.example_item_ids]
        block = render_patch(number, draft.rule_text, examples)
        pid = f"{run_id}-c{c.cluster_id}"
        skill_root = write_skill_text(with_patches(base_text, [(draft.section, block)]),
                                      f"{version}+{number}", out_dir / pid / "skills")
        candidates.append(Candidate(pid, c, draft, number, block, SkillLibrary(skill_root)))
        console.print(f"  patch {number} for cluster {c.cluster_id}: {draft.rule_text[:100]}")
    timings["write"] = time.perf_counter() - t0

    # --- verify ---
    t0 = time.perf_counter()
    control: dict = {}
    if candidates:
        cnum = next_rule_number(base_text, CONTROL_SECTION, taken)
        cblock = f"### {cnum} Rule\n{CONTROL_RULE}\n"
        croot = write_skill_text(with_patches(base_text, [(CONTROL_SECTION, cblock)]),
                                 f"{version}+control", out_dir / "control" / "skills")
        control = verify(candidates, docs, text_source, agents, tracer, SkillLibrary(croot))
        control["rule"] = cblock
    timings["verify"] = time.perf_counter() - t0
    guard = default_guard()
    for m in guard.loaded():
        guard.unload(m)

    now = datetime.now(UTC).isoformat()
    for cand in candidates:
        v = cand.verification
        record = {
            "patch_id": cand.patch_id, "rule_number": cand.number, "section": cand.draft.section,
            "rule_text": cand.draft.rule_text,
            "examples": [next(it.text for it in cand.cluster.visible if it.item_id == i)
                         for i in cand.draft.example_item_ids],
            "rationale": cand.draft.rationale,
            "affected_count": sum(len(it.words) for it in cand.cluster.items),
            "held_out": cand.cluster.held_out, "status": v["status"], "verification": v,
            "guideline_version": version, "created_at": now,
        }
        db.execute("INSERT OR REPLACE INTO patches VALUES "
                   "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (cand.patch_id, run_id, cand.cluster.cluster_id, cand.number,
                    cand.draft.section, cand.draft.rule_text, json.dumps(record["examples"]),
                    cand.draft.rationale, record["affected_count"], int(cand.cluster.held_out),
                    v["status"], json.dumps(v), now, None, None))
        (out_dir / f"{cand.patch_id}.json").write_text(json.dumps(record, indent=1,
                                                                  ensure_ascii=False))
        if v["status"] == "verified":
            pdir = ROOT / "guidelines" / "patches"
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / f"{cand.patch_id}.json").write_text(json.dumps(record, indent=1,
                                                                   ensure_ascii=False))
    db.commit()

    result = {
        "run_id": run_id, "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "accepted_words": accepted, "low_confidence_spans": len(low),
        "disagreement_spans": len(items), "cluster_stats": cstats,
        "clusters": [{"cluster_id": c.cluster_id, "size": len(c.items), "pattern": c.pattern,
                      "labels": list(c.labels), "held_out": c.held_out,
                      "diagnosis": d.diagnosis if (d := diagnoses[c.cluster_id]) else None}
                     for c in clusters],
        "diagnoses_forced_at_max_steps": forced,
        "control": control,
        "patches": [{"patch_id": c.patch_id, "rule_number": c.number, **c.verification}
                    for c in candidates],
    }
    (out_dir / "improve.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    console.print(f"timings (s): {result['timings_s']}")
    return result


def verify(candidates: list[Candidate], docs: dict[str, LabelDoc], text_source: str,
           agents: list[str], tracer: Tracer, control_skills: SkillLibrary) -> dict:
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
    acc_control = {a: gold_accuracy(gdocs, gold_control[a])[0] for a in agents}
    mean_c = sum(acc_control.values()) / len(agents)
    mean_b0 = sum(acc_before.values()) / len(agents)
    control = {"gold_accuracy_before": round(mean_b0, 4), "gold_accuracy_after": round(mean_c, 4),
               "gold_accuracy_by_labeler": {a: [round(acc_before[a], 4), round(acc_control[a], 4)]
                                            for a in agents}}
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
        c.verification = {
            "status": "rejected" if reasons else "verified",
            "reject_reasons": reasons,
            "gold_accuracy_before": round(mean_b, 4), "gold_accuracy_after": round(mean_a, 4),
            "gold_accuracy_by_labeler": {a: [round(acc_before[a], 4), round(acc_after[a], 4)]
                                         for a in agents},
            "cluster_alpha_before": round(a_before, 4), "cluster_alpha_after": round(a_after, 4),
            "scored_on": "held-out part" if c.cluster.held_out else "whole cluster (not held out)",
            "held_out_words": len(held[c.patch_id]),
            "gold_verified": bool((set(c.cluster.labels) - {"other"}) & gold_classes),
        }
        console.print(f"  patch {c.number}: {c.verification['status']} "
                      f"(gold {mean_b:.3f} -> {mean_a:.3f}, cluster alpha "
                      f"{a_before:.3f} -> {a_after:.3f}) {'; '.join(reasons)}")
    return control
