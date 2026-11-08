"""M4 batch: build docs (gold text or OCR), label with 3 labelers one model at a time, alpha.

Reads no v2 labels. Agreement with v2 is computed afterwards by `eval/labeling_report.py`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rich.console import Console

from rulebook.config import ROOT, load_config
from rulebook.data.cord import label_space, load_split
from rulebook.data.store import connect, load_votes, save_failure, save_run, save_votes
from rulebook.harness.agents import Agent, build_agent
from rulebook.harness.loop import run as run_agent
from rulebook.harness.memory import default_guard
from rulebook.harness.model import ModelError
from rulebook.harness.skills import SkillLibrary
from rulebook.harness.trace import Tracer
from rulebook.tools.guideline import sync_skill
from rulebook.tools.labels import (
    LabelDoc,
    Scope,
    Span,
    TextSource,
    batch_model,
    full_scope,
    gold_doc,
    ocr_doc,
    render,
    sort_spans,
)
from rulebook.tools.ocr import ocr_cached
from rulebook.tools.stats import agreement_pattern, alpha, alpha_per_label

LABELERS = ["labeler_a", "labeler_b", "labeler_c"]


def labeler_labels() -> tuple[str, ...]:
    """CORD categories (names only, from the work split) + the `other` label."""
    return (*label_space(), load_config()["labeling"]["other_label"])
console = Console()


@dataclass
class LabelerStats:
    agent: str
    model: str
    receipts_complete: int = 0  # every word got a label
    words_labeled: int = 0
    words_failed: int = 0
    calls: int = 0
    repairs: int = 0  # targeted repair calls
    seconds: float = 0.0
    tokens: int = 0
    warnings: list[str] = field(default_factory=list)


def chunks(doc: LabelDoc, size: int) -> list[list[int]]:
    """Consecutive line groups of about `size` lines."""
    ids = [ln.line_idx for ln in doc.lines]
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def _call(agent: Agent, task: str, scope: Scope, labels: tuple[str, ...], tracer: Tracer,
          skills: SkillLibrary, st: LabelerStats) -> list[Span]:
    agent.output_schema = batch_model(scope, labels)
    st.calls += 1
    t0 = time.perf_counter()
    try:
        res = run_agent(agent, task, tracer=tracer, skills=skills)
    except ModelError:
        return []
    finally:
        st.seconds += time.perf_counter() - t0
    st.tokens += res.prompt_tokens + res.completion_tokens
    st.warnings += res.warnings
    return list(res.output.labels) if res.output is not None else []


def label_doc(agent: Agent, doc: LabelDoc, labels: tuple[str, ...], chunk_lines: int,
              tracer: Tracer, skills: SkillLibrary, st: LabelerStats
              ) -> tuple[dict[tuple[int, int], Span], Scope]:
    """Label a receipt chunk by chunk; one targeted repair per chunk for words left over.
    Returns (word -> span, words that still have no label)."""
    accepted: dict[tuple[int, int], Span] = {}
    failed: Scope = {}
    for lines in chunks(doc, chunk_lines):
        scope = full_scope(doc, lines)
        spans = _call(agent, render(doc, scope), scope, labels, tracer, skills, st)
        good, todo, problems = sort_spans(scope, spans)
        accepted |= good
        if not todo:
            continue
        st.repairs += 1
        note = ("Some words were not labeled correctly (" + "; ".join(problems)[:300] + "). "
                "Label only the words marked below.")
        spans = _call(agent, render(doc, todo, note), todo, labels, tracer, skills, st)
        good, still, _ = sort_spans(todo, spans)
        accepted |= good
        for li, words in still.items():
            failed[li] = words
    return accepted, failed


def build_docs(text_source: TextSource, batch_size: int, offset: int) -> list[LabelDoc]:
    work = load_config()["data"]["splits"]["work"]
    receipts = load_split(work)[offset : offset + batch_size]  # first N by id (dev runs)
    if text_source == "gold":
        return [gold_doc(r) for r in receipts]
    return [ocr_doc(r, ocr_cached(r)) for r in receipts]


def run_batch(text_source: TextSource, batch_size: int, offset: int = 0,
              labelers: list[str] | None = None) -> dict:
    version = sync_skill()  # cord-guideline skill = current guideline version
    tracer = Tracer()
    run_id = tracer.run_id
    docs = build_docs(text_source, batch_size, offset)
    db = connect()
    save_run(db, run_id, text_source, version, docs)
    labels = labeler_labels()
    skills = SkillLibrary()
    console.print(f"run {run_id}: {len(docs)} receipts, text={text_source}, guideline {version}")

    chunk_lines = load_config()["labeling"]["chunk_lines"]
    stats: list[LabelerStats] = []
    agents = labelers or LABELERS
    for name in agents:  # one model at a time; the memory guard unloads the previous one
        agent = build_agent(name, output_schema=batch_model({0: [0]}, labels))
        agent.system_prompt = agent.system_prompt.replace("{LABELS}", ", ".join(labels))
        st = LabelerStats(name, agent.model)
        for doc in docs:
            got, failed = label_doc(agent, doc, labels, chunk_lines, tracer, skills, st)
            st.words_labeled += len(got)
            st.words_failed += sum(len(w) for w in failed.values())
            st.receipts_complete += not failed
            if failed:
                save_failure(db, run_id, doc.receipt_id, name,
                             f"unlabeled after repair (line: words): {failed}")
            save_votes(db, run_id, doc, name, agent.model, version, got)
        console.print(f"  {name} ({agent.model}): complete receipts {st.receipts_complete}/"
                      f"{len(docs)}, words labeled {st.words_labeled}, failed {st.words_failed}, "
                      f"calls {st.calls} (repairs {st.repairs}), {st.seconds:.0f}s, "
                      f"{st.tokens} tokens")
        stats.append(st)
    guard = default_guard()
    for m in guard.loaded():  # leave the GPU free
        guard.unload(m)

    units = list(load_votes(db, run_id).values())
    summary = {
        "run_id": run_id,
        "text_source": text_source,
        "guideline_version": version,
        "receipts": len(docs),
        "words": sum(len(ln.words) for d in docs for ln in d.lines),
        "labelers_run": agents,
        "alpha_word_level": alpha(units, agents),
        "agreement_pattern": agreement_pattern(units, agents),
        "alpha_per_label": alpha_per_label(units, agents),
        "labelers": [st.__dict__ for st in stats],
    }
    out = ROOT / load_config()["harness"]["runs_dir"] / run_id / "summary.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    console.print(f"  word-level alpha: {summary['alpha_word_level']}")
    console.print(f"  summary -> {out}")
    return summary
