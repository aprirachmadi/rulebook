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
from rulebook.harness.agents import build_agent
from rulebook.harness.loop import run as run_agent
from rulebook.harness.memory import default_guard
from rulebook.harness.model import ModelError
from rulebook.harness.skills import SkillLibrary
from rulebook.harness.trace import Tracer
from rulebook.tools.guideline import sync_skill
from rulebook.tools.labels import LabelDoc, TextSource, batch_model, gold_doc, ocr_doc, render
from rulebook.tools.ocr import ocr_cached
from rulebook.tools.stats import agreement_pattern, alpha, alpha_per_label

LABELERS = ["labeler_a", "labeler_b", "labeler_c"]
console = Console()


@dataclass
class LabelerStats:
    agent: str
    model: str
    ok: int = 0
    failed: int = 0
    repaired: int = 0
    seconds: float = 0.0
    tokens: int = 0
    warnings: list[str] = field(default_factory=list)


def build_docs(text_source: TextSource, batch_size: int, offset: int) -> list[LabelDoc]:
    work = load_config()["data"]["splits"]["work"]
    receipts = load_split(work)[offset : offset + batch_size]  # first N by id (dev runs)
    if text_source == "gold":
        return [gold_doc(r) for r in receipts]
    return [ocr_doc(r, ocr_cached(r)) for r in receipts]


def run_batch(text_source: TextSource, batch_size: int, offset: int = 0) -> dict:
    version = sync_skill()  # cord-guideline skill = current guideline version
    tracer = Tracer()
    run_id = tracer.run_id
    docs = build_docs(text_source, batch_size, offset)
    db = connect()
    save_run(db, run_id, text_source, version, docs)
    labels = label_space()
    skills = SkillLibrary()
    console.print(f"run {run_id}: {len(docs)} receipts, text={text_source}, guideline {version}")

    stats: list[LabelerStats] = []
    for name in LABELERS:  # one model at a time; the memory guard unloads the previous one
        base = build_agent(name, output_schema=batch_model(docs[0], labels))
        base.system_prompt = base.system_prompt.replace("{LABELS}", ", ".join(labels))
        st = LabelerStats(name, base.model)
        for doc in docs:
            base.output_schema = batch_model(doc, labels)
            t0 = time.perf_counter()
            try:
                res = run_agent(base, render(doc), tracer=tracer, skills=skills)
            except ModelError as e:
                st.failed += 1
                save_failure(db, run_id, doc.receipt_id, name, str(e))
                continue
            st.seconds += time.perf_counter() - t0
            st.tokens += res.prompt_tokens + res.completion_tokens
            st.warnings += res.warnings
            st.repaired += len(res.messages) > 3  # system, task, answer; more = repair retry
            if res.raw_failures:  # keep rejected answers in full for debugging
                raw = ROOT / load_config()["harness"]["runs_dir"] / run_id / "raw"
                raw.mkdir(parents=True, exist_ok=True)
                (raw / f"{name}_{doc.receipt_id}.json").write_text(
                    json.dumps(res.raw_failures, ensure_ascii=False, indent=1), encoding="utf-8")
            if res.output is None:
                st.failed += 1
                save_failure(db, run_id, doc.receipt_id, name, res.error or res.stop_reason)
                continue
            st.ok += 1
            save_votes(db, run_id, doc, name, base.model, version, res.output)
        console.print(f"  {name} ({base.model}): ok {st.ok}, failed {st.failed}, "
                      f"repaired {st.repaired}, {st.seconds:.0f}s, {st.tokens} tokens")
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
        "alpha_word_level": alpha(units, LABELERS),
        "agreement_pattern": agreement_pattern(units, LABELERS),
        "alpha_per_label": alpha_per_label(units, LABELERS),
        "labelers": [st.__dict__ for st in stats],
    }
    out = ROOT / load_config()["harness"]["runs_dir"] / run_id / "summary.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    console.print(f"  word-level alpha: {summary['alpha_word_level']}")
    console.print(f"  summary -> {out}")
    return summary
