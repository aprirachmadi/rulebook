"""After a run: words where all labelers agree but v2 disagrees, split by cause.

- OCR misread: the OCR word's text differs from its matched CORD word (case and spaces
  ignored). The labelers saw different text from what the CORD annotator labeled.
- Shared mistake: the text is the same, and all labelers gave the same wrong label.

Writes reports/run_<id>_agree_wrong.md.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from eval.answer_key import load_answer_key
from rulebook.config import ROOT, load_config
from rulebook.data.cord import load_split
from rulebook.data.store import connect, load_docs, load_votes
from rulebook.pipeline.labeling import LABELERS

OTHER = load_config()["labeling"]["other_label"]
TOP_PAIRS = 15
EXAMPLES = 3


def _same(a: str, b: str) -> bool:
    return "".join(a.lower().split()) == "".join(b.lower().split())


def report(run_id: str, out_dir: Path | None = None) -> str:
    db = connect()
    docs = load_docs(db, run_id)
    votes = load_votes(db, run_id)
    split = load_config()["data"]["splits"]["work"]
    key = load_answer_key(split)
    ids = {d.receipt_id for d in docs}
    cord = {r.receipt_id: r for r in load_split(split) if r.receipt_id in ids}
    summary = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())
    agents = summary.get("labelers_run", LABELERS)

    agree = 0
    misread: list[tuple[str, str, str, str]] = []  # (ocr text, cord text, voted, v2)
    shared: list[tuple[str, str, str]] = []  # (text, voted, v2)
    for d in docs:
        for ln in d.lines:
            for wi, w in enumerate(ln.words):
                v = votes.get((d.receipt_id, ln.line_idx, wi), {})
                if w.ref is None or len(v) != len(agents) or len(set(v.values())) != 1:
                    continue
                agree += 1
                voted, gold = next(iter(v.values())), key[(d.receipt_id, w.ref[0])]
                if voted == gold:
                    continue
                cline = next(c for c in cord[d.receipt_id].lines if c.line_idx == w.ref[0])
                ctext = cline.words[w.ref[1]].text
                if _same(w.text, ctext):
                    shared.append((w.text, voted, gold))
                else:
                    misread.append((w.text, ctext, voted, gold))

    wrong = len(misread) + len(shared)
    pairs = Counter((gold, voted) for _, voted, gold in shared)
    examples: dict[tuple[str, str], list[str]] = {}
    for text, voted, gold in shared:
        examples.setdefault((gold, voted), []).append(text)
    pct = lambda a, b: f"{a / b:.1%}" if b else "-"  # noqa: E731
    out = [
        f"# Run {run_id}: all labelers agree, v2 disagrees",
        "",
        f"Text source: {summary.get('text_source', '?')}. Labelers: {', '.join(agents)}. "
        f"CORD-matched words where all {len(agents)} labelers gave the same label: {agree}; "
        f"of those, v2 disagrees on **{wrong}**.",
        "",
        "| cause | words | share of the wrong ones |",
        "|---|---|---|",
        f"| OCR misread (OCR text differs from the CORD word) | {len(misread)} | "
        f"{pct(len(misread), wrong)} |",
        f"| shared mistake (same text, same wrong label from all) | {len(shared)} | "
        f"{pct(len(shared), wrong)} |",
        "",
        f"Shared mistakes where the shared label is `{OTHER}`: "
        f"{sum(voted == OTHER for _, voted, _ in shared)}.",
        "",
        "## Top confusion pairs among shared mistakes",
        "",
        "| v2 label | labelers' label | words | examples |",
        "|---|---|---|---|",
    ]
    for (gold, voted), n in pairs.most_common(TOP_PAIRS):
        ex = ", ".join(f"`{t}`" for t in dict.fromkeys(examples[(gold, voted)]))
        ex = ", ".join(ex.split(", ")[:EXAMPLES])
        out.append(f"| {gold} | {voted} | {n} | {ex} |")
    out += ["", "## OCR misreads (OCR text → CORD text, voted / v2)", ""]
    out += [f"- `{o}` → `{c}`: {voted} / {gold}" for o, c, voted, gold in misread]
    text = "\n".join(out) + "\n"
    path = (out_dir or ROOT / "reports") / f"run_{run_id}_agree_wrong.md"
    path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(report(sys.argv[1]))
