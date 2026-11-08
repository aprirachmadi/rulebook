"""After a run: word-level agreement of each labeler with v2 -> reports/run_<id>_labeling.md."""

from __future__ import annotations

import json
import sys
from collections import Counter

from eval.answer_key import load_answer_key
from rulebook.config import ROOT, load_config
from rulebook.data.store import connect, load_docs, load_votes
from rulebook.pipeline.labeling import LABELERS


def report(run_id: str) -> str:
    db = connect()
    text_source, version = db.execute(
        "SELECT text_source, guideline_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    docs = load_docs(db, run_id)
    votes = load_votes(db, run_id)
    key = load_answer_key(load_config()["data"]["splits"]["work"])
    summary = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())

    total = matched = 0
    correct = Counter()
    voted = Counter()
    majority_ok = majority_n = accepted_ok = accepted_n = 0
    confusions: Counter[tuple[str, str]] = Counter()
    for d in docs:
        for ln in d.lines:
            for wi, w in enumerate(ln.words):
                total += 1
                if w.ref is None:
                    continue  # OCR miss: no CORD word to score against
                matched += 1
                gold = key[(d.receipt_id, w.ref[0])]
                v = votes.get((d.receipt_id, ln.line_idx, wi), {})
                for a in LABELERS:
                    if a in v:
                        voted[a] += 1
                        correct[a] += v[a] == gold
                        if v[a] != gold:
                            confusions[(gold, v[a])] += 1
                top = Counter(v.values()).most_common(1)
                if top and top[0][1] >= 2:
                    majority_n += 1
                    majority_ok += top[0][0] == gold
                if len(v) == 3 and len(set(v.values())) == 1:
                    accepted_n += 1
                    accepted_ok += next(iter(v.values())) == gold

    pct = lambda a, b: f"{a / b:.1%}" if b else "-"  # noqa: E731
    models = {s["agent"]: s["model"] for s in summary["labelers"]}
    lines = [
        f"# Run {run_id}: labeling vs v2 ({text_source} text, guideline {version})",
        "",
        f"Receipts: {len(docs)}. Words: {total}. Matched to a CORD word: {matched} "
        f"({pct(matched, total)}); OCR misses: {total - matched}.",
        "",
        f"**Word-level Krippendorff's alpha (3 labelers, nominal): "
        f"{summary['alpha_word_level']:.3f}**" if summary["alpha_word_level"] is not None
        else "Alpha: n/a",
        f"Agreement pattern: {summary['agreement_pattern']}",
        "",
        "| labeler | model | words labeled (of matched) | agreement with v2 |",
        "|---|---|---|---|",
    ]
    for a in LABELERS:
        lines.append(f"| {a} | {models.get(a, '?')} | {voted[a]} / {matched} | "
                     f"{pct(correct[a], voted[a])} |")
    lines += [
        "",
        f"Majority vote (2+ agree) vs v2: {pct(majority_ok, majority_n)} of {majority_n} words.",
        f"All three agree: {accepted_n} words, of which {pct(accepted_ok, accepted_n)} match v2.",
        "",
        "Top 10 disagreements with v2 (v2 label -> labeler label, count over all labelers):",
        "",
    ]
    lines += [f"- {g} -> {p}: {n}" for (g, p), n in confusions.most_common(10)]
    lines += ["", "Labeler run stats:", ""]
    for s in summary["labelers"]:
        lines.append(f"- {s['agent']}: ok {s['ok']}, failed {s['failed']}, repaired "
                     f"{s['repaired']}, {s['seconds']:.0f}s, {s['tokens']} tokens")
    text = "\n".join(lines) + "\n"
    (ROOT / "reports" / f"run_{run_id}_labeling.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        print(report(rid))
