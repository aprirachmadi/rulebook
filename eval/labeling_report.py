"""After a run: word-level agreement of each labeler with v2 -> reports/run_<id>_labeling.md."""

from __future__ import annotations

import json
import sys
from collections import Counter

from eval.answer_key import load_answer_key
from rulebook.config import ROOT, load_config
from rulebook.data.store import connect, load_docs, load_votes
from rulebook.pipeline.labeling import LABELERS

OTHER = load_config()["labeling"]["other_label"]


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
    voted = Counter()  # matched words labeled with a CORD label (scored)
    other_matched = Counter()  # matched words labeled `other` (excluded from accuracy)
    other_unmatched = Counter()  # OCR-miss words labeled `other`
    labeled_any = Counter()
    majority_ok = majority_n = accepted_ok = accepted_n = 0
    confusions: Counter[tuple[str, str]] = Counter()
    for d in docs:
        for ln in d.lines:
            for wi, w in enumerate(ln.words):
                total += 1
                v = votes.get((d.receipt_id, ln.line_idx, wi), {})
                for a in v:
                    labeled_any[a] += 1
                if w.ref is None:  # OCR miss: no CORD word to score against
                    for a in v:
                        other_unmatched[a] += v[a] == OTHER
                    continue
                matched += 1
                gold = key[(d.receipt_id, w.ref[0])]
                for a in LABELERS:
                    if a in v and v[a] == OTHER:
                        other_matched[a] += 1
                    elif a in v:
                        voted[a] += 1
                        correct[a] += v[a] == gold
                        if v[a] != gold:
                            confusions[(gold, v[a])] += 1
                v = {a: lab for a, lab in v.items() if lab != OTHER}
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
        "| labeler | model | complete receipts | words labeled (of all) | agreement with v2 "
        "(excl. other) | `other` used (matched / OCR-miss words) |",
        "|---|---|---|---|---|---|",
    ]
    by_agent = {s["agent"]: s for s in summary["labelers"]}
    for a in LABELERS:
        other = other_matched[a] + other_unmatched[a]
        lines.append(
            f"| {a} | {models.get(a, '?')} | {by_agent[a].get('receipts_complete', '?')}/"
            f"{len(docs)} | {labeled_any[a]} / {total} | {pct(correct[a], voted[a])} "
            f"({voted[a]} words) | {pct(other, labeled_any[a])} "
            f"({other_matched[a]} / {other_unmatched[a]}) |")
    lines += [
        "",
        f"Majority vote (2+ agree, excl. other) vs v2: {pct(majority_ok, majority_n)} of "
        f"{majority_n} words.",
        f"All three agree (excl. other): {accepted_n} words, of which "
        f"{pct(accepted_ok, accepted_n)} match v2.",
        "",
        "Top 10 disagreements with v2 (v2 label -> labeler label, count over all labelers):",
        "",
    ]
    lines += [f"- {g} -> {p}: {n}" for (g, p), n in confusions.most_common(10)]
    lines += ["", "Labeler run stats:", ""]
    for s in summary["labelers"]:
        lines.append(f"- {s['agent']}: calls {s['calls']} (targeted repairs {s['repairs']}), "
                     f"words failed {s['words_failed']}, {s['seconds']:.0f}s, "
                     f"{s['tokens']} tokens")
    text = "\n".join(lines) + "\n"
    (ROOT / "reports" / f"run_{run_id}_labeling.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        print(report(rid))
