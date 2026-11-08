"""After a run: word-level agreement of each labeler with v2 -> reports/run_<id>_labeling.md.

Headline metric (decided 2026-11-08): agreement with v2 over CORD-matched words, counting
`other` on a matched word as wrong. Excluding `other` is reported as a secondary number.
OCR-miss words (no CORD match) are never scored.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval.answer_key import load_answer_key
from rulebook.config import ROOT, load_config
from rulebook.data.store import connect, load_docs, load_votes
from rulebook.pipeline.labeling import LABELERS

OTHER = load_config()["labeling"]["other_label"]


@dataclass
class Rate:
    hit: int = 0
    n: int = 0

    def add(self, ok: bool) -> None:
        self.hit += ok
        self.n += 1

    def __str__(self) -> str:
        return f"{self.hit / self.n:.1%} ({self.hit}/{self.n})" if self.n else "-"


def report(run_id: str, out_dir: Path | None = None) -> str:
    db = connect()
    text_source, version = db.execute(
        "SELECT text_source, guideline_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    docs = load_docs(db, run_id)
    votes = load_votes(db, run_id)
    key = load_answer_key(load_config()["data"]["splits"]["work"])
    summary = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())
    agents = summary.get("labelers_run", LABELERS)

    total = matched = 0
    headline = {a: Rate() for a in agents}  # other on a matched word = wrong
    excl_other = {a: Rate() for a in agents}  # other left out
    other_matched: Counter[str] = Counter()
    other_unmatched: Counter[str] = Counter()
    labeled_any: Counter[str] = Counter()
    all_agree, all_agree_x = Rate(), Rate()  # headline / excl. other
    majority, majority_x = Rate(), Rate()
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
                for a in agents:
                    if a not in v:
                        continue
                    headline[a].add(v[a] == gold)
                    if v[a] == OTHER:
                        other_matched[a] += 1
                    else:
                        excl_other[a].add(v[a] == gold)
                        if v[a] != gold:
                            confusions[(gold, v[a])] += 1
                if len(agents) < 2:
                    continue
                if len(v) == len(agents) and len(set(v.values())) == 1:
                    all_agree.add(next(iter(v.values())) == gold)
                top = Counter(v.values()).most_common(1)
                if top and top[0][1] >= 2:
                    majority.add(top[0][0] == gold)
                real = {a: lab for a, lab in v.items() if lab != OTHER}
                if len(real) == len(agents) and len(set(real.values())) == 1:
                    all_agree_x.add(next(iter(real.values())) == gold)
                top_x = Counter(real.values()).most_common(1)
                if top_x and top_x[0][1] >= 2:
                    majority_x.add(top_x[0][0] == gold)

    pct = lambda a, b: f"{a / b:.1%}" if b else "-"  # noqa: E731
    models = {s["agent"]: s["model"] for s in summary["labelers"]}
    by_agent = {s["agent"]: s for s in summary["labelers"]}
    alpha = summary["alpha_word_level"]
    lines = [
        f"# Run {run_id}: labeling vs v2 ({text_source} text, guideline {version})",
        "",
        f"Receipts: {len(docs)}. Words: {total}. Matched to a CORD word: {matched} "
        f"({pct(matched, total)}); OCR misses (not scored): {total - matched}.",
        f"Labelers in this run: {', '.join(agents)}.",
        "",
        f"**Word-level Krippendorff's alpha (nominal): {alpha:.3f}**" if alpha is not None
        else "Word-level alpha: n/a (needs 2+ labelers)",
        f"Agreement pattern: {summary['agreement_pattern']}",
        "",
        "| labeler | model | complete receipts | **agreement with v2** (other = wrong) | "
        "excl. other | `other` used (matched / OCR-miss words) |",
        "|---|---|---|---|---|---|",
    ]
    for a in agents:
        other = other_matched[a] + other_unmatched[a]
        lines.append(
            f"| {a} | {models.get(a, '?')} | {by_agent[a].get('receipts_complete', '?')}/"
            f"{len(docs)} | **{headline[a]}** | {excl_other[a]} | "
            f"{pct(other, labeled_any[a])} ({other_matched[a]} / {other_unmatched[a]}) |")
    lines += [
        "",
        "| agreement group (CORD-matched words) | matches v2 (other = wrong) | excl. other |",
        "|---|---|---|",
        f"| all labelers agree | {all_agree} | {all_agree_x} |",
        f"| majority (2+ agree) | {majority} | {majority_x} |",
        "",
        "Top 10 disagreements with v2 (v2 label -> labeler label, count over all labelers, "
        "excl. other):",
        "",
    ]
    lines += [f"- {g} -> {p}: {n}" for (g, p), n in confusions.most_common(10)]
    lines += ["", "Labeler run stats:", ""]
    for s in summary["labelers"]:
        lines.append(f"- {s['agent']}: calls {s['calls']} (targeted repairs {s['repairs']}), "
                     f"words failed {s['words_failed']}, {s['seconds']:.0f}s, "
                     f"{s['tokens']} tokens")
    text = "\n".join(lines) + "\n"
    (out_dir or ROOT / "reports").joinpath(f"run_{run_id}_labeling.md").write_text(
        text, encoding="utf-8")
    return text


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        print(report(rid))
