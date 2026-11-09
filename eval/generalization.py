"""Generalization check: the same receipts labeled under two guideline versions, scored vs v2.

Usage: python -m eval.generalization RUN_BEFORE RUN_AFTER -> reports/generalization_<a>_<b>.md

Pass (decision 2026-11-09): majority-vote accuracy after >= before + `min_majority_gain`,
alpha after > before, review share after < before. Scoring follows eval/labeling_report.py:
CORD-matched words only, `other` on a matched word counts as wrong. Review share is over all
words: a word reaches a human unless every labeler voted the same label at
>= `accept_min_confidence` (the pipeline's "accepted").
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from eval.answer_key import load_answer_key
from rulebook.config import ROOT, load_config
from rulebook.data.store import connect, load_docs, load_votes_full


@dataclass
class Scores:
    run_id: str
    version: str
    receipts: int
    words: int
    matched: int
    alpha: float
    headline: dict[str, float] = field(default_factory=dict)
    majority: float = 0.0
    majority_n: int = 0
    all_agree: float = 0.0
    all_agree_n: int = 0
    review_share: float = 0.0


def score(run_id: str) -> Scores:
    cfg = load_config()
    db = connect()
    version, = db.execute("SELECT guideline_version FROM runs WHERE run_id = ?",
                          (run_id,)).fetchone()
    summary = json.loads((ROOT / cfg["harness"]["runs_dir"] / run_id / "summary.json")
                         .read_text())
    agents = summary["labelers_run"]
    key = load_answer_key(cfg["data"]["splits"]["work"])
    min_conf = cfg["pipeline"]["accept_min_confidence"]
    docs = load_docs(db, run_id)
    votes = load_votes_full(db, run_id)

    words = matched = review = 0
    hits, scored = Counter(), Counter()  # per labeler; a word it never voted on is left out
    maj_hit = maj_n = agree_hit = agree_n = 0
    for d in docs:
        for ln in d.lines:
            for wi, w in enumerate(ln.words):
                words += 1
                v = votes.get((d.receipt_id, ln.line_idx, wi), {})
                labels = {a: lab for a, (lab, _, _) in v.items()}
                accepted = (len(v) == len(agents) and len(set(labels.values())) == 1
                            and all(c >= min_conf for _, c, _ in v.values()))
                review += not accepted
                if w.ref is None:  # OCR miss: no CORD word to score against
                    continue
                matched += 1
                gold = key[(d.receipt_id, w.ref[0])]
                for a in labels:
                    hits[a] += labels[a] == gold
                    scored[a] += 1
                if len(labels) == len(agents) and len(set(labels.values())) == 1:
                    agree_n += 1
                    agree_hit += next(iter(labels.values())) == gold
                top = Counter(labels.values()).most_common(1)
                if top and top[0][1] >= 2:
                    maj_n += 1
                    maj_hit += top[0][0] == gold
    return Scores(run_id, version, len(docs), words, matched, summary["alpha_word_level"],
                  {a: hits[a] / scored[a] for a in agents}, maj_hit / maj_n, maj_n,
                  agree_hit / agree_n, agree_n, review / words)


def verdict(before: Scores, after: Scores, min_majority_gain: float) -> dict[str, bool]:
    return {
        f"majority-vote accuracy rises by >= {min_majority_gain * 100:.0f} points":
            after.majority >= before.majority + min_majority_gain - 1e-9,
        "alpha rises": after.alpha > before.alpha,
        "review share falls": after.review_share < before.review_share,
    }


def report(before_id: str, after_id: str, out_dir: Path | None = None) -> str:
    gain = load_config()["eval"]["generalization"]["min_majority_gain"]
    b, a = score(before_id), score(after_id)
    checks = verdict(b, a, gain)
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    pts = lambda x, y: f"{(y - x) * 100:+.1f} pts"  # noqa: E731
    lines = [
        f"# Generalization: guideline {b.version} vs {a.version} on the same receipts",
        "",
        f"Runs: `{b.run_id}` ({b.version}) and `{a.run_id}` ({a.version}). Receipts: "
        f"{b.receipts}; words: {b.words}; CORD-matched words: {b.matched}.",
        "",
        f"| metric | {b.version} | {a.version} | change |",
        "|---|---|---|---|",
        f"| word-level alpha | {b.alpha:.3f} | {a.alpha:.3f} | {a.alpha - b.alpha:+.3f} |",
    ]
    for ag in b.headline:
        lines.append(f"| headline {ag} | {pct(b.headline[ag])} | {pct(a.headline[ag])} | "
                     f"{pts(b.headline[ag], a.headline[ag])} |")
    lines += [
        f"| majority vote matches v2 | {pct(b.majority)} (n={b.majority_n}) | "
        f"{pct(a.majority)} (n={a.majority_n}) | {pts(b.majority, a.majority)} |",
        f"| all-3-agree matches v2 | {pct(b.all_agree)} (n={b.all_agree_n}) | "
        f"{pct(a.all_agree)} (n={a.all_agree_n}) | {pts(b.all_agree, a.all_agree)} |",
        f"| words sent to review | {pct(b.review_share)} | {pct(a.review_share)} | "
        f"{pts(b.review_share, a.review_share)} |",
        "",
        "| check | result |",
        "|---|---|",
    ]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks.items()]
    lines += ["", f"**Overall: {'PASS' if all(checks.values()) else 'FAIL'}**"]
    text = "\n".join(lines) + "\n"
    (out_dir or ROOT / "reports").joinpath(
        f"generalization_{before_id}_{after_id}.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(report(sys.argv[1], sys.argv[2]))
