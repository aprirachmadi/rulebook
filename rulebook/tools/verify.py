"""Verifier helpers: gold-set scoring (the only route to v2 outside eval/) and cluster alpha.

Gold labels come only through `rulebook.data.gold.gold_labels()` (30 dev receipts).
Accuracy uses the headline metric: agreement with v2 over CORD-matched words, `other` = wrong.
"""

from __future__ import annotations

from rulebook.config import load_config
from rulebook.data.cord import load_split
from rulebook.data.gold import GOLD_SPLIT, gold_labels, gold_receipt_ids
from rulebook.tools.labels import LabelDoc, Span, TextSource, gold_doc, ocr_doc
from rulebook.tools.ocr import ocr_cached
from rulebook.tools.stats import alpha

WordLabels = dict[tuple[str, int, int], str]  # (receipt_id, line_idx, word_idx) -> label


def gold_docs(text_source: TextSource) -> list[LabelDoc]:
    ids = set(gold_receipt_ids())
    receipts = [r for r in load_split(GOLD_SPLIT) if r.receipt_id in ids]
    if text_source == "gold":
        return [gold_doc(r) for r in receipts]
    return [ocr_doc(r, ocr_cached(r)) for r in receipts]


def as_word_labels(doc: LabelDoc, labels: dict[tuple[int, int], Span]) -> WordLabels:
    return {(doc.receipt_id, li, wi): s.label for (li, wi), s in labels.items()}


def gold_accuracy(docs: list[LabelDoc], labels: WordLabels) -> tuple[float, int]:
    """Headline accuracy of one labeler on the gold set: (accuracy, scored words)."""
    key = gold_labels()
    hit = n = 0
    for d in docs:
        for ln in d.lines:
            for wi, w in enumerate(ln.words):
                if w.ref is None:
                    continue  # OCR miss: not scored
                n += 1
                hit += labels.get((d.receipt_id, ln.line_idx, wi)) == key[(d.receipt_id, w.ref[0])]
    return (hit / n if n else 0.0), n


def gold_label_set() -> set[str]:
    """Which v2 labels occur in the gold set (for "not gold-verified" patches)."""
    return set(gold_labels().values())


def words_alpha(words: list[tuple[str, int, int]], by_agent: dict[str, WordLabels]) -> float:
    """Alpha over these words. If every word is unanimous, agreement is perfect: 1.0."""
    agents = list(by_agent)
    units = [{a: by_agent[a][k] for a in agents if k in by_agent[a]} for k in words]
    complete = [u for u in units if len(u) == len(agents)]
    if complete and all(len(set(u.values())) == 1 for u in complete):
        return 1.0
    a = alpha(units, agents)
    return a if a is not None else 0.0


def thresholds() -> dict:
    return load_config()["verifier"]
