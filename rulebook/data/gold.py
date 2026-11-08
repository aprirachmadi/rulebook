"""The gold set: the only sanctioned route to v2 labels outside `eval/`.

`gold_labels()` is for the Verifier. It returns labels only for the configured gold receipts,
all from the dev split, and refuses anything else. The leakage test enforces this.
"""

from __future__ import annotations

import random

from rulebook.config import load_config
from rulebook.data.cord import _raw_ground_truth, receipt_id

GOLD_SPLIT = "validation"

LineKey = tuple[str, int]  # (receipt_id, line_idx)


class LeakageError(RuntimeError):
    pass


def select_gold_receipts(n: int, seed: int) -> list[str]:
    """Pick `n` dev receipts so every dev category appears at least once where possible.

    Greedy: walk categories from rarest (fewest dev receipts) to most common; if a category
    is not covered yet, add a random receipt that has it. Then fill up with random receipts.
    Returns receipt ids only.
    """
    rng = random.Random(seed)
    cats_by_receipt = {
        receipt_id(GOLD_SPLIT, gt["meta"]["image_id"]): {ln["category"] for ln in gt["valid_line"]}
        for gt in _raw_ground_truth(GOLD_SPLIT)
    }
    all_cats = sorted(set().union(*cats_by_receipt.values()))
    rarity = {c: sum(c in cs for cs in cats_by_receipt.values()) for c in all_cats}

    chosen: list[str] = []
    covered: set[str] = set()
    for cat in sorted(all_cats, key=lambda c: (rarity[c], c)):
        if cat in covered or len(chosen) == n:
            continue
        candidates = sorted(r for r, cs in cats_by_receipt.items() if cat in cs and r not in chosen)
        pick = rng.choice(candidates)
        chosen.append(pick)
        covered |= cats_by_receipt[pick]
    rest = sorted(r for r in cats_by_receipt if r not in chosen)
    chosen += rng.sample(rest, n - len(chosen))
    return sorted(chosen)


def gold_receipt_ids() -> list[str]:
    cfg = load_config()["data"]
    configured = cfg["splits"]["gold"]
    if configured != GOLD_SPLIT:
        raise LeakageError(f"gold split must be {GOLD_SPLIT!r}, config says {configured!r}")
    ids: list[str] = cfg["gold_receipt_ids"]
    bad = [r for r in ids if not r.startswith(f"{GOLD_SPLIT}_")]
    if bad:
        raise LeakageError(f"gold receipts outside the {GOLD_SPLIT} split: {bad}")
    return ids


def gold_labels() -> dict[LineKey, str]:
    """v2 labels for the gold receipts only. The Verifier's sanctioned accessor."""
    ids = set(gold_receipt_ids())
    labels = {}
    for gt in _raw_ground_truth(GOLD_SPLIT):
        rid = receipt_id(GOLD_SPLIT, gt["meta"]["image_id"])
        if rid in ids:
            for idx, line in enumerate(gt["valid_line"]):
                labels[(rid, idx)] = line["category"]
    return labels
