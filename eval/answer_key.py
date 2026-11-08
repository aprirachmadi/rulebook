"""The v2 answer key. Only `eval/` scripts may use this, and only after a run finishes."""

from __future__ import annotations

from rulebook.data.cord import _raw_ground_truth, receipt_id

LineKey = tuple[str, int]  # (receipt_id, line_idx)


def load_answer_key(split: str) -> dict[LineKey, str]:
    """Map every line of `split` to its v2 category."""
    return {
        (receipt_id(split, gt["meta"]["image_id"]), idx): line["category"]
        for gt in _raw_ground_truth(split)
        for idx, line in enumerate(gt["valid_line"])
    }
