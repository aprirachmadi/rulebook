"""CORD v2 data layer.

Agent-facing objects (`Receipt`, `Line`, `Word`) carry no labels. The raw ground truth,
which holds the v2 `category` of every line, is reachable only through `_raw_ground_truth`.
Only the answer key (`eval/`), `label_space()` and the gold-set tool may call it; the
leakage test enforces this.
"""

from __future__ import annotations

import io
import json
from functools import cache
from typing import Any

from datasets import Dataset, Image, load_dataset
from PIL import Image as PILImage
from pydantic import BaseModel

from rulebook.config import load_config

Box = tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


class Word(BaseModel):
    text: str
    box: Box
    row_id: int
    is_key: bool


class Line(BaseModel):
    receipt_id: str
    line_idx: int  # index into the receipt's `valid_line`; stable key shared with the answer key
    text: str
    box: Box
    words: list[Word]


class Receipt(BaseModel):
    receipt_id: str
    split: str
    image_id: int
    width: int
    height: int
    lines: list[Line]


def receipt_id(split: str, image_id: int) -> str:
    return f"{split}_{image_id:04d}"


def quad_box(quad: dict[str, int]) -> Box:
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return min(xs), min(ys), max(xs), max(ys)


def union_box(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def reading_order(words: list[Word]) -> list[Word]:
    """Rows top to bottom (by mean top edge), words left to right within a row.

    CORD's word lists are not always in reading order, so they are re-sorted here.
    """
    rows: dict[int, list[Word]] = {}
    for w in words:
        rows.setdefault(w.row_id, []).append(w)
    ordered_rows = sorted(rows.values(), key=lambda r: sum(w.box[1] for w in r) / len(r))
    return [w for row in ordered_rows for w in sorted(row, key=lambda w: w.box[0])]


def parse_receipt(split: str, gt: dict[str, Any]) -> Receipt:
    """Build a label-free `Receipt` from one raw ground-truth dict."""
    meta = gt["meta"]
    rid = receipt_id(split, meta["image_id"])
    lines = []
    for idx, raw in enumerate(gt["valid_line"]):
        words = reading_order(
            [
                Word(
                    text=w["text"],
                    box=quad_box(w["quad"]),
                    row_id=w["row_id"],
                    is_key=bool(w["is_key"]),
                )
                for w in raw["words"]
            ]
        )
        lines.append(
            Line(
                receipt_id=rid,
                line_idx=idx,
                text=" ".join(w.text for w in words),
                box=union_box([w.box for w in words]),
                words=words,
            )
        )
    return Receipt(
        receipt_id=rid,
        split=split,
        image_id=meta["image_id"],
        width=meta["image_size"]["width"],
        height=meta["image_size"]["height"],
        lines=lines,
    )


@cache
def _dataset(split: str) -> Dataset:
    name = load_config()["data"]["dataset"]
    return load_dataset(name, split=split).cast_column("image", Image(decode=False))


def _raw_ground_truth(split: str) -> list[dict[str, Any]]:
    """Raw v2 ground truth, labels included. Restricted: see module docstring."""
    return [json.loads(gt) for gt in _dataset(split)["ground_truth"]]


@cache
def load_split(split: str) -> list[Receipt]:
    return [parse_receipt(split, gt) for gt in _raw_ground_truth(split)]


def load_image(split: str, image_id: int) -> PILImage.Image:
    row = _dataset(split)[image_id]
    return PILImage.open(io.BytesIO(row["image"]["bytes"]))


@cache
def label_space() -> tuple[str, ...]:
    """Category names seen in the work split. Names only, never per-line labels."""
    work = load_config()["data"]["splits"]["work"]
    return tuple(
        sorted({line["category"] for gt in _raw_ground_truth(work) for line in gt["valid_line"]})
    )
