"""What labelers see and return: one document format for both text sources.

A `LabelDoc` is a receipt as numbered lines of numbered words. Each word carries a `ref` to the
CORD word it corresponds to (gold mode: itself; OCR mode: the matched CORD word or None).
Refs are positions only; labels are never stored here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, conint, create_model, field_validator

from rulebook.data.cord import Box, Receipt
from rulebook.tools.ocr import OcrResult, WordRef, match_word

TextSource = Literal["gold", "ocr"]


class DocWord(BaseModel):
    text: str
    box: Box
    ref: WordRef | None  # CORD (line_idx, word position); None = OCR miss


class DocLine(BaseModel):
    line_idx: int
    text: str
    box: Box
    conf: float | None = None  # OCR confidence; None in gold mode
    words: list[DocWord]


class LabelDoc(BaseModel):
    receipt_id: str
    text_source: TextSource
    width: int
    height: int
    lines: list[DocLine]

    def word_keys(self) -> list[tuple[int, int]]:
        return [(ln.line_idx, i) for ln in self.lines for i in range(len(ln.words))]


def _top_left(box: Box) -> tuple[int, int]:
    return box[1], box[0]


def gold_doc(receipt: Receipt) -> LabelDoc:
    """CORD's own lines and words. Lines are sorted top-to-bottom, left-to-right, because
    CORD's `valid_line` order follows its annotation groups and could hint at labels."""
    ordered = sorted(receipt.lines, key=lambda ln: _top_left(ln.box))
    lines = [
        DocLine(
            line_idx=i,
            text=ln.text,
            box=ln.box,
            words=[DocWord(text=w.text, box=w.box, ref=(ln.line_idx, j))
                   for j, w in enumerate(ln.words)],
        )
        for i, ln in enumerate(ordered)
    ]
    return LabelDoc(receipt_id=receipt.receipt_id, text_source="gold", width=receipt.width,
                    height=receipt.height, lines=lines)


def ocr_doc(receipt: Receipt, ocr: OcrResult) -> LabelDoc:
    ordered = sorted(ocr.lines, key=lambda ln: _top_left(ln.box))
    lines = [
        DocLine(
            line_idx=i,
            text=ln.text,
            box=ln.box,
            conf=ln.conf,
            words=[DocWord(text=w.text, box=w.box, ref=match_word(w, receipt)) for w in ln.words],
        )
        for i, ln in enumerate(ordered)
    ]
    return LabelDoc(receipt_id=receipt.receipt_id, text_source="ocr", width=ocr.width,
                    height=ocr.height, lines=lines)


def render(doc: LabelDoc) -> str:
    """The receipt as the labeler sees it: line index, rough position (0-100), numbered words."""
    out = [
        f"Receipt {doc.receipt_id}. One line per row: [line_idx] (x, y) then numbered words.",
        "x and y are the line's centre, 0-100 from the left / top of the receipt.",
        "",
    ]
    for ln in doc.lines:
        x = round(100 * (ln.box[0] + ln.box[2]) / 2 / doc.width)
        y = round(100 * (ln.box[1] + ln.box[3]) / 2 / doc.height)
        words = " ".join(f"{i}:{w.text}" for i, w in enumerate(ln.words))
        out.append(f"[{ln.line_idx}] ({x}, {y}) {words}")
    return "\n".join(out)


# --- answer schema ---


class Span(BaseModel):
    line_idx: int
    words: list[int]
    label: str
    confidence: float = Field(ge=0, le=1)
    reason: str


class LabelBatch(BaseModel):
    labels: list[Span]


def coverage_errors(doc: LabelDoc, spans: list[Span]) -> list[str]:
    """Every word of every line must get exactly one label."""
    sizes = {ln.line_idx: len(ln.words) for ln in doc.lines}
    seen: dict[tuple[int, int], int] = {}
    errors = []
    for s in spans:
        if s.line_idx not in sizes:
            errors.append(f"line_idx {s.line_idx} does not exist")
            continue
        for w in s.words:
            if not 0 <= w < sizes[s.line_idx]:
                errors.append(f"line {s.line_idx} has no word {w}")
            else:
                seen[(s.line_idx, w)] = seen.get((s.line_idx, w), 0) + 1
    twice = sorted(k for k, n in seen.items() if n > 1)
    missing = [k for k in doc.word_keys() if k not in seen]
    if twice:
        errors.append(f"words labeled more than once (line, word): {twice[:15]}")
    if missing:
        errors.append(f"words with no label (line, word): {missing[:15]}"
                      + (f" and {len(missing) - 15} more" if len(missing) > 15 else ""))
    return errors


def batch_model(doc: LabelDoc, label_space: tuple[str, ...]) -> type[LabelBatch]:
    """A LabelBatch schema for this receipt: labels restricted to the label space (as an enum,
    so constrained decoding enforces it) and a validator for full, non-overlapping coverage."""
    n_lines = len(doc.lines)
    max_words = max((len(ln.words) for ln in doc.lines), default=1)
    n_words = len(doc.word_keys())
    # Bounds go into the JSON schema, so constrained decoding cannot invent lines or words.
    span = create_model(
        "Span",
        __base__=Span,
        line_idx=(conint(ge=0, le=max(n_lines - 1, 0)), ...),
        words=(list[conint(ge=0, le=max_words - 1)], Field(min_length=1)),  # type: ignore[valid-type]
        label=(Literal[label_space], ...),  # type: ignore[valid-type]
    )

    def _cover(cls: Any, v: list[Span], info: ValidationInfo) -> list[Span]:
        errs = coverage_errors(doc, v)
        if errs:
            raise ValueError("; ".join(errs))
        return v

    return create_model(
        "LabelBatch",
        __base__=LabelBatch,
        labels=(list[span], Field(min_length=1, max_length=max(n_words, 1))),  # type: ignore[valid-type]
        __validators__={"cover": field_validator("labels")(_cover)},
    )


def word_labels(doc: LabelDoc, batch: LabelBatch) -> dict[tuple[int, int], Span]:
    """(line_idx, word_idx) -> the span that labels it."""
    return {(s.line_idx, w): s for s in batch.labels for w in s.words}
