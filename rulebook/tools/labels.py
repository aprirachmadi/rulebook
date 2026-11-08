"""What labelers see and return: one document format for both text sources.

A `LabelDoc` is a receipt as numbered lines of numbered words. Each word carries a `ref` to the
CORD word it corresponds to (gold mode: itself; OCR mode: the matched CORD word or None).
Refs are positions only; labels are never stored here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

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


Scope = dict[int, list[int]]  # line_idx -> word indices to label


def full_scope(doc: LabelDoc, lines: list[int] | None = None) -> Scope:
    wanted = set(lines) if lines is not None else None
    return {ln.line_idx: list(range(len(ln.words))) for ln in doc.lines
            if wanted is None or ln.line_idx in wanted}


def render(doc: LabelDoc, scope: Scope | None = None, note: str = "") -> str:
    """Lines as the labeler sees them: [line_idx] (x, y) then numbered words.

    `scope` limits which lines are shown (a chunk, or a repair); a line in a repair whose
    scope is only some of its words says which ones to label.
    """
    scope = scope if scope is not None else full_scope(doc)
    shown = [ln for ln in doc.lines if ln.line_idx in scope]
    first, last = shown[0].line_idx, shown[-1].line_idx
    out = [
        f"Receipt {doc.receipt_id}, lines {first}-{last} of {len(doc.lines)}. "
        "One line per row: [line_idx] (x, y) then numbered words.",
        "x and y are the line's centre, 0-100 from the left / top of the receipt.",
    ]
    if note:
        out.append(note)
    out.append("")
    for ln in shown:
        x = round(100 * (ln.box[0] + ln.box[2]) / 2 / doc.width)
        y = round(100 * (ln.box[1] + ln.box[3]) / 2 / doc.height)
        words = " ".join(f"{i}:{w.text}" for i, w in enumerate(ln.words))
        row = f"[{ln.line_idx}] ({x}, {y}) {words}"
        if len(scope[ln.line_idx]) < len(ln.words):
            row += f"   <- label only words {scope[ln.line_idx]}"
        out.append(row)
    return "\n".join(out)


# --- answer schema ---


class Span(BaseModel):
    line_idx: int
    words: list[int] = Field(min_length=1)
    label: str
    confidence: float = Field(ge=0, le=1)
    reason: str = ""  # asked for only when confidence < 0.7


class LabelBatch(BaseModel):
    labels: list[Span]


def answer_schema(scope: Scope, label_space: tuple[str, ...]) -> dict[str, Any]:
    """JSON schema with one alternative per line: `line_idx` fixed and that line's own word
    numbers as an enum, so constrained decoding can't use a word number the line lacks."""
    per_line = [
        {
            "type": "object",
            "properties": {
                "line_idx": {"const": li},
                "words": {"type": "array", "items": {"enum": words}, "minItems": 1,
                          "maxItems": len(words)},
                "label": {"enum": list(label_space)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["line_idx", "words", "label", "confidence"],
        }
        for li, words in scope.items()
    ]
    n_words = sum(len(w) for w in scope.values())
    return {
        "title": "LabelBatch",
        "type": "object",
        "properties": {
            "labels": {"type": "array", "items": {"anyOf": per_line}, "minItems": 1,
                       "maxItems": n_words}
        },
        "required": ["labels"],
    }


def batch_model(scope: Scope, label_space: tuple[str, ...]) -> type[LabelBatch]:
    """Pydantic model for one call: label enum enforced; its JSON schema is `answer_schema`.
    Coverage is not validated here; `sort_spans` handles it so good spans are kept."""
    span = create_model("Span", __base__=Span,
                        label=(Literal[label_space], ...))  # type: ignore[valid-type]
    schema = answer_schema(scope, label_space)

    def use_answer_schema(s: dict[str, Any], _cls: Any) -> None:
        s.clear()
        s.update(schema)

    class Batch(LabelBatch):
        model_config = ConfigDict(json_schema_extra=use_answer_schema)
        labels: list[span]  # type: ignore[valid-type]

    return Batch


def sort_spans(
    scope: Scope, spans: list[Span]
) -> tuple[dict[tuple[int, int], Span], Scope, list[str]]:
    """Keep every word labeled once and in scope; return (accepted, still to label, problems).

    Out-of-scope lines/words are dropped. A word given two different labels goes back to
    "still to label"; the same label twice is fine.
    """
    accepted: dict[tuple[int, int], Span] = {}
    conflicts: set[tuple[int, int]] = set()
    problems: list[str] = []
    for s in spans:
        allowed = set(scope.get(s.line_idx, []))
        if not allowed:
            problems.append(f"line {s.line_idx} was not asked for")
            continue
        for w in s.words:
            key = (s.line_idx, w)
            if w not in allowed:
                problems.append(f"line {s.line_idx} has no word {w} to label")
            elif key in conflicts:
                continue
            elif key in accepted and accepted[key].label != s.label:
                conflicts.add(key)
                del accepted[key]
            elif key not in accepted:
                accepted[key] = s
    todo: Scope = {}
    for li, words in scope.items():
        left = [w for w in words if (li, w) not in accepted]
        if left:
            todo[li] = left
    if conflicts:
        problems.append(f"words given two different labels: {sorted(conflicts)[:10]}")
    missing = sum(len(w) for w in todo.values()) - len(conflicts)
    if missing > 0:
        problems.append(f"{missing} words got no label")
    return accepted, todo, problems
