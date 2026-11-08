"""PaddleOCR (CPU, local PP-OCRv6 small models) and OCR-word -> CORD-word matching.

Runs on CPU only, so it never competes with Ollama for VRAM (~1 GB RAM, ~4 s per receipt).
Matching uses boxes and text only; it never reads labels.
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from rulebook.config import ROOT, load_config
from rulebook.data.cord import Box, Receipt, load_image, union_box


class OcrWord(BaseModel):
    text: str
    box: Box


class OcrLine(BaseModel):
    text: str
    box: Box
    conf: float  # PaddleOCR gives confidence per line, not per word
    words: list[OcrWord]


class OcrResult(BaseModel):
    receipt_id: str
    width: int
    height: int
    lines: list[OcrLine]


def _box(b: Any) -> Box:
    x0, y0, x1, y1 = (int(round(float(v))) for v in b)
    return x0, y0, x1, y1


def merge_tokens(tokens: list[str], boxes: list[Any]) -> list[OcrWord]:
    """PaddleOCR splits at separators ('75', ',', '000') and emits spaces as tokens.
    Join everything between spaces into one word whose box is the union of its pieces."""
    words: list[OcrWord] = []
    text, parts = "", []
    for tok, b in zip(tokens, boxes, strict=True):
        if tok.isspace() or tok == "":
            if text:
                words.append(OcrWord(text=text, box=union_box(parts)))
            text, parts = "", []
            continue
        text += tok
        parts.append(_box(b))
    if text:
        words.append(OcrWord(text=text, box=union_box(parts)))
    return words


@cache
def _engine() -> Any:
    cfg = load_config()["ocr"]
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # stay offline
    from paddleocr import PaddleOCR  # slow import; only when OCR actually runs

    model_dir = ROOT / cfg["model_dir"]
    return PaddleOCR(
        text_detection_model_name=cfg["det_model"],
        text_detection_model_dir=str(model_dir / cfg["det_model"]),
        text_recognition_model_name=cfg["rec_model"],
        text_recognition_model_dir=str(model_dir / cfg["rec_model"]),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=cfg["cpu_threads"],
        return_word_box=True,
    )


def run_ocr(receipt: Receipt) -> OcrResult:
    img = np.array(load_image(receipt.split, receipt.image_id).convert("RGB"))
    res = _engine().predict(img)[0].json["res"]
    lines = []
    for text, score, box, tokens, tboxes in zip(
        res["rec_texts"], res["rec_scores"], res["rec_boxes"], res["text_word"],
        res["text_word_boxes"], strict=True,
    ):
        words = merge_tokens(list(tokens), list(tboxes))
        if words:
            lines.append(OcrLine(text=text, box=_box(box), conf=float(score), words=words))
    return OcrResult(
        receipt_id=receipt.receipt_id, width=receipt.width, height=receipt.height, lines=lines
    )


def ocr_cached(receipt: Receipt) -> OcrResult:
    """OCR is deterministic, so results are cached per receipt under runs/ocr_cache/."""
    cache_dir = ROOT / load_config()["harness"]["runs_dir"] / "ocr_cache"
    path = cache_dir / f"{receipt.receipt_id}.json"
    if path.exists():
        return OcrResult.model_validate_json(path.read_text(encoding="utf-8"))
    result = run_ocr(receipt)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    return result


# --- matching ---

WordRef = tuple[int, int]  # (CORD line_idx, word position in reading order)


def _center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(p: tuple[float, float], b: Box) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_word(word: OcrWord, receipt: Receipt) -> WordRef | None:
    """The CORD word whose box contains this word's centre; ties broken by text similarity."""
    c = _center(word.box)
    candidates = [
        (similarity(word.text, w.text), line.line_idx, i)
        for line in receipt.lines
        for i, w in enumerate(line.words)
        if _inside(c, w.box)
    ]
    if not candidates:
        return None
    _, line_idx, i = max(candidates, key=lambda t: (t[0], -t[1], -t[2]))
    return line_idx, i


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
