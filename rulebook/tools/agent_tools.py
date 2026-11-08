"""Tools for the diagnoser and rule writer, bound to one run.

They read labeler votes, OCR text/confidence and the guideline. They never read v2 labels.
The rule writer's view of a cluster is its visible part only (the hold-out stays hidden).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from rulebook.data.store import Vote
from rulebook.harness.tools import ToolRegistry, tool
from rulebook.tools.clusters import Cluster, Embed, Item
from rulebook.tools.guideline import guideline_text
from rulebook.tools.labels import LabelDoc
from rulebook.tools.receipt import sum_check as receipt_sum_check

MAX_ITEMS_SHOWN = 10  # tool results are cut at 2,000 chars (SPEC); keep items compact


@dataclass
class RunContext:
    run_id: str
    text_source: str
    docs: dict[str, LabelDoc]
    votes: dict[tuple[str, int, int], dict[str, Vote]]
    agents: list[str]
    clusters: dict[int, Cluster]
    embed: Embed
    guideline_version: str
    view: Literal["all", "visible"] = "all"  # "visible" for the rule writer
    _line_vecs: np.ndarray | None = field(default=None, repr=False)
    _line_keys: list[tuple[str, int]] = field(default_factory=list, repr=False)

    def majority(self, rid: str, li: int, wi: int) -> str:
        v = self.votes.get((rid, li, wi), {})
        top = Counter(lab for lab, _, _ in v.values()).most_common(1)
        return top[0][0] if top else "-"


def votes_str(v: dict[str, Vote]) -> str:
    """'a=menu.nm 0.9 | b=menu.etc 0.6 (no price)' with labeler_a -> a."""
    return " | ".join(f"{a.rsplit('_', 1)[-1]}={lab} {conf:g}" + (f" ({r})" if r else "")
                      for a, (lab, conf, r) in sorted(v.items()))


def _item_view(ctx: RunContext, it: Item) -> dict:
    return {"id": it.item_id, "text": it.text, "line": it.line_text,
            "votes": votes_str(it.votes)}


class ClusterArgs(BaseModel):
    cluster_id: int


class LineArgs(BaseModel):
    receipt_id: str
    line_idx: int


class ReceiptArgs(BaseModel):
    receipt_id: str


class SearchArgs(BaseModel):
    text: str
    k: int = Field(default=5, ge=1, le=10)


class NoArgs(BaseModel):
    pass


def build_registry(ctx: RunContext) -> ToolRegistry:
    reg = ToolRegistry()

    @tool("get_cluster", "Show a cluster of disagreements: shared labels, text pattern and its "
          "items with each labeler's vote.", registry=reg)
    def get_cluster(a: ClusterArgs) -> str:
        c = ctx.clusters.get(a.cluster_id)
        if c is None:
            return f"error: no cluster {a.cluster_id}; have {sorted(ctx.clusters)}"
        items = c.visible if ctx.view == "visible" else c.items
        return json.dumps({
            "cluster_id": c.cluster_id,
            "labels_voted": list(c.labels),
            "text_pattern": c.pattern,
            "items_total": len(items),
            "items_shown": min(len(items), MAX_ITEMS_SHOWN),
            "items": [_item_view(ctx, it) for it in items[:MAX_ITEMS_SHOWN]],
        }, ensure_ascii=False)

    @tool("get_votes", "Every word of one receipt line with each labeler's label, confidence "
          "and reason.", registry=reg)
    def get_votes(a: LineArgs) -> str:
        d = ctx.docs.get(a.receipt_id)
        if d is None or not any(ln.line_idx == a.line_idx for ln in d.lines):
            return "error: no such receipt line"
        ln = next(ln for ln in d.lines if ln.line_idx == a.line_idx)
        return json.dumps([
            {"word": i, "text": w.text,
             "votes": votes_str(ctx.votes.get((a.receipt_id, a.line_idx, i), {}))}
            for i, w in enumerate(ln.words)
        ], ensure_ascii=False)

    @tool("get_ocr_conf", "The OCR text of a line and PaddleOCR's confidence (0-1). In gold-text "
          "mode there is no OCR confidence.", registry=reg)
    def get_ocr_conf(a: LineArgs) -> str:
        d = ctx.docs.get(a.receipt_id)
        ln = next((ln for ln in d.lines if ln.line_idx == a.line_idx), None) if d else None
        if ln is None:
            return "error: no such receipt line"
        return json.dumps({"text": ln.text, "text_source": ctx.text_source,
                           "ocr_confidence": ln.conf}, ensure_ascii=False)

    @tool("sum_check", "Check the receipt's arithmetic (items - discounts + tax/service = "
          "subtotal/total, within 1%) using the labelers' majority labels. Numbers are parsed "
          "in code.", registry=reg)
    def sum_check(a: ReceiptArgs) -> str:
        d = ctx.docs.get(a.receipt_id)
        if d is None:
            return "error: no such receipt"
        spans: list[tuple[str, str]] = []  # consecutive words with the same majority label
        for ln in d.lines:
            prev = None
            for wi, w in enumerate(ln.words):
                lab = ctx.majority(d.receipt_id, ln.line_idx, wi)
                if lab == prev:
                    spans[-1] = (spans[-1][0] + " " + w.text, lab)
                else:
                    spans.append((w.text, lab))
                prev = lab
        res = receipt_sum_check(spans)
        return json.dumps({"passed": res.passed, "checks": res.checks,
                           "suspicious": [spans[i][0] for i in res.suspicious]},
                          ensure_ascii=False)

    @tool("search_similar_lines", "Find receipt lines in this batch similar to a text, with "
          "the labelers' majority label per word.", registry=reg)
    def search_similar_lines(a: SearchArgs) -> str:
        if ctx._line_vecs is None:
            ctx._line_keys = [(rid, ln.line_idx) for rid, d in ctx.docs.items() for ln in d.lines]
            texts = [next(ln.text for ln in ctx.docs[rid].lines if ln.line_idx == li)
                     for rid, li in ctx._line_keys]
            vecs = np.asarray(ctx.embed(texts), dtype=float)
            ctx._line_vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        q = np.asarray(ctx.embed([a.text])[0], dtype=float)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        best = np.argsort(-(ctx._line_vecs @ q))[: a.k]
        out = []
        for i in best:
            rid, li = ctx._line_keys[int(i)]
            ln = next(x for x in ctx.docs[rid].lines if x.line_idx == li)
            out.append({"receipt_id": rid, "line_idx": li, "text": ln.text,
                        "majority_labels": [ctx.majority(rid, li, wi)
                                            for wi in range(len(ln.words))]})
        return json.dumps(out, ensure_ascii=False)

    @tool("get_guideline", "The current labeling guideline text.", registry=reg)
    def get_guideline(_: NoArgs) -> str:
        return guideline_text(ctx.guideline_version)

    return reg
