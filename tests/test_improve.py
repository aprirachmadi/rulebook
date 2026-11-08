"""M5: disagreement spans, clustering, hold-out, rule-writer view, verifier helpers, leakage."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rulebook.tools.agent_tools import RunContext, build_registry
from rulebook.tools.clusters import (
    Item,
    agglomerative,
    cluster_items,
    disagreement_spans,
    make_clusters,
    normalize,
)
from rulebook.tools.labels import DocLine, DocWord, LabelDoc
from rulebook.tools.verify import words_alpha

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ["labeler_a", "labeler_b", "labeler_c"]
CFG = {"max_group_size": 50, "tiny_group_size": 3, "tiny_share_resplit": 0.5,
       "distance_threshold": 0.25, "min_cluster_for_rule": 3, "max_clusters_per_run": 8,
       "holdout_fraction": 0.3, "min_size_for_holdout": 6, "seed": 0}


def _doc(rid: str, lines: list[list[str]]) -> LabelDoc:
    return LabelDoc(receipt_id=rid, text_source="ocr", width=100, height=100, lines=[
        DocLine(line_idx=i, text=" ".join(ws), box=(0, i * 10, 50, i * 10 + 8), conf=0.9,
                words=[DocWord(text=w, box=(j * 10, i * 10, j * 10 + 8, i * 10 + 8), ref=None)
                       for j, w in enumerate(ws)])
        for i, ws in enumerate(lines)])


def _v(*labels, conf=0.9):
    return {a: (lab, conf, "") for a, lab in zip(AGENTS, labels, strict=True)}


def test_normalize() -> None:
    assert normalize("DISC  10%") == "disc ##%"
    assert normalize("Total Item: 2") == "total item: #"


def test_spans_accept_and_group_consecutive_words() -> None:
    doc = _doc("r1", [["Total", "Item:", "2"], ["Nasi", "25.000"]])
    votes = {
        ("r1", 0, 0): _v("t.qty", "t.type", "t.qty"),
        ("r1", 0, 1): _v("t.qty", "t.type", "t.qty"),
        ("r1", 0, 2): _v("t.qty", "m.cnt", "t.qty"),  # different pattern -> new span
        ("r1", 1, 0): _v("m.nm", "m.nm", "m.nm"),  # accepted
        ("r1", 1, 1): _v("m.price", "m.price", "m.price", conf=0.5),  # low conf -> disagreement
    }
    accepted, items = disagreement_spans([doc], votes, AGENTS, 0.7)
    assert accepted == 1
    assert [(it.line_idx, it.words, it.text) for it in items] == [
        (0, [0, 1], "Total Item:"), (0, [2], "2"), (1, [1], "25.000")]
    assert items[0].labels == ("t.qty", "t.type") and items[0].pattern == "total item:"


def test_agglomerative_merges_close_vectors_only() -> None:
    x = np.array([[1, 0], [0.99, 0.05], [0, 1], [0.02, 1]], dtype=float)
    labels = agglomerative(x, 0.25)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


def _items(n: int, text: str, labels=("a", "b")) -> list[Item]:
    return [Item(f"r{i}", 0, [0], text, text, {}, labels, normalize(text)) for i in range(n)]


def fake_embed(texts: list[str]) -> np.ndarray:
    return np.array([[1.0, 0.0] if t.startswith("x") else [0.0, 1.0] for t in texts])


def test_cluster_items_resplits_tiny_groups_by_embedding() -> None:
    items = _items(1, "x1") + _items(1, "x2 pcs") + _items(1, "Total")  # all tiny
    groups, stats = cluster_items(items, fake_embed, CFG)
    assert stats["resplit_tiny"]
    assert sorted(len(g) for g in groups) == [1, 2]  # x1 + x2 pcs merged


def test_make_clusters_caps_and_holds_out() -> None:
    groups = [_items(10, "a"), _items(4, "b"), _items(2, "c")]
    groups += [_items(3, f"g{i}") for i in range(10)]
    clusters, rest = make_clusters(groups, CFG)
    assert len(clusters) == 8  # cap
    assert clusters[0].held_out and len(clusters[0].held) == 3 and len(clusters[0].visible) == 7
    assert not set(i.item_id for i in clusters[0].held) & set(i.item_id
                                                              for i in clusters[0].visible)
    assert not clusters[1].held_out and clusters[1].held == clusters[1].visible  # size 4 < 6
    assert any(len(g) == 2 for g in rest) and len(rest) == 5  # small one + 4 beyond the cap


def _ctx(view: str) -> RunContext:
    doc = _doc("r1", [["x1"], ["x2"], ["x3"], ["x4"], ["x5"], ["x6"], ["TOTAL", "45.500"]])
    votes = {("r1", i, 0): _v("menu.cnt", "menu.nm", "menu.cnt") for i in range(6)}
    votes[("r1", 6, 0)] = _v("total.total_price", "total.total_price", "total.total_price")
    votes[("r1", 6, 1)] = _v("total.total_price", "total.total_price", "total.total_price")
    _, items = disagreement_spans([doc], votes, AGENTS, 0.7)
    groups, _ = cluster_items(items, fake_embed, CFG)
    clusters, _ = make_clusters(groups, CFG)
    return RunContext("run", "ocr", {"r1": doc}, votes, AGENTS,
                      {c.cluster_id: c for c in clusters}, fake_embed, "v1", view=view)


def test_rule_writer_view_hides_held_out_items() -> None:
    ctx = _ctx("visible")
    c = ctx.clusters[0]
    assert c.held_out and len(c.held) == 2
    shown = json.loads(build_registry(ctx).run("get_cluster", {"cluster_id": 0}, 2000).content)
    ids = {it["id"] for it in shown["items"]}
    assert ids == {it.item_id for it in c.visible}
    assert not ids & {it.item_id for it in c.held}
    everything = json.loads(build_registry(_ctx("all")).run("get_cluster", {"cluster_id": 0},
                                                             2000).content)
    assert everything["items_total"] == 6


def test_tool_outputs_fit_the_result_limit() -> None:
    ctx = _ctx("all")
    reg = build_registry(ctx)
    for name, args in [("get_cluster", {"cluster_id": 0}), ("get_votes", {"receipt_id": "r1",
                       "line_idx": 6}), ("get_ocr_conf", {"receipt_id": "r1", "line_idx": 0}),
                       ("sum_check", {"receipt_id": "r1"}), ("get_guideline", {})]:
        res = reg.run(name, args, 2000)
        assert res.ok, (name, res.content)
        assert "[truncated" not in res.content or name == "get_guideline", name
    votes = json.loads(reg.run("get_votes", {"receipt_id": "r1", "line_idx": 0}, 2000).content)
    assert votes[0]["votes"] == "a=menu.cnt 0.9 | b=menu.nm 0.9 | c=menu.cnt 0.9"


def test_draft_examples_must_be_visible() -> None:
    from rulebook.pipeline.improve import draft_model

    model = draft_model({"a", "b", "c"}, ["1", "2"])
    ok = model.model_validate({"section": "1.", "rule_text": "r", "rationale": "x",
                               "example_item_ids": ["a", "b", "c"]})
    assert ok.section == "1"
    with pytest.raises(ValidationError):
        model.model_validate({"section": "1", "rule_text": "r", "rationale": "x",
                              "example_item_ids": ["a", "b", "held"]})
    with pytest.raises(ValidationError):
        model.model_validate({"section": "9", "rule_text": "r", "rationale": "x",
                              "example_item_ids": ["a", "b", "c"]})


def test_words_alpha_unanimous_is_perfect() -> None:
    words = [("r", 0, 0), ("r", 0, 1)]
    same = {a: {w: "x" for w in words} for a in AGENTS}
    assert words_alpha(words, same) == 1.0
    split = {"labeler_a": {words[0]: "x", words[1]: "y"},
             "labeler_b": {words[0]: "y", words[1]: "x"},
             "labeler_c": {words[0]: "x", words[1]: "x"}}
    assert words_alpha(words, split) < 0.5


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


@pytest.mark.parametrize("module", ["rulebook/tools/agent_tools.py", "rulebook/tools/clusters.py",
                                    "rulebook/tools/labels.py", "rulebook/pipeline/labeling.py"])
def test_agent_facing_modules_never_reach_gold_or_v2(module: str) -> None:
    banned = {"rulebook.data.gold", "rulebook.tools.verify", "eval.answer_key", "eval"}
    assert not _imports(ROOT / module) & banned, module
