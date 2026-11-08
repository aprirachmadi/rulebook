"""M4: OCR word merging, matching, label docs, span validation, alpha, store."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rulebook.data.cord import load_split
from rulebook.data.store import connect, load_docs, load_votes, save_run, save_votes
from rulebook.tools.labels import (
    LabelBatch,
    Span,
    batch_model,
    coverage_errors,
    gold_doc,
    render,
)
from rulebook.tools.ocr import OcrWord, match_word, merge_tokens
from rulebook.tools.stats import agreement_pattern, alpha, alpha_per_label

LABELS = ("menu.nm", "menu.price", "total.total_price")


def test_merge_tokens_joins_separator_pieces() -> None:
    # real PaddleOCR output for train_0000: '75', ',', '000' and space tokens
    words = merge_tokens(
        ["Nasi", " ", "Campur", " ", "75", ",", "000"],
        [[309, 366, 351, 386], [355, 366, 359, 386], [372, 366, 435, 386],
         [436, 366, 440, 386], [549, 360, 567, 384], [571, 360, 576, 384], [585, 360, 612, 384]],
    )
    assert [w.text for w in words] == ["Nasi", "Campur", "75,000"]
    assert words[2].box == (549, 360, 612, 384)


def test_match_word_uses_centre_then_text() -> None:
    receipt = load_split("validation")[0]
    line = receipt.lines[0]
    w = line.words[0]
    assert match_word(OcrWord(text=w.text, box=w.box), receipt) == (line.line_idx, 0)
    assert match_word(OcrWord(text="x", box=(-50, -50, -40, -40)), receipt) is None


@pytest.fixture
def doc():
    return gold_doc(load_split("validation")[0])


def test_gold_doc_is_label_free_and_top_to_bottom(doc) -> None:
    assert "category" not in doc.model_dump_json()
    keys = [(ln.box[1], ln.box[0]) for ln in doc.lines]
    assert keys == sorted(keys)  # top-to-bottom, then left-to-right
    assert all(w.ref is not None for ln in doc.lines for w in ln.words)


def test_render_numbers_lines_and_words(doc) -> None:
    text = render(doc)
    first = doc.lines[0]
    assert "[0] (" in text and f"0:{first.words[0].text}" in text


def _full_cover(doc, label="menu.nm"):
    return [Span(line_idx=ln.line_idx, words=list(range(len(ln.words))), label=label,
                 confidence=0.9, reason="r") for ln in doc.lines]


def test_coverage_accepts_full_and_rejects_gaps_and_overlaps(doc) -> None:
    spans = _full_cover(doc)
    assert coverage_errors(doc, spans) == []
    assert any("no label" in e for e in coverage_errors(doc, spans[1:]))
    assert any("more than once" in e for e in coverage_errors(doc, spans + spans[:1]))
    bad = [*spans, Span(line_idx=999, words=[0], label="menu.nm", confidence=1, reason="")]
    assert any("does not exist" in e for e in coverage_errors(doc, bad))


def test_batch_model_enforces_label_enum_and_coverage(doc) -> None:
    model = batch_model(doc, LABELS)
    ok = {"labels": [s.model_dump() for s in _full_cover(doc)]}
    assert isinstance(model.model_validate(ok), LabelBatch)
    with pytest.raises(ValidationError):
        model.model_validate({"labels": [s.model_dump() for s in _full_cover(doc, "nope")]})
    with pytest.raises(ValidationError):
        model.model_validate({"labels": ok["labels"][1:]})
    enum = model.model_json_schema()["$defs"]["Span"]["properties"]["label"]["enum"]
    assert enum == list(LABELS)


def test_alpha_basics() -> None:
    agents = ["a", "b", "c"]
    same = [{"a": "x", "b": "x", "c": "x"}, {"a": "y", "b": "y", "c": "y"}] * 5
    assert alpha(same, agents) == pytest.approx(1.0)
    mixed = [*same, {"a": "x", "b": "y", "c": "x"}]
    assert alpha(mixed, agents) < 1.0
    assert set(alpha_per_label(mixed, agents)) == {"x", "y"}
    assert agreement_pattern(mixed, agents) == {"all_agree": 10, "two_agree": 1,
                                                "all_differ": 0, "incomplete": 0}
    assert alpha([{"a": "x"}], agents) is None  # one label only


def test_store_roundtrip(tmp_path: Path, doc) -> None:
    db = connect(tmp_path / "t.db")
    save_run(db, "r1", "gold", "v1", [doc])
    batch = batch_model(doc, LABELS).model_validate(
        {"labels": [s.model_dump() for s in _full_cover(doc)]})
    save_votes(db, "r1", doc, "labeler_a", "m", "v1", batch)
    assert load_docs(db, "r1")[0] == doc
    votes = load_votes(db, "r1")
    assert len(votes) == sum(len(ln.words) for ln in doc.lines)
    assert all(v == {"labeler_a": "menu.nm"} for v in votes.values())


def test_alpha_is_none_with_a_single_labeler() -> None:
    assert alpha([{"a": "x"}, {"a": "y"}], ["a", "b", "c"]) is None


def test_batch_schema_bounds_lines_words_and_spans(doc) -> None:
    schema = batch_model(doc, LABELS).model_json_schema()
    span = schema["$defs"]["Span"]["properties"]
    assert span["line_idx"]["maximum"] == len(doc.lines) - 1
    assert span["words"]["items"]["maximum"] == max(len(ln.words) for ln in doc.lines) - 1
    assert schema["properties"]["labels"]["maxItems"] == len(doc.word_keys())
    assert schema["properties"]["labels"]["minItems"] == 1


def test_structured_repair_starts_fresh(tmp_path) -> None:
    from rulebook.harness.agents import Agent
    from rulebook.harness.loop import run
    from rulebook.harness.tools import ToolRegistry
    from rulebook.harness.trace import Tracer
    from tests.test_harness import FakeServer, _reply

    class Out(LabelBatch):
        pass

    long_bad = '{"labels": "' + "x" * 5000 + '"}'
    server = FakeServer(_reply(content=long_bad), _reply(content='{"labels": []}'))
    run(Agent("t", "sys", "fake", output_schema=Out), "task", registry=ToolRegistry(),
        client=server.client(), tracer=Tracer(runs_dir=tmp_path))
    repair = server.requests[1]["messages"]
    assert [m["role"] for m in repair] == ["system", "user", "user"]
    assert "x" * 100 not in str(repair) and "rejected" in repair[-1]["content"]
