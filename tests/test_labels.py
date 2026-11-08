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
    full_scope,
    gold_doc,
    render,
    sort_spans,
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
    assert f"lines 0-{len(doc.lines) - 1} of {len(doc.lines)}" in text


def test_render_chunk_and_repair_scope(doc) -> None:
    chunk = render(doc, full_scope(doc, [2, 3]))
    assert "[2] (" in chunk and "[0] (" not in chunk and "lines 2-3" in chunk
    long_line = max(doc.lines, key=lambda ln: len(ln.words))
    repair = render(doc, {long_line.line_idx: [0]}, note="fix these")
    assert "fix these" in repair and "<- label only words [0]" in repair


def _spans(scope, label="menu.nm"):
    return [Span(line_idx=li, words=ws, label=label, confidence=0.9) for li, ws in scope.items()]


def test_sort_spans_keeps_good_and_returns_the_rest(doc) -> None:
    scope = full_scope(doc, [0, 1, 2])
    acc, todo, problems = sort_spans(scope, _spans(scope))
    assert len(acc) == sum(len(w) for w in scope.values()) and todo == {} and problems == []
    acc, todo, problems = sort_spans(scope, _spans(scope)[1:])  # line 0 missing
    assert todo == {0: scope[0]} and any("no label" in p for p in problems)
    extra = Span(line_idx=99, words=[0], label="menu.nm", confidence=1)
    _, _, problems = sort_spans(scope, [*_spans(scope), extra])
    assert any("not asked for" in p for p in problems)


def test_sort_spans_conflicting_labels_go_back_to_todo(doc) -> None:
    scope = full_scope(doc, [0])
    spans = [*_spans(scope), Span(line_idx=0, words=[0], label="menu.price", confidence=0.5)]
    acc, todo, problems = sort_spans(scope, spans)
    assert (0, 0) not in acc and 0 in todo[0] and any("two different" in p for p in problems)
    same = [*_spans(scope), *_spans(scope)]  # same label twice is fine
    assert sort_spans(scope, same)[1] == {}


def test_answer_schema_bounds_words_per_line(doc) -> None:
    scope = full_scope(doc, [0, 1])
    schema = batch_model(scope, LABELS).model_json_schema()
    alts = schema["properties"]["labels"]["items"]["anyOf"]
    assert [a["properties"]["line_idx"]["const"] for a in alts] == [0, 1]
    for alt, words in zip(alts, scope.values(), strict=True):
        assert alt["properties"]["words"]["items"]["enum"] == words
        assert alt["properties"]["label"]["enum"] == list(LABELS)
        assert "reason" not in alt["required"]


def test_batch_model_enforces_label_enum(doc) -> None:
    scope = full_scope(doc, [0])
    model = batch_model(scope, LABELS)
    assert isinstance(model.model_validate({"labels": [s.model_dump() for s in _spans(scope)]}),
                      LabelBatch)
    with pytest.raises(ValidationError):
        model.model_validate({"labels": [s.model_dump() for s in _spans(scope, "nope")]})


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
    scope = full_scope(doc)
    accepted, _, _ = sort_spans(scope, _spans(scope))
    save_votes(db, "r1", doc, "labeler_a", "m", "v1", accepted)
    assert load_docs(db, "r1")[0] == doc
    votes = load_votes(db, "r1")
    assert len(votes) == sum(len(ln.words) for ln in doc.lines)
    assert all(v == {"labeler_a": "menu.nm"} for v in votes.values())


def test_alpha_is_none_with_a_single_labeler() -> None:
    assert alpha([{"a": "x"}, {"a": "y"}], ["a", "b", "c"]) is None


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


def test_label_doc_chunks_and_targeted_repair(doc, tmp_path, monkeypatch) -> None:
    """Each chunk's first answer skips word 0 of its first line; the repair is asked only
    for that word, and the merged result covers every word."""
    import json

    from rulebook.harness.agents import Agent
    from rulebook.harness.skills import SkillLibrary
    from rulebook.harness.trace import Tracer
    from rulebook.pipeline import labeling
    from tests.test_harness import FakeServer, _reply

    def answer(spans: list[Span]) -> dict:
        return _reply(content=json.dumps({"labels": [x.model_dump() for x in spans]}))

    chunk_list = labeling.chunks(doc, 3)
    replies = []
    for lines in chunk_list:
        scope = full_scope(doc, lines)
        first_line = lines[0]
        first = [Span(line_idx=li, words=[w for w in ws if (li, w) != (first_line, 0)],
                      label="menu.nm", confidence=0.9)
                 for li, ws in scope.items() if [w for w in ws if (li, w) != (first_line, 0)]]
        replies += [answer(first), answer(_spans({first_line: [0]}, "menu.price"))]
    server = FakeServer(*replies)
    real_run = labeling.run_agent
    monkeypatch.setattr(labeling, "run_agent",
                        lambda a, t, **kw: real_run(a, t, client=server.client(), **kw))
    st = labeling.LabelerStats("labeler_a", "fake")
    got, failed = labeling.label_doc(Agent("labeler_a", "sys", "fake"), doc, LABELS, 3,
                                     Tracer(runs_dir=tmp_path), SkillLibrary(tmp_path), st)
    assert failed == {} and len(got) == len(doc.word_keys())
    assert st.calls == 2 * len(chunk_list) and st.repairs == len(chunk_list)
    assert all(got[(c[0], 0)].label == "menu.price" for c in chunk_list)
    repair_task = server.requests[1]["messages"][1]["content"]
    assert "Label only the words marked below" in repair_task
