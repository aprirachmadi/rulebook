from eval.generalization import Scores, verdict
from rulebook.pipeline import labeling


def _s(alpha: float, majority: float, review: float) -> Scores:
    return Scores("r", "v", 1, 10, 10, alpha, majority=majority, review_share=review)


def test_verdict_needs_all_three() -> None:
    before = _s(0.60, 0.70, 0.40)
    assert all(verdict(before, _s(0.65, 0.72, 0.35), 0.02).values())
    got = verdict(before, _s(0.65, 0.715, 0.35), 0.02)
    assert list(got.values()) == [False, True, True]
    got = verdict(before, _s(0.60, 0.80, 0.41), 0.02)
    assert list(got.values()) == [True, False, False]


def test_run_batch_restores_current_guideline(monkeypatch) -> None:
    synced: list[str | None] = []
    monkeypatch.setattr(labeling, "sync_skill", lambda v=None: synced.append(v) or (v or "v9"))
    monkeypatch.setattr(labeling, "_run_batch", lambda *a: {"version": a[-1]})
    assert labeling.run_batch("ocr", 1, 0, None, "v1") == {"version": "v1"}
    assert synced == ["v1", None]
    synced.clear()
    assert labeling.run_batch("ocr", 1)["version"] == "v9"
    assert synced == [None]
