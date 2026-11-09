"""M6: the loop's mechanics with fake labelers/improve (no models): approval, apply, relabel,
exports, final labels; and the queue approver."""

import json
import threading
import time
from pathlib import Path

import pytest

from rulebook.config import load_config
from rulebook.data.store import connect, save_run, save_votes
from rulebook.harness import guard
from rulebook.harness.guard import ApprovalRequest, CLIApprover, QueueApprover
from rulebook.pipeline import exports, orchestrate
from rulebook.tools import guideline
from rulebook.tools.labels import DocLine, DocWord, LabelDoc, Span

AGENTS = ["labeler_a", "labeler_b", "labeler_c"]
GUIDE = "# Guideline v1\n\n## 1. Menu items\n\n- `menu.nm`: item name\n\n## 3. Total and payment\n"


def _doc() -> LabelDoc:
    return LabelDoc(receipt_id="train_0000", text_source="gold", width=100, height=100, lines=[
        DocLine(line_idx=0, text="Nasi 25.000", box=(0, 0, 50, 8), words=[
            DocWord(text="Nasi", box=(0, 0, 10, 8), ref=(0, 0)),
            DocWord(text="25.000", box=(20, 0, 30, 8), ref=(0, 1))]),
        DocLine(line_idx=1, text="TOTAL", box=(0, 10, 50, 18), words=[
            DocWord(text="TOTAL", box=(0, 10, 10, 18), ref=(1, 0))])])


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch):
    cfg = load_config()
    monkeypatch.setitem(cfg["storage"], "db_path", str(tmp_path / "t.db"))
    monkeypatch.setitem(cfg["harness"], "runs_dir", str(tmp_path / "runs"))
    gdir = tmp_path / "guidelines"
    gdir.mkdir()
    (gdir / "v1.md").write_text(GUIDE)
    for mod in (guideline, orchestrate, exports):
        monkeypatch.setattr(mod, "GUIDELINES", gdir)
    monkeypatch.setattr(guideline, "SKILL_PATH", tmp_path / "skill" / "SKILL.md")
    monkeypatch.setattr(exports, "EXPORTS", tmp_path / "exports")
    calls = {"n": 0}

    def fake_run_batch(text_source, batch_size, offset, labelers=None):
        calls["n"] += 1
        version = guideline.current_version()
        run_id = f"run{calls['n']}"
        db = connect()
        save_run(db, run_id, text_source, version, [_doc()])
        total = "total.total_price" if version != "v1" else "other"  # the patch fixes TOTAL
        for a in AGENTS:
            lab = {(0, 0): Span(line_idx=0, words=[0], label="menu.nm", confidence=0.9),
                   (0, 1): Span(line_idx=0, words=[1], label="menu.price", confidence=0.9),
                   (1, 0): Span(line_idx=1, words=[0], label=total if a != "labeler_c"
                                else "total.total_price", confidence=0.9)}
            save_votes(db, run_id, _doc(), a, "m", version, lab)
        out = tmp_path / "runs" / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"alpha_word_level": 0.5}))
        return {"run_id": run_id, "guideline_version": version, "alpha_word_level": 0.5,
                "labelers_run": AGENTS}

    def fake_improve(run_id):
        db = connect()
        db.execute("INSERT INTO clusters VALUES (?, 0, ?, 'total', 1, ?, 'diagnosed', "
                   "'GUIDELINE_GAP', 'r')", (run_id, json.dumps(["other", "total.total_price"]),
                                            json.dumps(["train_0000:1:0-0"])))
        db.execute("INSERT INTO patches VALUES (?, ?, 0, '3.1', '3', 'TOTAL gets "
                   "total.total_price.', ?, 'why', 1, 0, 'proposed', '{}', 'now', NULL, NULL)",
                   (f"{run_id}-c0", run_id, json.dumps(["TOTAL", "TOTAL", "TOTAL"])))
        db.commit()
        return {"timings_s": {"cluster": 0.0}, "patches": [
            {"patch_id": f"{run_id}-c0", "rule_number": "3.1", "verdict": "fail"}]}

    monkeypatch.setattr(orchestrate, "run_batch", fake_run_batch)
    monkeypatch.setattr(orchestrate, "improve", fake_improve)
    return tmp_path


def test_loop_approve_applies_relabels_and_exports(sandbox: Path, monkeypatch) -> None:
    answers = iter(["a", "a"])  # approve the patch, approve the relabel
    monkeypatch.setattr(orchestrate, "CLIApprover",
                        lambda: CLIApprover(ask=lambda _: next(answers), approver="tester"))
    loop_id = orchestrate.run_loop("gold", 1, 1, approver_kind="cli", loop_id="L1")
    assert guideline.current_version() == "v2"
    v2 = (sandbox / "guidelines" / "v2.md").read_text()
    assert "### 3.1 Rule\nTOTAL gets total.total_price." in v2
    assert "# Guideline v2\n" in v2
    assert guideline.changelog("v2") == ["v2: rule 3.1 added (run1-c0)"]
    rec = json.loads((sandbox / "exports" / "receipts" / "train_0000.json").read_text())
    assert rec["guideline_version"] == "v2" and rec["run_id"] == "run2"
    total = rec["lines"][1]["spans"][0]
    assert (total["label"], total["status"], total["rule"]) == ("total.total_price",
                                                                "relabeled", "3.1")
    assert rec["lines"][0]["spans"][1]["value"] == 25000
    db = connect()
    assert db.execute("SELECT status, approver FROM patches").fetchone() == ("approved", "tester")
    row = db.execute("SELECT run_id, relabel_run_id, version_before, version_after "
                     "FROM loop_batches WHERE loop_id = ?", (loop_id,)).fetchone()
    assert row == ("run1", "run2", "v1", "v2")
    assert (sandbox / "exports" / "reports" / "loop_L1.md").exists()
    assert (sandbox / "exports" / "guidelines" / "v2.md").exists()


def test_loop_reject_keeps_version(sandbox: Path, monkeypatch) -> None:
    answers = iter(["r", "not convinced"])
    monkeypatch.setattr(orchestrate, "CLIApprover",
                        lambda: CLIApprover(ask=lambda _: next(answers), approver="tester"))
    orchestrate.run_loop("gold", 1, 1, approver_kind="cli", loop_id="L2")
    assert guideline.current_version() == "v1"
    db = connect()
    assert db.execute("SELECT status FROM patches").fetchone() == ("rejected",)
    assert db.execute("SELECT relabel_run_id FROM loop_batches").fetchone() == (None,)


def test_queue_approver_blocks_until_decided(tmp_path: Path) -> None:
    path = tmp_path / "a.db"
    ap = QueueApprover(path, scope="L", poll_s=0.05)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault(
        "a", ap.request(ApprovalRequest("apply_patch", {"rule_text": "x"}, "orchestrator"))))
    t.start()
    for _ in range(100):
        rows = guard.connect_approvals(path).execute(
            "SELECT id FROM approvals WHERE status = 'pending'").fetchall()
        if rows:
            break
        time.sleep(0.02)
    assert t.is_alive()  # still waiting
    guard.decide(path, rows[0][0], "edit", "ui-user", {"rule_text": "y"})
    t.join(timeout=5)
    assert out["a"].decision == "edit" and out["a"].args == {"rule_text": "y"}
    with pytest.raises(ValueError):
        guard.decide(path, rows[0][0], "approve", "ui-user")
