"""M7 API: decisions from the UI reach the approval queue; dashboard and traces respond."""

from pathlib import Path

from fastapi.testclient import TestClient

from rulebook.api import app as api
from rulebook.harness import guard


def test_decide_edit_and_reject(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "a.db"
    monkeypatch.setattr(api, "db_path", lambda: path)
    db = guard.connect_approvals(path)
    for _ in range(2):
        db.execute("INSERT INTO approvals (scope, tool, agent, args_json, status) VALUES "
                   "('L', 'apply_patch', 'orchestrator', '{\"patch_id\": \"p\", "
                   "\"rule_text\": \"old\"}', 'pending')")
    db.commit()
    c = TestClient(api.app)
    assert c.post("/api/approvals/1", json={"decision": "edit"}).status_code == 400
    assert c.post("/api/approvals/1", json={"decision": "edit", "rule_text": "new"}).json()["ok"]
    assert c.post("/api/approvals/1", json={"decision": "approve"}).status_code == 409
    assert c.post("/api/approvals/2", json={"decision": "reject", "note": "no"}).json()["ok"]
    rows = db.execute("SELECT status, decided_args_json, note FROM approvals "
                      "ORDER BY id").fetchall()
    assert rows[0][0] == "edit" and '"new"' in rows[0][1]
    assert rows[1] == ("reject", None, "no")


def test_dashboard_and_traces_respond() -> None:
    c = TestClient(api.app)
    d = c.get("/api/dashboard").json()
    assert {"current_version", "changelog", "runs", "alpha_by_version", "loops"} <= set(d)
    assert c.get("/api/traces").status_code == 200
    assert c.get("/api/traces/does-not-exist").status_code == 404
