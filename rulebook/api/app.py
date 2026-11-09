"""M7 API (FastAPI): dashboard, review queue (decisions drive the loop), trace viewer.

Run: `poetry run uvicorn rulebook.api.app:app --port 8000`. Serves the built React app
(`web/dist`) at `/`. Reads the run store and traces only, never v2 labels.
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rulebook.config import ROOT, load_config
from rulebook.data.store import connect, load_docs, load_votes_full
from rulebook.harness import guard
from rulebook.harness.trace import read_trace
from rulebook.pipeline.orchestrate import db_path, new_loop_id
from rulebook.tools.guideline import changelog, current_version, guideline_text, versions

app = FastAPI(title="Rulebook")
WEB = ROOT / "web" / "dist"
SAMPLE_LINES = 10


def _runs_dir() -> Path:
    return ROOT / load_config()["harness"]["runs_dir"]


def _summary(run_id: str) -> dict:
    p = _runs_dir() / run_id / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/dashboard")
def dashboard() -> dict:
    db = connect()
    runs = []
    for run_id, src, version, created, ids in db.execute(
            "SELECT run_id, text_source, guideline_version, created_at, receipt_ids FROM runs "
            "ORDER BY created_at DESC"):
        s = _summary(run_id)
        runs.append({"run_id": run_id, "text_source": src, "guideline_version": version,
                     "created_at": created, "receipts": len(json.loads(ids or "[]")),
                     "words": s.get("words"), "alpha": s.get("alpha_word_level"),
                     "agreement": s.get("agreement_pattern"),
                     "labelers": [{"agent": x.get("agent"), "model": x.get("model"),
                                   "complete": x.get("receipts_complete"),
                                   "calls": x.get("calls"), "seconds": round(x.get("seconds", 0))}
                                  for x in s.get("labelers", [])]})
    current = set(load_config()["models"]["labelers"].values())
    by_version: dict[tuple[str, str], list[float]] = {}
    for r in runs:  # only runs by the current 3 labeler models are comparable
        r["current_labelers"] = {x["model"] for x in r["labelers"]} == current
        if r["alpha"] is not None and r["current_labelers"]:
            by_version.setdefault((r["guideline_version"], r["text_source"]),
                                  []).append(r["alpha"])
    alpha_by_version = [{"version": v, "text_source": src, "runs": len(by_version[(v, src)]),
                         "alpha_mean": sum(by_version[(v, src)]) / len(by_version[(v, src)])}
                        for v in versions() for src in ("ocr", "gold")
                        if (v, src) in by_version]
    loops = [{"loop_id": lid, "created_at": c, "config": json.loads(cfg), "status": st,
              "step": step, "updated_at": up}
             for lid, c, cfg, st, step, up in db.execute(
                 "SELECT loop_id, created_at, config_json, status, step, updated_at FROM loops "
                 "ORDER BY created_at DESC")]
    batches = [{"loop_id": lid, "batch": b, "run_id": rid, "relabel_run_id": rr,
                "version_before": vb, "version_after": va, "alpha_before": ab,
                "alpha_after": aa, "patches": json.loads(pj or "[]"),
                "timings": json.loads(tj or "{}")}
               for lid, b, rid, rr, vb, va, ab, aa, pj, tj in db.execute(
                   "SELECT * FROM loop_batches ORDER BY loop_id DESC, batch")]
    patch_status = dict(Counter(s for (s,) in db.execute("SELECT status FROM patches")))
    return {"current_version": current_version(), "changelog": changelog(), "runs": runs,
            "alpha_by_version": alpha_by_version, "loops": loops, "batches": batches,
            "patch_status": patch_status}


@app.get("/api/guideline/{version}")
def guideline(version: str) -> dict:
    if version not in versions():
        raise HTTPException(404, f"no guideline {version}")
    return {"version": version, "text": guideline_text(version), "changelog": changelog(version)}


def _line_view(docs: dict, votes: dict, finals: dict, rid: str, li: int,
               words: set[int] | None = None) -> dict:
    d = docs[rid]
    ln = next(x for x in d.lines if x.line_idx == li)
    return {"receipt_id": rid, "line_idx": li, "text": ln.text, "ocr_conf": ln.conf,
            "words": [{"i": wi, "text": w.text, "in_item": words is None or wi in words,
                       "votes": {a: {"label": v[0], "confidence": v[1], "reason": v[2]}
                                 for a, v in votes.get((rid, li, wi), {}).items()},
                       "final": finals.get((rid, li, wi))}
                      for wi, w in enumerate(ln.words)]}


def _run_view(db, run_id: str):
    docs = {d.receipt_id: d for d in load_docs(db, run_id)}
    votes = load_votes_full(db, run_id)
    finals = {(r, li, wi): {"label": lab, "status": st, "rule": rule}
              for r, li, wi, lab, st, rule in db.execute(
                  "SELECT receipt_id, line_idx, word_idx, label, status, rule_id FROM "
                  "final_labels WHERE run_id = ?", (run_id,))}
    return docs, votes, finals


def _items(items_json: str) -> list[tuple[str, int, set[int]]]:
    out = []
    for item in json.loads(items_json):
        rid, li, span = item.rsplit(":", 2)
        w0, wn = (int(x) for x in span.split("-"))
        out.append((rid, int(li), set(range(w0, wn + 1))))
    return out


@app.get("/api/review")
def review() -> dict:
    """Pending approvals (patches and relabels) with everything needed to decide."""
    db = connect()
    adb = guard.connect_approvals(db_path())
    pending = []
    for aid, scope, tool, args_json, ctx_json, created in adb.execute(
            "SELECT id, scope, tool, args_json, context_json, created_at FROM approvals "
            "WHERE status = 'pending' ORDER BY id"):
        args, ctx = json.loads(args_json), json.loads(ctx_json or "{}")
        item = {"id": aid, "loop_id": scope, "tool": tool, "args": args, "context": ctx,
                "created_at": created}
        if tool == "apply_patch":
            row = db.execute(
                "SELECT p.run_id, p.rule_number, p.section, p.examples_json, p.rationale, "
                "p.affected_count, p.held_out, p.verification_json, c.labels, c.pattern, "
                "c.size, c.diagnosis, c.diagnosis_reason, c.items_json FROM patches p JOIN "
                "clusters c ON c.run_id = p.run_id AND c.cluster_id = p.cluster_id "
                "WHERE p.patch_id = ?", (args["patch_id"],)).fetchone()
            if row:
                (run_id, num, section, ex, why, affected, held_out, ver, labels, pattern,
                 size, diag, diag_reason, items_json) = row
                docs, votes, finals = _run_view(db, run_id)
                item["patch"] = {"run_id": run_id, "rule_number": num, "section": section,
                                 "examples": json.loads(ex), "rationale": why,
                                 "affected_words": affected, "held_out": bool(held_out),
                                 "verification": json.loads(ver or "{}")}
                item["cluster"] = {"labels": json.loads(labels), "pattern": pattern,
                                   "size": size, "diagnosis": diag,
                                   "diagnosis_reason": diag_reason}
                item["lines"] = [_line_view(docs, votes, finals, rid, li, ws)
                                 for rid, li, ws in _items(items_json)[:SAMPLE_LINES]]
        pending.append(item)
    decided = [{"id": i, "loop_id": s, "tool": t, "args": json.loads(a),
                "status": st, "approver": ap, "decided_at": d, "note": n}
               for i, s, t, a, st, ap, d, n in adb.execute(
                   "SELECT id, scope, tool, COALESCE(decided_args_json, args_json), status, "
                   "approver, decided_at, note FROM approvals WHERE status != 'pending' "
                   "ORDER BY id DESC LIMIT 20")]
    return {"pending": pending, "decided": decided}


@app.get("/api/queue/{run_id}")
def queue(run_id: str, limit: int = 50) -> dict:
    """Items routed to humans for a run (low confidence, small clusters, non-gap clusters),
    with their receipt lines, labels and status."""
    db = connect()
    docs, votes, finals = _run_view(db, run_id)
    rows = db.execute("SELECT kind, ref_json, reason FROM review_queue WHERE run_id = ?",
                      (run_id,)).fetchall()
    counts = dict(Counter(reason for _, _, reason in rows))
    out = []
    for kind, ref_json, reason in rows:
        ref = json.loads(ref_json)
        if kind != "item" or len(out) >= limit:
            continue
        rid, li, ws = _items(json.dumps([ref["item_id"]]))[0]
        if rid in docs:
            out.append({"reason": reason, **_line_view(docs, votes, finals, rid, li, ws)})
    return {"run_id": run_id, "counts": counts, "items": out}


@app.get("/api/receipts/{run_id}/{receipt_id}")
def receipt(run_id: str, receipt_id: str) -> dict:
    db = connect()
    docs, votes, finals = _run_view(db, run_id)
    if receipt_id not in docs:
        raise HTTPException(404, "no such receipt in this run")
    return {"run_id": run_id, "receipt_id": receipt_id,
            "lines": [_line_view(docs, votes, finals, receipt_id, ln.line_idx)
                      for ln in docs[receipt_id].lines]}


class DecisionIn(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    rule_text: str | None = Field(default=None, description="for edit on apply_patch")
    note: str = ""


@app.post("/api/approvals/{approval_id}")
def decide(approval_id: int, body: DecisionIn) -> dict:
    adb = guard.connect_approvals(db_path())
    row = adb.execute("SELECT tool, args_json FROM approvals WHERE id = ?",
                      (approval_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such approval")
    args = json.loads(row[1])
    if body.decision == "edit":
        if row[0] != "apply_patch" or not (body.rule_text or "").strip():
            raise HTTPException(400, "edit needs a rule_text (apply_patch only)")
        args["rule_text"] = body.rule_text.strip()
    try:
        guard.decide(db_path(), approval_id, body.decision, f"ui:{getpass.getuser()}",
                     args if body.decision == "edit" else None, body.note)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True}


class LoopIn(BaseModel):
    batch_size: int = Field(default=10, ge=1, le=800)
    batches: int = Field(default=1, ge=1, le=80)
    offset: int = Field(default=0, ge=0)
    text_source: Literal["ocr", "gold"] = "ocr"
    max_clusters: int = Field(default=0, ge=0)


@app.post("/api/loops")
def start_loop(body: LoopIn) -> dict:
    """Start a loop as a background process (one at a time: the GPU holds one model)."""
    db = connect()
    if db.execute("SELECT 1 FROM loops WHERE status = 'running'").fetchone():
        raise HTTPException(409, "a loop is already running")
    loop_id = new_loop_id()
    log = _runs_dir() / f"{loop_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "rulebook.pipeline.run", "loop", "--loop-id", loop_id,
           "--batch-size", str(body.batch_size), "--batches", str(body.batches),
           "--offset", str(body.offset), "--text-source", body.text_source,
           "--max-clusters", str(body.max_clusters), "--approver", "queue"]
    with log.open("w") as fh:
        subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True,
                         env={**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"})
    return {"loop_id": loop_id, "log": str(log)}


@app.get("/api/loops/{loop_id}/log")
def loop_log(loop_id: str, tail: int = 60) -> dict:
    log = _runs_dir() / f"{loop_id}.log"
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    return {"lines": lines[-tail:]}


@app.get("/api/traces")
def traces() -> list[dict]:
    out = []
    for p in sorted(_runs_dir().glob("*/trace.jsonl"), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        out.append({"run_id": p.parent.name, "size_kb": round(p.stat().st_size / 1024),
                    "modified": p.stat().st_mtime})
    return out


@app.get("/api/traces/{run_id}")
def trace(run_id: str, agent: str = "", limit: int = 500, offset: int = 0) -> dict:
    p = _runs_dir() / run_id / "trace.jsonl"
    if not p.exists() or p.parent.parent != _runs_dir():
        raise HTTPException(404, "no such trace")
    recs = read_trace(p)
    agents = dict(Counter(r.agent for r in recs))
    if agent:
        recs = [r for r in recs if r.agent == agent]
    return {"run_id": run_id, "agents": agents, "total": len(recs),
            "records": [r.model_dump() for r in recs[offset:offset + limit]]}


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        return FileResponse(WEB / "index.html")
