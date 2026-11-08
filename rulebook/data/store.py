"""SQLite store (`rulebook.db`). M4 tables: runs, docs, votes, label_failures."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from rulebook.config import ROOT, load_config
from rulebook.tools.labels import LabelDoc, Span

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, text_source TEXT, guideline_version TEXT, created_at TEXT,
  receipt_ids TEXT
);
CREATE TABLE IF NOT EXISTS docs (
  run_id TEXT, receipt_id TEXT, doc_json TEXT, PRIMARY KEY (run_id, receipt_id)
);
CREATE TABLE IF NOT EXISTS votes (
  run_id TEXT, receipt_id TEXT, line_idx INTEGER, word_idx INTEGER, agent TEXT, model TEXT,
  guideline_version TEXT, label TEXT, confidence REAL, reason TEXT,
  PRIMARY KEY (run_id, receipt_id, line_idx, word_idx, agent)
);
CREATE TABLE IF NOT EXISTS label_failures (
  run_id TEXT, receipt_id TEXT, agent TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
  run_id TEXT, cluster_id INTEGER, labels TEXT, pattern TEXT, size INTEGER, items_json TEXT,
  status TEXT, diagnosis TEXT, diagnosis_reason TEXT, PRIMARY KEY (run_id, cluster_id)
);
CREATE TABLE IF NOT EXISTS patches (
  patch_id TEXT PRIMARY KEY, run_id TEXT, cluster_id INTEGER, rule_number TEXT, section TEXT,
  rule_text TEXT, examples_json TEXT, rationale TEXT, affected_count INTEGER, held_out INTEGER,
  status TEXT, verification_json TEXT, created_at TEXT, approver TEXT, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS review_queue (
  run_id TEXT, kind TEXT, ref_json TEXT, reason TEXT
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    db = sqlite3.connect(path or ROOT / load_config()["storage"]["db_path"])
    db.executescript(SCHEMA)
    return db


def save_run(db: sqlite3.Connection, run_id: str, text_source: str, version: str,
             docs: list[LabelDoc]) -> None:
    db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
               (run_id, text_source, version, datetime.now(UTC).isoformat(),
                json.dumps([d.receipt_id for d in docs])))
    db.executemany("INSERT INTO docs VALUES (?, ?, ?)",
                   [(run_id, d.receipt_id, d.model_dump_json()) for d in docs])
    db.commit()


def save_votes(db: sqlite3.Connection, run_id: str, doc: LabelDoc, agent: str, model: str,
               version: str, labels: dict[tuple[int, int], Span]) -> None:
    rows = [(run_id, doc.receipt_id, li, wi, agent, model, version, s.label, s.confidence,
             s.reason) for (li, wi), s in labels.items()]
    db.executemany("INSERT INTO votes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.commit()


def save_failure(db: sqlite3.Connection, run_id: str, receipt_id: str, agent: str,
                 error: str) -> None:
    db.execute("INSERT INTO label_failures VALUES (?, ?, ?, ?)",
               (run_id, receipt_id, agent, error[:2000]))
    db.commit()


def load_docs(db: sqlite3.Connection, run_id: str) -> list[LabelDoc]:
    rows = db.execute("SELECT doc_json FROM docs WHERE run_id = ? ORDER BY receipt_id", (run_id,))
    return [LabelDoc.model_validate_json(r[0]) for r in rows]


def load_votes(db: sqlite3.Connection, run_id: str) -> dict[tuple[str, int, int], dict[str, str]]:
    """(receipt_id, line_idx, word_idx) -> {agent: label}."""
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    for rid, li, wi, agent, label in db.execute(
        "SELECT receipt_id, line_idx, word_idx, agent, label FROM votes WHERE run_id = ?",
        (run_id,),
    ):
        out.setdefault((rid, li, wi), {})[agent] = label
    return out


Vote = tuple[str, float, str]  # label, confidence, reason


def load_votes_full(
    db: sqlite3.Connection, run_id: str
) -> dict[tuple[str, int, int], dict[str, Vote]]:
    """(receipt_id, line_idx, word_idx) -> {agent: (label, confidence, reason)}."""
    out: dict[tuple[str, int, int], dict[str, Vote]] = {}
    for rid, li, wi, agent, label, conf, reason in db.execute(
        "SELECT receipt_id, line_idx, word_idx, agent, label, confidence, reason FROM votes "
        "WHERE run_id = ?", (run_id,),
    ):
        out.setdefault((rid, li, wi), {})[agent] = (label, conf, reason or "")
    return out
