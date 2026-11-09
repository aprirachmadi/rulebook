"""M6 exports: final labels, receipt JSON, guideline copy, traces, loop report (SPEC section 6).

Reads labeler votes and run records only, never v2. The v2 metrics report is `eval/` (M8).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from rulebook.config import ROOT, load_config
from rulebook.data.store import load_docs, load_votes, load_votes_full
from rulebook.tools.guideline import GUIDELINES, changelog
from rulebook.tools.receipt import parse_amount, sum_check

WordKey = tuple[str, int, int]
EXPORTS = ROOT / "exports"


def final_labels(db: sqlite3.Connection, run_id: str, agents: list[str],
                 base_run_id: str | None = None,
                 rules: dict[WordKey, str] | None = None) -> dict[WordKey, tuple[str, str, str]]:
    """Per word: (label, status, rule_id). Status: accepted (all agree, confident),
    low_confidence (all agree, someone unsure), majority (2+ agree), ambiguous (all differ),
    relabeled (majority changed after a guideline patch; rule_id = the patch's rule)."""
    min_conf = load_config()["pipeline"]["accept_min_confidence"]
    votes = load_votes_full(db, run_id)
    base = load_votes(db, base_run_id) if base_run_id else {}
    out: dict[WordKey, tuple[str, str, str]] = {}
    for key, v in votes.items():
        top, n = Counter(lab for lab, _, _ in v.values()).most_common(1)[0]
        if n == len(agents):
            status = "accepted" if all(c >= min_conf for _, c, _ in v.values()) else \
                "low_confidence"
        else:
            status = "majority" if n >= 2 else "ambiguous"
        rule = ""
        if base_run_id and key in base and status != "ambiguous":
            before = Counter(base[key].values()).most_common(1)[0][0]
            if before != top:
                status, rule = "relabeled", (rules or {}).get(key, "")
        out[key] = (top, status, rule)
    db.execute("DELETE FROM final_labels WHERE run_id = ?", (run_id,))
    db.executemany("INSERT INTO final_labels VALUES (?, ?, ?, ?, ?, ?, ?)",
                   [(run_id, *k, *val) for k, val in out.items()])
    db.commit()
    return out


def export_receipts(db: sqlite3.Connection, run_id: str, version: str,
                    labels: dict[WordKey, tuple[str, str, str]], out_dir: Path) -> list[Path]:
    """exports/receipts/<id>.json: lines -> spans of consecutive words with one final label."""
    votes = load_votes(db, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in load_docs(db, run_id):
        lines, all_spans = [], []
        for ln in d.lines:
            spans: list[dict] = []
            for wi, w in enumerate(ln.words):
                key = (d.receipt_id, ln.line_idx, wi)
                label, status, rule = labels.get(key, ("", "unlabeled", ""))
                if spans and spans[-1]["label"] == label:
                    spans[-1]["text"] += " " + w.text
                    if status == "relabeled":
                        spans[-1]["status"], spans[-1]["rule"] = status, rule
                    continue
                spans.append({"text": w.text, "label": label, "status": status, "rule": rule,
                              "votes": votes.get(key, {})})
            for sp in spans:
                sp["value"] = parse_amount(sp["text"]) if "price" in sp["label"] else None
                all_spans.append((sp["text"], sp["label"]))
            lines.append({"line_idx": ln.line_idx, "text": ln.text, "ocr_conf": ln.conf,
                          "spans": spans})
        check = sum_check(all_spans)
        rec = {"receipt_id": d.receipt_id, "run_id": run_id, "guideline_version": version,
               "text_source": d.text_source, "lines": lines,
               "checks": {"sum_matches_total": check.passed, "details": check.checks}}
        path = out_dir / f"{d.receipt_id}.json"
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    return paths


def export_guideline(version: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{version}.md"
    shutil.copyfile(GUIDELINES / f"{version}.md", dest)
    return dest


def export_traces(run_ids: list[str], out_dir: Path) -> list[Path]:
    runs = ROOT / load_config()["harness"]["runs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for rid in run_ids:
        src = runs / rid / "trace.jsonl"
        if src.exists():
            dest = out_dir / f"{rid}.jsonl"
            shutil.copyfile(src, dest)
            paths.append(dest)
    return paths


def loop_report(db: sqlite3.Connection, loop_id: str, out_dir: Path) -> Path:
    """exports/reports/loop_<id>.md: per batch versions, alpha, patches and decisions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, status = db.execute("SELECT config_json, status FROM loops WHERE loop_id = ?",
                             (loop_id,)).fetchone()
    rows = db.execute("SELECT batch, run_id, relabel_run_id, version_before, version_after, "
                      "alpha_before, alpha_after, patches_json, timings_json FROM loop_batches "
                      "WHERE loop_id = ? ORDER BY batch", (loop_id,)).fetchall()
    fmt = lambda a: f"{a:.3f}" if a is not None else "-"  # noqa: E731
    out = [f"# Loop {loop_id}", "", f"Status: {status}. Config: `{cfg}`", "",
           "| batch | run | relabel run | guideline | alpha before | alpha after |",
           "|---|---|---|---|---|---|"]
    for b, rid, rrid, vb, va, ab, aa, _, _ in rows:
        out.append(f"| {b} | {rid} | {rrid or '-'} | {vb} -> {va} | {fmt(ab)} | {fmt(aa)} |")
    for b, rid, _, _, _, _, _, pj, tj in rows:
        out += ["", f"## Batch {b}", "", f"Timings (s): {tj}", ""]
        imp = ROOT / load_config()["harness"]["runs_dir"] / rid / "improve" / "improve.json"
        if imp.exists() and (res := json.loads(imp.read_text())).get("orchestrator") == "agent":
            rep = res.get("orchestrator_report") or {}
            out.append(f"Orchestrator agent ({res.get('orchestrator_stop')}, "
                       f"{res.get('orchestrator_steps')} steps): {rep.get('summary', '-')}")
            out += [f"- cluster {d['cluster_id']}: {d['action']} ({d['reason'] or 'no reason'})"
                    for d in res.get("orchestrator_decisions", [])]
            out.append("")
        for p in json.loads(pj or "[]"):
            v = p.get("verification", {})
            out.append(f"- rule {p['rule_number']} ({p['patch_id']}): **{p['decision']}** by "
                       f"{p.get('approver') or '-'}; verifier {v.get('verdict')} "
                       f"(gold {v.get('gold_accuracy_before')} -> {v.get('gold_accuracy_after')}, "
                       f"cluster alpha {v.get('cluster_alpha_before')} -> "
                       f"{v.get('cluster_alpha_after')}). Rule: {p['rule_text']}")
    out += ["", "## Guideline changelog", ""] + [f"- {c}" for c in changelog()]
    path = out_dir / f"loop_{loop_id}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
