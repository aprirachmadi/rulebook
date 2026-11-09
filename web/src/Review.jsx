import { useEffect, useState } from "react";
import { fmt, get, post } from "./api.js";

const short = (a) => a.replace("labeler_", "");

function Line({ line }) {
  return (
    <div className="line">
      <div className="lineHead">
        <code>{line.receipt_id}:{line.line_idx}</code> {line.reason && <em>{line.reason}</em>}
        {line.ocr_conf != null && <span className="muted"> OCR conf {fmt(line.ocr_conf, 2)}</span>}
      </div>
      <div className="words">
        {line.words.map((w) => {
          const labs = Object.values(w.votes).map((v) => v.label);
          const agree = labs.length && new Set(labs).size === 1;
          return (
            <div key={w.i} className={`word ${w.in_item ? "item" : ""} ${agree ? "" : "split"}`}>
              <div className="wtext">{w.text}</div>
              {Object.entries(w.votes).map(([a, v]) => (
                <div key={a} className="vote" title={v.reason || ""}>
                  {short(a)}: {v.label} <span className="muted">{fmt(v.confidence, 1)}</span>
                </div>
              ))}
              {w.final && <div className="final">→ {w.final.label} ({w.final.status}
                {w.final.rule && `, rule ${w.final.rule}`})</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Verifier({ v }) {
  if (!v || !v.verdict) return null;
  return (
    <div>
      <p>
        Verifier (advisory): <span className={`verdict ${v.verdict}`}>{v.verdict}</span>
        {v.fail_reasons?.length > 0 && <span className="muted"> — {v.fail_reasons.join("; ")}</span>}
      </p>
      <table>
        <thead><tr><th>labeler</th><th>gold accuracy before</th><th>after patch</th>
          <th>change</th></tr></thead>
        <tbody>
          {Object.entries(v.gold_accuracy_by_labeler || {}).map(([a, [b, c]]) => (
            <tr key={a}><td>{a}</td><td>{fmt(b)}</td><td>{fmt(c)}</td>
              <td>{(c - b >= 0 ? "+" : "") + fmt(c - b)}</td></tr>
          ))}
          <tr><td><strong>mean</strong></td><td>{fmt(v.gold_accuracy_before)}</td>
            <td>{fmt(v.gold_accuracy_after)}</td>
            <td>{fmt(v.gold_accuracy_after - v.gold_accuracy_before)}</td></tr>
        </tbody>
      </table>
      <p>Cluster alpha ({v.scored_on}, {v.held_out_words} words): {fmt(v.cluster_alpha_before)} →{" "}
        {fmt(v.cluster_alpha_after)}. {v.gold_verified ? "" : "Not gold-verified (no gold lines)."}</p>
    </div>
  );
}

function Pending({ item, onDone }) {
  const [text, setText] = useState(item.args.rule_text || "");
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const send = (decision) => {
    setBusy(true);
    post(`/api/approvals/${item.id}`, { decision, note,
      rule_text: decision === "edit" ? text : undefined })
      .then(onDone).catch((e) => { setErr(String(e)); setBusy(false); });
  };
  const edited = item.tool === "apply_patch" && text.trim() !== item.args.rule_text;
  const p = item.patch, c = item.cluster;
  return (
    <div className="card">
      <h3>#{item.id} {item.tool === "apply_patch" ? `Proposed rule ${p?.rule_number ?? ""}
        (section ${p?.section ?? "?"})` : "Relabel the batch"}
        <span className="muted small"> loop {item.loop_id}</span></h3>
      {item.tool === "apply_patch" && p && (
        <>
          <label className="block">Rule text (edit, then “Approve edited”)
            <textarea rows="4" value={text} onChange={(e) => setText(e.target.value)} />
          </label>
          <p><strong>Examples:</strong> {p.examples.map((e) => <code key={e}>{e}</code>)}</p>
          <p><strong>Rationale:</strong> {p.rationale}</p>
          <p className="muted">Cluster: labels {c.labels.join(" vs ")}, pattern “{c.pattern}”,
            {" "}{c.size} items, {p.affected_words} words. Diagnosis {c.diagnosis}:
            {" "}{c.diagnosis_reason}</p>
          <Verifier v={p.verification} />
          <h4>Receipt lines in this cluster (first {item.lines.length})</h4>
          {item.lines.map((l) => <Line key={`${l.receipt_id}:${l.line_idx}`} line={l} />)}
        </>
      )}
      {item.tool === "relabel" && (
        <p>Relabel {item.context.receipts} receipts of run <code>{item.args.run_id}</code> with
          guideline {item.context.from} → <strong>{item.args.guideline_version}</strong>
          {" "}(runs all 3 labelers).</p>
      )}
      <div className="actions">
        {item.tool === "apply_patch" && edited
          ? <button disabled={busy} onClick={() => send("edit")}>Approve edited</button>
          : <button disabled={busy} onClick={() => send("approve")}>Approve</button>}
        <input placeholder="note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
        <button disabled={busy} className="reject" onClick={() => send("reject")}>Reject</button>
      </div>
      {err && <p className="error">{err}</p>}
    </div>
  );
}

function RunQueue() {
  const [runs, setRuns] = useState([]);
  const [run, setRun] = useState("");
  const [q, setQ] = useState(null);
  useEffect(() => {
    get("/api/dashboard").then((d) => {
      const ids = [...new Set(d.batches.map((b) => b.run_id))];
      setRuns(ids.length ? ids : d.runs.slice(0, 5).map((r) => r.run_id));
    });
  }, []);
  useEffect(() => { if (run) get(`/api/queue/${run}?limit=30`).then(setQ); }, [run]);
  return (
    <div className="card">
      <h3>Lines sent to humans</h3>
      <select value={run} onChange={(e) => setRun(e.target.value)}>
        <option value="">choose a run…</option>
        {runs.map((r) => <option key={r}>{r}</option>)}
      </select>
      {q && (
        <>
          <p className="muted">{Object.entries(q.counts).map(([k, v]) => `${k}: ${v}`)
            .join(" · ") || "nothing queued"}</p>
          {q.items.map((l, i) => <Line key={i} line={l} />)}
        </>
      )}
    </div>
  );
}

export default function Review() {
  const [r, setR] = useState(null);
  const load = () => get("/api/review").then(setR);
  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);
  if (!r) return <p>Loading…</p>;
  return (
    <div>
      <h2>Waiting for a decision ({r.pending.length})</h2>
      {r.pending.length === 0 && <p className="muted">Nothing pending. A running loop stops here
        when it needs an approval.</p>}
      {r.pending.map((item) => <Pending key={item.id} item={item} onDone={load} />)}
      <RunQueue />
      <div className="card">
        <h3>Recent decisions</h3>
        <table>
          <thead><tr><th>#</th><th>tool</th><th>decision</th><th>by</th><th>at</th>
            <th>note</th></tr></thead>
          <tbody>{r.decided.map((d) => (
            <tr key={d.id}><td>{d.id}</td><td>{d.tool} {d.args.patch_id || d.args.run_id}</td>
              <td>{d.status}</td><td>{d.approver}</td>
              <td className="small">{d.decided_at?.slice(0, 19).replace("T", " ")}</td>
              <td>{d.note}</td></tr>))}</tbody>
        </table>
      </div>
    </div>
  );
}
