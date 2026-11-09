import { useEffect, useState } from "react";
import { fmt, get, post } from "./api.js";

const SOURCES = ["ocr", "gold"]; // fixed order: color follows the entity (slot 1, slot 2)

function AlphaChart({ rows }) {
  const versions = [...new Set(rows.map((r) => r.version))];
  if (!versions.length) return <p className="muted">No runs with the current labelers yet.</p>;
  const W = Math.max(320, versions.length * 120), H = 180, pad = 28, top = 16;
  const group = (W - pad) / versions.length;
  const bw = Math.min(28, group / 3);
  const y = (a) => top + (H - top - pad) * (1 - Math.max(0, Math.min(1, a)));
  return (
    <div>
      <div className="legend">
        {SOURCES.map((s, i) => (
          <span key={s}><i className={`sw s${i + 1}`} />{s} text</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
           aria-label="Word-level Krippendorff alpha per guideline version">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={pad} x2={W} y1={y(t)} y2={y(t)} className="grid" />
            <text x={pad - 6} y={y(t) + 4} className="axis" textAnchor="end">{t}</text>
          </g>
        ))}
        {versions.map((v, gi) => (
          <g key={v}>
            {SOURCES.map((s, si) => {
              const r = rows.find((x) => x.version === v && x.text_source === s);
              if (!r) return null;
              const x = pad + gi * group + group / 2 + (si - 1) * (bw + 2);
              return (
                <g key={s} className="bar">
                  <rect x={x} y={y(r.alpha_mean)} width={bw} height={y(0) - y(r.alpha_mean)}
                        rx="4" className={`s${si + 1}`}>
                    <title>{`${v}, ${s} text: alpha ${fmt(r.alpha_mean)} (${r.runs} runs)`}</title>
                  </rect>
                  <text x={x + bw / 2} y={y(r.alpha_mean) - 4} className="label"
                        textAnchor="middle">{fmt(r.alpha_mean, 2)}</text>
                </g>
              );
            })}
            <text x={pad + gi * group + group / 2} y={H - 8} className="axis"
                  textAnchor="middle">{v}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function StartLoop({ onStarted }) {
  const [f, setF] = useState({ batch_size: 10, batches: 1, offset: 50, text_source: "ocr",
                               max_clusters: 2, orchestrator: "agent" });
  const [err, setErr] = useState("");
  const set = (k) => (e) => setF({ ...f, [k]: ["text_source", "orchestrator"].includes(k)
    ? e.target.value : Number(e.target.value) });
  const start = () => post("/api/loops", f).then(onStarted).catch((e) => setErr(String(e)));
  return (
    <div className="card">
      <h3>Start a loop</h3>
      <div className="form">
        {["batch_size", "batches", "offset", "max_clusters"].map((k) => (
          <label key={k}>{k.replace("_", " ")}
            <input type="number" min="0" value={f[k]} onChange={set(k)} />
          </label>
        ))}
        <label>text source
          <select value={f.text_source} onChange={set("text_source")}>
            <option>ocr</option><option>gold</option>
          </select>
        </label>
        <label>orchestrator
          <select value={f.orchestrator} onChange={set("orchestrator")}>
            <option value="agent">agent (LLM)</option><option value="code">code (fallback)</option>
          </select>
        </label>
        <button onClick={start}>Start</button>
      </div>
      <p className="muted">One loop at a time. max clusters 0 = config default.</p>
      {err && <p className="error">{err}</p>}
    </div>
  );
}

function LoopLog({ loopId }) {
  const [lines, setLines] = useState([]);
  useEffect(() => {
    const tick = () => get(`/api/loops/${loopId}/log?tail=15`).then((r) => setLines(r.lines));
    tick();
    const t = setInterval(tick, 3000);
    return () => clearInterval(t);
  }, [loopId]);
  return <pre className="log">{lines.join("\n") || "(no output yet)"}</pre>;
}

export default function Dashboard() {
  const [d, setD] = useState(null);
  const [guide, setGuide] = useState(null);
  const load = () => get("/api/dashboard").then(setD);
  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);
  if (!d) return <p>Loading…</p>;
  const running = d.loops.find((l) => l.status === "running");
  return (
    <div className="grid2">
      <div className="card">
        <h3>Guideline {d.current_version}</h3>
        <p className="muted">Patches: {Object.entries(d.patch_status).map(([k, v]) => `${v} ${k}`)
          .join(", ") || "none"}</p>
        <h4>Changelog</h4>
        {d.changelog.length ? (
          <ul>{d.changelog.map((c) => <li key={c}>{c}</li>)}</ul>
        ) : <p className="muted">v1 (no rules added yet)</p>}
        <button className="link" onClick={() => get(`/api/guideline/${d.current_version}`)
          .then(setGuide)}>Show guideline text</button>
        {guide && <pre className="guide">{guide.text}</pre>}
      </div>

      <div className="card">
        <h3>Alpha per guideline version</h3>
        <AlphaChart rows={d.alpha_by_version} />
        <table>
          <thead><tr><th>version</th><th>text</th><th>runs</th><th>mean alpha</th></tr></thead>
          <tbody>{d.alpha_by_version.map((r) => (
            <tr key={r.version + r.text_source}>
              <td>{r.version}</td><td>{r.text_source}</td><td>{r.runs}</td>
              <td>{fmt(r.alpha_mean)}</td></tr>))}</tbody>
        </table>
        <p className="muted">Word-level Krippendorff alpha, runs by the current 3 labelers.</p>
      </div>

      {running ? (
        <div className="card wide">
          <h3>Loop {running.loop_id}: running</h3>
          <p><strong>Now:</strong> {running.step}
            {running.step.includes("waiting") && <> — <a href="#/review">decide in the review
              queue</a></>}</p>
          <LoopLog loopId={running.loop_id} />
        </div>
      ) : <div className="wide"><StartLoop onStarted={load} /></div>}

      <div className="card wide">
        <h3>Loops</h3>
        <table>
          <thead><tr><th>loop</th><th>status</th><th>step</th><th>config</th></tr></thead>
          <tbody>{d.loops.map((l) => (
            <tr key={l.loop_id}><td>{l.loop_id}</td><td>{l.status}</td><td>{l.step}</td>
              <td className="small">{JSON.stringify(l.config)}</td></tr>))}</tbody>
        </table>
        <h4>Batches</h4>
        <table>
          <thead><tr><th>loop</th><th>batch</th><th>run</th><th>relabel run</th>
            <th>guideline</th><th>alpha before → after</th><th>patches</th><th>timings (s)</th>
          </tr></thead>
          <tbody>{d.batches.map((b) => (
            <tr key={b.loop_id + b.batch}>
              <td>{b.loop_id}</td><td>{b.batch}</td>
              <td><a href={`#/traces/${b.run_id}`}>{b.run_id}</a></td>
              <td>{b.relabel_run_id ? <a href={`#/traces/${b.relabel_run_id}`}>
                {b.relabel_run_id}</a> : "-"}</td>
              <td>{b.version_before} → {b.version_after}</td>
              <td>{fmt(b.alpha_before)} → {fmt(b.alpha_after)}</td>
              <td>{b.patches.map((p) => `${p.rule_number}: ${p.decision}`).join("; ") || "-"}</td>
              <td className="small">{Object.entries(b.timings).map(([k, v]) => `${k} ${v}`)
                .join(", ")}</td>
            </tr>))}</tbody>
        </table>
      </div>

      <div className="card wide">
        <h3>Runs</h3>
        <table>
          <thead><tr><th>run</th><th>created</th><th>text</th><th>guideline</th>
            <th>receipts</th><th>words</th><th>alpha</th><th>labelers</th></tr></thead>
          <tbody>{d.runs.map((r) => (
            <tr key={r.run_id} className={r.current_labelers ? "" : "dim"}>
              <td><a href={`#/traces/${r.run_id}`}>{r.run_id}</a></td>
              <td className="small">{r.created_at?.slice(0, 16).replace("T", " ")}</td>
              <td>{r.text_source}</td><td>{r.guideline_version}</td><td>{r.receipts}</td>
              <td>{r.words ?? "-"}</td><td>{fmt(r.alpha)}</td>
              <td className="small">{r.labelers.map((x) => x.model).join(", ")}</td>
            </tr>))}</tbody>
        </table>
        <p className="muted">Greyed rows: runs by earlier labeler models.</p>
      </div>
    </div>
  );
}
