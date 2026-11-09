import { useEffect, useState } from "react";
import { fmt, get } from "./api.js";

function Record({ r }) {
  const [open, setOpen] = useState(false);
  const body = r.event === "model"
    ? (r.args?.tool_calls?.length ? JSON.stringify(r.args.tool_calls, null, 1) : r.result)
    : r.event === "tool" ? `${JSON.stringify(r.args)}\n→ ${r.result}` : r.result;
  return (
    <tr className={`ev ${r.event}`} onClick={() => setOpen(!open)}>
      <td>{r.agent}</td><td>{r.step}</td><td>{r.event}</td><td>{r.tool || ""}</td>
      <td className="small">{fmt(r.latency_s, 1)}s · {r.prompt_tokens}+{r.completion_tokens} tok</td>
      <td className="body">{open ? <pre>{body}</pre>
        : <span className="muted">{String(body ?? "").slice(0, 120)}</span>}</td>
    </tr>
  );
}

export default function Traces({ runId }) {
  const [list, setList] = useState([]);
  const [t, setT] = useState(null);
  const [agent, setAgent] = useState("");
  const [page, setPage] = useState(0);
  const per = 200;
  useEffect(() => { get("/api/traces").then(setList); }, []);
  useEffect(() => { setPage(0); setAgent(""); }, [runId]);
  useEffect(() => {
    if (runId) get(`/api/traces/${runId}?agent=${agent}&limit=${per}&offset=${page * per}`)
      .then(setT);
  }, [runId, agent, page]);
  return (
    <div>
      <div className="card">
        <h3>Trace viewer</h3>
        <select value={runId || ""} onChange={(e) => { window.location.hash = `#/traces/${e.target.value}`; }}>
          <option value="">choose a run…</option>
          {list.map((x) => <option key={x.run_id} value={x.run_id}>{x.run_id} ({x.size_kb} KB)</option>)}
        </select>
        {t && (
          <span className="agents">
            <button className={agent === "" ? "on" : ""} onClick={() => setAgent("")}>all</button>
            {Object.entries(t.agents).map(([a, n]) => (
              <button key={a} className={agent === a ? "on" : ""} onClick={() => { setAgent(a); setPage(0); }}>
                {a} ({n})</button>))}
          </span>
        )}
      </div>
      {t && (
        <div className="card">
          <p className="muted">{t.total} records{agent && ` for ${agent}`}. Click a row to expand.
            {" "}Showing {page * per + 1}–{Math.min(t.total, (page + 1) * per)}.
            {page > 0 && <button className="link" onClick={() => setPage(page - 1)}>prev</button>}
            {(page + 1) * per < t.total && <button className="link" onClick={() => setPage(page + 1)}>next</button>}
          </p>
          <table className="trace">
            <thead><tr><th>agent</th><th>step</th><th>event</th><th>tool</th><th>cost</th>
              <th>content</th></tr></thead>
            <tbody>{t.records.map((r, i) => <Record key={i} r={r} />)}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}
