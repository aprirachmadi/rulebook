import { useEffect, useState } from "react";
import Dashboard from "./Dashboard.jsx";
import Review from "./Review.jsx";
import Traces from "./Traces.jsx";
import { get } from "./api.js";

const SCREENS = { dashboard: "Dashboard", review: "Review queue", traces: "Trace viewer" };

function useHash() {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/");
  const [parts, setParts] = useState(read());
  useEffect(() => {
    const on = () => setParts(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return parts;
}

export default function App() {
  const [screen, arg] = useHash();
  const current = SCREENS[screen] ? screen : "dashboard";
  const [pending, setPending] = useState(0);
  useEffect(() => {
    const tick = () => get("/api/review").then((r) => setPending(r.pending.length)).catch(() => {});
    tick();
    const t = setInterval(tick, 3000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="app">
      <header>
        <strong>Rulebook</strong>
        <nav>
          {Object.entries(SCREENS).map(([k, label]) => (
            <a key={k} href={`#/${k}`} className={k === current ? "active" : ""}>
              {label}
              {k === "review" && pending > 0 && <span className="badge">{pending}</span>}
            </a>
          ))}
        </nav>
      </header>
      <main>
        {current === "dashboard" && <Dashboard />}
        {current === "review" && <Review />}
        {current === "traces" && <Traces runId={arg} />}
      </main>
    </div>
  );
}
