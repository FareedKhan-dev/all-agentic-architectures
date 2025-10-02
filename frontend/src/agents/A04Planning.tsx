import { useState } from "react";
export default function A04Planning() {
  const [task, setTask] = useState("Préparer un cours sur les graphes pour des lycéens.");
  const [out, setOut] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/a04/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task }) }).then(r => r.json());
      setOut(r);
    } catch (e: any) { setOut({ answer: `Erreur: ${e.message}` }); }
    finally { setLoading(false); }
  };
  return (<section><h2>A04 — Planning</h2>
    <input value={task} onChange={e => setTask(e.target.value)} />
    <button onClick={run} disabled={loading}>{loading ? "…" : "Planifier"}</button>
    <div className="result"><h3>Plan (synthèse)</h3><pre>{out.answer || "—"}</pre></div>
  </section>);
}
