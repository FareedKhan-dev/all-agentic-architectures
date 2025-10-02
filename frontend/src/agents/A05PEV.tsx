import { useState } from "react";
export default function A05PEV() {
  const [task, setTask] = useState("Écrire un plan, exécuter mentalement, puis vérifier.");
  const [out, setOut] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/a05/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task }) }).then(r => r.json());
      setOut(r);
    } catch (e: any) { setOut({ answer: `Erreur: ${e.message}` }); }
    finally { setLoading(false); }
  };
  return (<section><h2>A05 — PEV</h2>
    <input value={task} onChange={e => setTask(e.target.value)} />
    <button onClick={run} disabled={loading}>{loading ? "…" : "Exécuter"}</button>
    <div className="result"><pre>{out.answer || "—"}</pre></div>
  </section>);
}
