import { useState } from "react";
export default function A06Blackboard() {
  const [task, setTask] = useState("Génère deux idées et fusionne-les.");
  const [out, setOut] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/a06/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task }) }).then(r => r.json());
      setOut(r);
    } catch (e: any) { setOut({ answer: `Erreur: ${e.message}` }); }
    finally { setLoading(false); }
  };
  return (<section><h2>A06 — Blackboard</h2>
    <input value={task} onChange={e => setTask(e.target.value)} />
    <button onClick={run} disabled={loading}>{loading ? "…" : "Fusionner"}</button>
    <div className="result"><pre>{out.answer || "—"}</pre></div>
  </section>);
}
