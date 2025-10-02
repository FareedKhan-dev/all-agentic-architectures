import { useState } from "react";

export default function A09ToT() {
  const [task, setTask] = useState("Résumer un article de presse en identifiant 3 angles différents.");
  const [breadth, setBreadth] = useState(3);
  const [depth, setDepth] = useState(2);
  const [out, setOut] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/a09/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task, breadth, depth }) }).then(r => r.json());
      setOut(r);
    } catch (e: any) { setOut({ answer: `Erreur: ${e.message}` }); }
    finally { setLoading(false); }
  };
  return (<section>
    <h2>A09 — Tree-of-Thoughts</h2>
    <label>Tâche</label>
    <input value={task} onChange={e => setTask(e.target.value)} />
    <div style={{ display: "flex", gap: 8 }}>
      <label>breadth <input type="number" min={2} max={5} value={breadth} onChange={e => setBreadth(parseInt(e.target.value || "3", 10))} /></label>
      <label>depth <input type="number" min={1} max={3} value={depth} onChange={e => setDepth(parseInt(e.target.value || "2", 10))} /></label>
    </div>
    <button onClick={run} disabled={loading}>{loading ? "…" : "Explorer"}</button>
    <div className="result"><h3>Réponse</h3><pre>{out.answer || "—"}</pre></div>
    {out.best && <div className="result"><h4>Meilleure piste</h4><pre>{JSON.stringify(out.best, null, 2)}</pre></div>}
    {out.trace && <div className="result"><h4>Trace</h4><pre>{JSON.stringify(out.trace, null, 2)}</pre></div>}
  </section>);
}
