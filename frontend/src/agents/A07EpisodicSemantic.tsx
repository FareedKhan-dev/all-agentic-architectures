import { useState } from "react";

export default function A07EpisodicSemantic() {
  const [question, setQuestion] = useState("Explique la loi des grands nombres et donne des exemples.");
  const [out, setOut] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/a07/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) }).then(r => r.json());
      setOut(r);
    } catch (e: any) { setOut({ answer: `Erreur: ${e.message}` }); }
    finally { setLoading(false); }
  };
  return (
    <section>
      <h2>A07 — Episodic + Semantic</h2>
      <input value={question} onChange={e => setQuestion(e.target.value)} />
      <button onClick={run} disabled={loading}>{loading ? "…" : "Répondre"}</button>
      <div className="result"><h3>Réponse</h3><pre>{out.answer || "—"}</pre></div>
      {out.episodic && <div className="result"><h4>Episodic</h4><pre>{JSON.stringify(out.episodic, null, 2)}</pre></div>}
      {out.semantic && <div className="result"><h4>Semantic</h4><pre>{JSON.stringify(out.semantic, null, 2)}</pre></div>}
    </section>
  );
}
