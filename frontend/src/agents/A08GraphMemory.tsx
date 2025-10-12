import { useState } from "react";
export default function A08GraphMemory() {
  const [text, setText] = useState("Paris est la capitale de la France. La France est en Europe.");
  const [question, setQuestion] = useState("Quelle est la capitale de la France ?");
  const [out, setOut] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const run = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/a08/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, question }) }).then(r => r.json());
      setOut(r);
    } catch (e: any) { setOut({ answer: `Erreur: ${e.message}` }); }
    finally { setLoading(false); }
  };
  return (<section><h2>A08 — Graph Memory</h2>
    <textarea rows={4} value={text} onChange={e => setText(e.target.value)} />
    <input value={question} onChange={e => setQuestion(e.target.value)} />
    <button onClick={run} disabled={loading}>{loading ? "…" : "Construire + Répondre"}</button>
    <div className="result"><h3>Réponse</h3><pre>{out.answer || "—"}</pre></div>
    {out.graph && <div className="result"><h4>Relations</h4><pre>{JSON.stringify(out.graph, null, 2)}</pre></div>}
  </section>);
}
