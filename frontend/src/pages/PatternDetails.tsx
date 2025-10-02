import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { fetchPattern, runDemo } from "../api/patterns";
import type { PatternMeta } from "../types/patterns";

export default function PatternDetails({ id, onBack }: { id: string; onBack: () => void; }) {
  const [m, setM] = useState<PatternMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [demoOut, setDemoOut] = useState<any>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    setLoading(true);
    fetchPattern(id).then(setM).catch(e => setErr(String(e))).finally(() => setLoading(false));
  }, [id]);

  const onRun = async () => {
    if (!m?.has_demo) return;
    setErr(""); setDemoOut(null);
    try {
      const input = (m.id_pattern === "tool_use")
        ? { question: "Quelles ont été les annonces majeures de la dernière WWDC ?" }
        : (m.id_pattern === "reflection")
          ? { prompt: "Explique la loi des grands nombres simplement.", criteria: ["justesse", "clarté"] }
          : {};
      const out = await runDemo(m.id_pattern, input);
      setDemoOut(out);
    } catch (e: any) {
      setErr(e.message || String(e));
    }
  };

  if (loading) return <div>Chargement…</div>;
  if (err) return <div style={{ color: "crimson" }}>Erreur: {err}</div>;
  if (!m) return <div>Introuvable.</div>;

  return (
    <div>
      <button onClick={onBack}>← Retour</button>
      <h2 style={{ marginTop: 8 }}>{m.nom_fr}</h2>
      <p><b>Catégorie:</b> {m.categorie}</p>
      <p>{m.fonctionnement_court}</p>

      <section>
        <h3>Utilité (cas concrets)</h3>
        <ul>{m.utilite_concrets.map((u, i) => <li key={i}>{u}</li>)}</ul>
      </section>

      <section>
        <h3>Détails techniques</h3>
        <ReactMarkdown>{m.detail_technique || "_(à compléter)_"}</ReactMarkdown>
      </section>
      <section>
        <h3>Implications backend</h3>
        <ReactMarkdown>{m.implication_backend || "_(à compléter)_"}</ReactMarkdown>
      </section>
      <section>
        <h3>Compromis coût/latence</h3>
        <ReactMarkdown>{m.compromis_cout_latence || "_(à compléter)_"}</ReactMarkdown>
      </section>

      <section>
        <h3>Tags</h3>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {m.tags.map(t => <code key={t} style={{ background: "#f3f3f3", padding: "1px 6px", borderRadius: 8 }}>{t}</code>)}
        </div>
      </section>

      <section style={{ marginTop: 16 }}>
        <h3>Demo</h3>
        <button onClick={onRun} disabled={!m.has_demo} title={m.has_demo ? "Lancer la démo" : "Non implémenté"}>
          {m.has_demo ? "▶︎ Run demo" : "—"}
        </button>
        {err && <p style={{ color: "crimson" }}>{err}</p>}
        {demoOut && (
          <pre style={{ marginTop: 12, whiteSpace: "pre-wrap" }}>{JSON.stringify(demoOut, null, 2)}</pre>
        )}
      </section>
    </div>
  );
}
