import "./gallery.css";

type Card = { code: string; title: string; desc: string; };

const CARDS: Card[] = [
  { code: "a01", title: "A01 — Reflection", desc: "Brouillon → Critique → Réécriture" },
  { code: "a02", title: "A02 — Tool Use", desc: "Outils externes (web search…)" },
  { code: "a03", title: "A03 — ReAct", desc: "Reason/Act en boucle" },
  { code: "a04", title: "A04 — Planning", desc: "Plan en étapes + synthèse" },
  { code: "a05", title: "A05 — PEV", desc: "Planner–Executor–Verifier" },
  { code: "a06", title: "A06 — Blackboard", desc: "Mémoire partagée multi-agents" },
  { code: "a07", title: "A07 — Episodic+Semantic", desc: "Mémoire court/long terme (RAG)" },
  { code: "a08", title: "A08 — Graph Memory", desc: "Monde/relations (graphe)" },
  { code: "a09", title: "A09 — Tree-of-Thoughts", desc: "Largeur/profondeur, scoring" },
  { code: "a10", title: "A10 — Ensemble Decision", desc: "Votes/agrégation multi-agents" },
  { code: "a11", title: "A11 — Meta-Controller", desc: "Router vers RAG/Tool/Reflection" },
  { code: "a12", title: "A12 — Self-Ask", desc: "Auto-questionnement" },
  { code: "a13", title: "A13 — Chain-of-Verification", desc: "Checklist & corrections" },
  { code: "a14", title: "A14 — Dry-Run", desc: "Jeu de tests / report" },
  { code: "a15", title: "A15 — RLHF-like Loop", desc: "Answer→Critique→Refine + score" },
  { code: "a16", title: "A16 — Simulator (MITL)", desc: "Monde mental + feedback" },
  { code: "a17", title: "A17 — Reflexive Metacognitive", desc: "Incertitudes & unknowns" },
];

export default function Gallery17({ onSelect }: { onSelect: (code: string) => void; }) {
  return (
    <div className="gallery">
      {CARDS.map((c) => (
        <button className="card" key={c.code} onClick={() => onSelect(c.code)}>
          <div className="code">{c.title}</div>
          <div className="desc">{c.desc}</div>
        </button>
      ))}
    </div>
  );
}
