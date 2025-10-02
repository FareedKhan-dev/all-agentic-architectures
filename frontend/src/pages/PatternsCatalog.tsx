import { useEffect, useMemo, useState } from "react";
import { fetchPatterns } from "../api/patterns";
import type { PatternMeta } from "../types/patterns";
import "./gallery.css";

const CATS = ["Tous", "Raisonnement", "Action", "Amélioration", "Orchestration"] as const;

export default function PatternsCatalog({ onOpen }: { onOpen: (id: string) => void; }) {
  const [items, setItems] = useState<PatternMeta[]>([]);
  const [cat, setCat] = useState<typeof CATS[number]>("Tous");
  const [q, setQ] = useState("");

  useEffect(() => {
    fetchPatterns().then(setItems).catch(e => console.error(e));
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    return items.filter(it => {
      const catOk = (cat === "Tous") || (it.categorie === cat);
      const qOk = !s || (
        it.nom_fr.toLowerCase().includes(s) ||
        it.fonctionnement_court.toLowerCase().includes(s) ||
        it.tags.join(" ").toLowerCase().includes(s)
      );
      return catOk && qOk;
    });
  }, [items, cat, q]);

  return (
    <div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "8px 0" }}>
        <select value={cat} onChange={e => setCat(e.target.value as any)}>
          {CATS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Recherche titre/tags…" style={{ flex: 1 }} />
        <small>{filtered.length} / {items.length}</small>
      </div>
      <div className="gallery">
        {filtered.map(it => (
          <button className="card" key={it.id_pattern} onClick={() => onOpen(it.id_pattern)} title={it.fonctionnement_court}>
            <div className="code" style={{ display: "flex", justifyContent: "space-between" }}>
              <span>{it.nom_fr}</span>
              <span style={{ fontSize: "0.8rem", opacity: 0.7 }}>{it.categorie}</span>
            </div>
            <div className="desc">{it.fonctionnement_court}</div>
            <div style={{ marginTop: 6, display: "flex", gap: 6, flexWrap: "wrap" }}>
              {it.tags.slice(0, 4).map(t => <code key={t} style={{ background: "#f3f3f3", padding: "1px 6px", borderRadius: 8 }}>{t}</code>)}
            </div>
            <div style={{ marginTop: 6, fontSize: "0.85rem", opacity: 0.8 }}>
              {it.has_demo ? "✅ Demo" : "—"}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
