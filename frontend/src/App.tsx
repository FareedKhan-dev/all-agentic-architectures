import { useEffect, useState } from "react";
import A04Planning from "./agents/A04Planning";
import A05PEV from "./agents/A05PEV";
import A06Blackboard from "./agents/A06Blackboard";
import A07EpisodicSemantic from "./agents/A07EpisodicSemantic";
import A08GraphMemory from "./agents/A08GraphMemory";
import A09ToT from "./agents/A09ToT";
import { API } from "./api";
import "./App.css";
import ProviderToggle from "./components/ProviderToggle";
import Gallery17 from "./pages/Gallery17";
import PatternDetails from "./pages/PatternDetails";
import PatternsCatalog from "./pages/PatternsCatalog";

type Tab = "gallery" | "catalog" | "a01" | "rag" | "a02" | "a04" | "a05" | "a06" | "a07" | "a08" | "a09";

function App() {
  const [tab, setTab] = useState<Tab>("gallery");
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);
  const [health, setHealth] = useState<string>("(en attente…)");
  useEffect(() => {
    API.health()
      .then((h) => setHealth(`ok=${h.ok} | provider=${h.env?.DEFAULT_LLM_PROVIDER} | embed=${h.env?.DEFAULT_EMBEDDING_PROVIDER}`))
      .catch((e) => setHealth(`Erreur healthz: ${e}`));
  }, []);

  return (
    <div className="container">
      <header>
        <h1>All Agentic — Démo locale</h1>
        <p className="health">Health: {health}</p>
        <nav>
          <button className={tab === "gallery" ? "active" : ""} onClick={() => setTab("gallery")}>🏠 Gallery 17</button>
          <button className={tab === "catalog" ? "active" : ""} onClick={() => { setSelectedPattern(null); setTab("catalog"); }}>📚 Catalogue</button>
          <button className={tab === "a01" ? "active" : ""} onClick={() => setTab("a01")}>A01 — Reflection</button>
          <button className={tab === "rag" ? "active" : ""} onClick={() => setTab("rag")}>RAG — QA</button>
          <button className={tab === "a02" ? "active" : ""} onClick={() => setTab("a02")}>A02 — Tool Use</button>
          <button className={tab === "a04" ? "active" : ""} onClick={() => setTab("a04")}>A04 — Planning</button>
          <button className={tab === "a05" ? "active" : ""} onClick={() => setTab("a05")}>A05 — PEV</button>
          <button className={tab === "a06" ? "active" : ""} onClick={() => setTab("a06")}>A06 — Blackboard</button>
          <button className={tab === "a07" ? "active" : ""} onClick={() => setTab("a07")}>A07 — Episodic+Semantic</button>
          <button className={tab === "a08" ? "active" : ""} onClick={() => setTab("a08")}>A08 — Graph Memory</button>
          <button className={tab === "a09" ? "active" : ""} onClick={() => setTab("a09")}>A09 — ToT</button>
        </nav>
      </header>
      <main>
        <ProviderToggle />
        {tab === "gallery" && <Gallery17 onSelect={(code) => setTab(code as Tab)} />}
        {tab === "catalog" && (
          selectedPattern ? (
            <PatternDetails id={selectedPattern} onBack={() => setSelectedPattern(null)} />
          ) : (
            <PatternsCatalog onOpen={(id) => setSelectedPattern(id)} />
          )
        )}
        {tab === "a01" && <A01 />}
        {tab === "rag" && <RAG />}
        {tab === "a02" && <A02 />}
        {tab === "a04" && <A04Planning />}
        {tab === "a05" && <A05PEV />}
        {tab === "a06" && <A06Blackboard />}
        {tab === "a07" && <A07EpisodicSemantic />}
        {tab === "a08" && <A08GraphMemory />}
        {tab === "a09" && <A09ToT />}
      </main>
      <footer>
        <small>Même origine que l’API (StaticFiles) — conditions proches prod.</small>
      </footer>
    </div>
  );
}

function A01() {
  const [prompt, setPrompt] = useState("Explique la loi des grands nombres simplement.");
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState<{ answer?: string; critic?: string; }>({});
  const run = async () => {
    setLoading(true);
    try {
      const res = await API.a01({ prompt });
      setOut(res);
    } catch (e: any) {
      setOut({ answer: `Erreur: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };
  return (
    <section>
      <h2>Reflection (A01)</h2>
      <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} />
      <button onClick={run} disabled={loading}>{loading ? "…" : "Lancer"}</button>
      <div className="result">
        <h3>Réponse</h3>
        <pre>{out.answer || "—"}</pre>
        {out.critic && <>
          <h4>Critique</h4>
          <pre>{out.critic}</pre>
        </>}
      </div>
    </section>
  );
}

function RAG() {
  const [question, setQuestion] = useState("Quels thèmes sont évoqués dans la démo RAG ?");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState<{ answer?: string; sources?: string[]; }>({});
  const run = async () => {
    setLoading(true);
    try {
      const res = await API.rag({ question, k });
      setOut(res);
    } catch (e: any) {
      setOut({ answer: `Erreur: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };
  return (
    <section>
      <h2>RAG — QA</h2>
      <input value={question} onChange={e => setQuestion(e.target.value)} />
      <label>k:
        <input type="number" value={k} min={1} max={10} onChange={e => setK(parseInt(e.target.value || "5", 10))} />
      </label>
      <button onClick={run} disabled={loading}>{loading ? "…" : "Interroger"}</button>
      <div className="result">
        <h3>Réponse</h3>
        <pre>{out.answer || "—"}</pre>
        {out.sources?.length ? <>
          <h4>Sources</h4>
          <ul>{out.sources.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </> : null}
      </div>
    </section>
  );
}

function A02() {
  const [question, setQuestion] = useState("Quelles ont été les annonces majeures de la dernière WWDC ?");
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState<{ answer?: string; sources?: string[]; }>({});
  const run = async () => {
    setLoading(true);
    try {
      const res = await API.a02({ question });
      setOut(res);
    } catch (e: any) {
      setOut({ answer: `Erreur: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };
  return (
    <section>
      <h2>Tool Use (A02)</h2>
      <input value={question} onChange={e => setQuestion(e.target.value)} />
      <button onClick={run} disabled={loading}>{loading ? "…" : "Rechercher"}</button>
      <div className="result">
        <h3>Réponse</h3>
        <pre>{out.answer || "—"}</pre>
        {out.sources?.length ? <>
          <h4>Sources</h4>
          <ul>{out.sources.map((s, i) => <li key={i}><a href={s} target="_blank" rel="noreferrer">{s}</a></li>)}</ul>
        </> : null}
      </div>
    </section>
  );
}

export default App;
