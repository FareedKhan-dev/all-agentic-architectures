import { useEffect, useState } from "react";

type ProviderCfg = {
  llm: { provider: string; model: string; };
  embeddings: { provider: string; model: string; };
};

export default function ProviderToggle() {
  const [cfg, setCfg] = useState<ProviderCfg>({
    llm: { provider: "", model: "" },
    embeddings: { provider: "", model: "" },
  });
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const load = async () => {
    try {
      const h = await fetch("/healthz").then((r) => r.json());
      setCfg({
        llm: {
          provider: h.env?.DEFAULT_LLM_PROVIDER || "",
          model: h.env?.DEFAULT_LLM_MODEL || "",
        },
        embeddings: {
          provider: h.env?.DEFAULT_EMBEDDING_PROVIDER || "",
          model: h.env?.DEFAULT_EMBEDDING_MODEL || "",
        },
      });
    } catch (e: any) {
      setMsg(`Erreur healthz: ${e.message}`);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const apply = async () => {
    setLoading(true);
    setMsg("");
    try {
      await fetch("/api/config/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      }).then((r) => r.json());
      setMsg("✅ Appliqué. (In-memory) — Les prochains appels utiliseront ces valeurs.");
    } catch (e: any) {
      setMsg(`❌ Erreur: ${e.message}`);
    } finally {
      setLoading(false);
      load();
    }
  };

  return (
    <div className="provider-toggle">
      <h3>Provider Toggle (in-memory)</h3>
      <div className="row">
        <div>
          <label>LLM Provider</label>
          <select
            value={cfg.llm.provider}
            onChange={(e) =>
              setCfg((c) => ({ ...c, llm: { ...c.llm, provider: e.target.value } }))
            }
          >
            <option value="">(auto)</option>
            <option value="openai">openai</option>
            <option value="gemini">gemini</option>
          </select>
        </div>
        <div>
          <label>LLM Model</label>
          <input
            value={cfg.llm.model}
            onChange={(e) =>
              setCfg((c) => ({ ...c, llm: { ...c.llm, model: e.target.value } }))
            }
            placeholder="ex: gpt-4o-mini ou gemini-1.5-pro"
          />
        </div>
      </div>
      <div className="row">
        <div>
          <label>Embeddings Provider</label>
          <select
            value={cfg.embeddings.provider}
            onChange={(e) =>
              setCfg((c) => ({
                ...c,
                embeddings: { ...c.embeddings, provider: e.target.value },
              }))
            }
          >
            <option value="">(auto)</option>
            <option value="openai">openai</option>
            <option value="gemini">gemini</option>
          </select>
        </div>
        <div>
          <label>Embeddings Model</label>
          <input
            value={cfg.embeddings.model}
            onChange={(e) =>
              setCfg((c) => ({
                ...c,
                embeddings: { ...c.embeddings, model: e.target.value },
              }))
            }
            placeholder="ex: text-embedding-3-large ou text-embedding-004"
          />
        </div>
      </div>
      <button onClick={apply} disabled={loading}>
        {loading ? "…" : "Appliquer"}
      </button>
      {msg && <p className="msg">{msg}</p>}
    </div>
  );
}
