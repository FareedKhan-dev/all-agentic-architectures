# Installation

```bash
pip install agentic-architectures
```

The base install gets you the `Architecture` base class, every architecture's Python class, and the LangGraph/LangChain runtime. For LLM providers, vector stores, and tools you pick the extras you want:

=== "Nebius (default, recommended)"

    ```bash
    pip install "agentic-architectures[nebius,faiss,tavily]"
    ```

    Then set `NEBIUS_API_KEY` in `.env`. Pre-configured to use `meta-llama/Llama-3.3-70B-Instruct` for general work and `Qwen/Qwen3-235B-A22B-Thinking-2507-fast` for reasoning architectures (nb 19, 20, 21, 22).

=== "OpenAI"

    ```bash
    pip install "agentic-architectures[openai,faiss,tavily]"
    ```

    Set `OPENAI_API_KEY` and `LLM_PROVIDER=openai` (with `LLM_MODEL=gpt-4o-mini` or similar).

=== "Anthropic"

    ```bash
    pip install "agentic-architectures[anthropic,faiss,tavily]"
    ```

    Set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic`.

=== "Ollama (local)"

    ```bash
    pip install "agentic-architectures[ollama,faiss,tavily]"
    ```

    Run `ollama pull llama3.3:70b` first. Set `LLM_PROVIDER=ollama`. No API key needed.

=== "Everything"

    ```bash
    pip install "agentic-architectures[all]"
    ```

    All provider integrations + all memory backends + all tools. ~1.5 GB.

## Memory backend extras

| Extra | What it gets you | Use when |
|---|---|---|
| `faiss` (recommended) | FAISS-CPU + langchain-community | Local in-process vector store. Used by nb 08, 18, 23-26, 29, 31, 35. |
| `chroma` | Chroma + chromadb | When you want persistence and an HTTP server. |
| `qdrant` | Qdrant + qdrant-client | Production-grade, persistent, multi-tenant. |
| `neo4j` | langchain-neo4j + neo4j driver | For nb 08, 12, 27 — graph queries via Cypher. |
| `networkx` | NetworkX (in-process graph) | Default graph backend; no extra service to run. |

## Tool extras

| Extra | Used by |
|---|---|
| `tavily` | Web search — nb 02, 03, 04, 05, 07, 11, 24, 30 |

## Browser-using architecture (nb 34)

Notebook 34 (`BrowserAgent`) drives a real headless Chromium. After the base install:

```bash
pip install playwright
playwright install chromium
```

`playwright install chromium` downloads ~200 MB of browser binaries. Run once per machine.

## Verify your install

```bash
python -c "from agentic_architectures import get_llm, settings; print(settings.llm_provider, settings.llm_model)"
# nebius meta-llama/Llama-3.3-70B-Instruct
```

If you see your configured provider and model, you're ready. Continue to the [Quickstart](quickstart.md).
