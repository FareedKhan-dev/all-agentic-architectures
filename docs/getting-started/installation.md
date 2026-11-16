# Installation

```bash
pip install agentic-architectures
```

The base install gets you the `Architecture` base class, every architecture's Python class, and the LangGraph/LangChain runtime. For LLM providers, vector stores, and tools you pick the extras you want:

=== "Nebius (default, recommended)"

    ```bash
    pip install "agentic-architectures[nebius,mcp]"
    ```

    Then set `NEBIUS_API_KEY` in `.env`. With `LLM_MODEL` empty it uses Nebius' recommended model, `nvidia/Nemotron-3_5-Lightning`; set `LLM_REASONING_MODEL` to a thinking model for the reasoning notebooks (nb 03, 04, 09, 19, 20, 22).

=== "OpenAI"

    ```bash
    pip install "agentic-architectures[openai,mcp]"
    ```

    Set `OPENAI_API_KEY` and `LLM_PROVIDER=openai`. With `LLM_MODEL` empty it uses `gpt-6-luna`.

=== "Anthropic"

    ```bash
    pip install "agentic-architectures[anthropic,mcp]"
    ```

    Set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic`.

=== "Ollama (local)"

    ```bash
    pip install "agentic-architectures[ollama,mcp]"
    ```

    Run `ollama pull qwen3:8b` (or `qwen3:14b` if it fits your GPU) and `ollama pull nomic-embed-text` first. Set `LLM_PROVIDER=ollama`. No API key needed.

=== "Everything"

    ```bash
    pip install "agentic-architectures[all]"
    ```

    All provider integrations + all memory backends + all tools. ~1.5 GB.

## Memory backend extras

| Extra | What it gets you | Use when |
|---|---|---|
| (none, the default) | langchain-core's `InMemoryVectorStore` | Local in-process vector store, `VECTOR_BACKEND=memory`. Used by nb 08, 18, 23-26, 29, 31, 35. |
| `faiss` | FAISS-CPU + langchain-community | `VECTOR_BACKEND=faiss`. langchain-community is being sunset by LangChain. |
| `chroma` | Chroma + chromadb | When you want persistence and an HTTP server. |
| `qdrant` | Qdrant + qdrant-client | Production-grade, persistent, multi-tenant. |
| `neo4j` | langchain-neo4j + neo4j driver | For nb 08, 12, 27: graph queries via Cypher. |
| `networkx` | NetworkX (in-process graph) | Default graph backend; no extra service to run. |

## Tool extras

| Extra | Used by |
|---|---|
| `tavily` | Web search with a Tavily key: nb 02, 03, 04, 05, 07, 11, 24, 30 |
| `mcp` | Web search with no key (`SEARCH_BACKEND=mcp`), and `mcp_tools()` for any MCP server |

## Browser-using architecture (nb 34)

Notebook 34 (`BrowserAgent`) drives a real headless Chromium. After the base install:

```bash
pip install "agentic-architectures[browser]"
playwright install chromium
```

`playwright install chromium` downloads ~200 MB of browser binaries. Run once per machine.

## Verify your install

```bash
python -c "from agentic_architectures import get_llm, settings; print(settings.llm_provider, settings.llm_model)"
# nebius nvidia/Nemotron-3_5-Lightning
```

If you see your configured provider and model, you're ready. Continue to the [Quickstart](quickstart.md).
