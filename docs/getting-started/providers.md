# Switching LLM providers

Every architecture in this library is provider-agnostic. The same `Reflection` class works against Nebius, OpenAI, Anthropic Claude, DeepSeek, Groq, Ollama, Together, Fireworks, Mistral, Google Gemini or OpenRouter: you change one config setting.

## The `get_llm()` factory

```python
from agentic_architectures import get_llm

# Use whatever's in .env / settings
llm = get_llm()

# Override per-call; leave out the model to get that provider's recommended one
llm = get_llm(provider="openai", temperature=0.4)
llm = get_llm(provider="anthropic", model="claude-haiku-5-5")
llm = get_llm(provider="ollama", model="qwen3:14b")
```

Pass the returned LLM into any architecture:

```python
from agentic_architectures.architectures import Reflection
arch = Reflection(llm=llm)
```

`LLM_MODEL` belongs to `LLM_PROVIDER`. Asking for another provider without naming a model gives that provider's recommended model, never the configured provider's model name.

## Provider compatibility matrix

The recommended models were checked against each provider's own model and deprecation pages in November 2026.

| Provider | Extra to install | Env var | Recommended model | Notes |
|---|---|---|---|---|
| **Nebius** (default) | `[nebius]` | `NEBIUS_API_KEY` | `nvidia/Nemotron-3_5-Lightning` | Nebius' replacement for Llama 3.3 70B, which left its serverless API on Aug 31, 2026. |
| **OpenAI** | `[openai]` | `OPENAI_API_KEY` | `gpt-6-luna` | Calls tools on Chat Completions only with `reasoning_effort="none"`, which `get_llm()` sets for this model. |
| **Anthropic** | `[anthropic]` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | Strong on long context and tool use. `claude-haiku-5-5` is the cheaper choice. |
| **DeepSeek** | `[deepseek]` | `DEEPSEEK_API_KEY` | `deepseek-flash` | Through `langchain-deepseek`. `deepseek-chat` and `deepseek-reasoner` were retired on Jul 24, 2026. |
| **Groq** | `[groq]` | `GROQ_API_KEY` | `openai/gpt-oss-120b` | Fast inference; good for high-volume Self-Consistency (nb 21). |
| **Ollama** | `[ollama]` | none | `qwen3:8b` | Fully local, no API key. `qwen3:14b` is stronger if it fits your GPU. |
| **Together** | `[together]` | `TOGETHER_API_KEY` | `openai/gpt-oss-120b` | Wide model catalogue. |
| **Fireworks** | `[fireworks]` | `FIREWORKS_API_KEY` | `accounts/fireworks/models/glm-5p3-flash` | Function calling first-class. |
| **Mistral** | `[mistral]` | `MISTRAL_API_KEY` | `mistral-large-latest` | EU-hosted option. The alias points to Mistral Large 4. |
| **Google** | `[google]` | `GOOGLE_API_KEY` | `gemini-3.5-flash-lite` | Free tier available. |
| **OpenRouter** | `[openrouter]` | `OPENROUTER_API_KEY` | `openai/gpt-oss-120b` | One key for hundreds of models, MiniMax (`minimax/minimax-m3.1-flash`) and Qwen among them. |

Providers retire models often: six of the defaults this library shipped with in May 2026 had been retired or deprecated by November. If a recommended model stops answering, set `LLM_MODEL` to the replacement the provider's deprecation page names.

### Any OpenAI-compatible server

vLLM, LM Studio, llama.cpp's server and most hosted APIs speak the OpenAI protocol. Point the `openai` provider at one with `OPENAI_BASE_URL`:

```ini
# e.g. a local vLLM server: vllm serve Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes
LLM_PROVIDER=openai
LLM_MODEL=Qwen/Qwen3-14B
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=unused
```

Provider-specific extras, such as replaying a reasoning model's thinking across tool turns, need that provider's own LangChain integration.

## Environment configuration

Settings load from `.env` at the repo root (start from `.env.example`). Minimum:

```ini
# .env
LLM_PROVIDER=nebius
LLM_MODEL=                         # empty: the provider's recommended model
LLM_TEMPERATURE=0.2

NEBIUS_API_KEY=...
SEARCH_BACKEND=mcp                 # web search with no key, see below
LANGSMITH_API_KEY=ls-...           # optional, for tracing
LANGSMITH_PROJECT=all-agentic-architectures
```

To switch globally, change `LLM_PROVIDER` and the corresponding key. No code changes needed. An empty value in `.env` means "not set".

## Web search without a key

The architectures that search (ToolUse, ReAct, Planning, PEV, Multi-Agent, ...) use Tavily by default, which needs `TAVILY_API_KEY`. With `SEARCH_BACKEND=mcp` they use the search tool of the MCP server at `SEARCH_MCP_URL` instead: Parallel's free, keyless search server by default, so they run without signing up anywhere.

```ini
SEARCH_BACKEND=mcp
SEARCH_MCP_URL=https://search.parallel.ai/mcp
```

Any MCP server's tools can be handed to an architecture directly, too:

```python
from agentic_architectures.architectures import ReAct
from agentic_architectures.tools import mcp_tools

ReAct(tools=mcp_tools("https://search.parallel.ai/mcp"))
```

Both need `pip install agentic-architectures[mcp]`.

## When you'd override per-architecture

The reasoning notebooks (nb 03, 04, 09, 19, 20, 22) use `LLM_REASONING_MODEL` where a reasoning ("thinking") model makes the difference. It defaults to `LLM_MODEL`; set it to a thinking model to see the contrast:

```ini
LLM_REASONING_MODEL=deepseek-v4-pro
```

Or force a single architecture onto a different model in code:

```python
from agentic_architectures.architectures import ChainOfVerification
arch = ChainOfVerification(llm=get_llm(provider="anthropic", model="claude-opus-5-5"))
```

## Capability matrix

| Feature | Required for | Providers known good |
|---|---|---|
| `with_structured_output(Schema)` | nb 04, 06, 09, 10, 13, 15, 17-36 | All listed above |
| `bind_tools([...])` | nb 02, 03, 05, 11 | All except Ollama models without tool support |
| `bind(temperature=...)` per-call override | nb 21 Self-Consistency | All |

If an architecture fails on a provider, the failure is usually a missing capability: see the notebook's §10 "Try other providers" cell for graceful-skip patterns.

## Local models with Ollama

`get_llm(provider="ollama")` asks Ollama for a 16,384-token context window. Ollama's own default is 4,096 tokens, and it silently drops the start of a longer prompt, which the RAG and multi-agent prompts here exceed. Pass `num_ctx=` to choose another size.
