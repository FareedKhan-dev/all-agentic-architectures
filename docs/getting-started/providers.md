# Switching LLM providers

Every architecture in this library is provider-agnostic. The same `Reflection` class works against Nebius Llama, OpenAI GPT-4o, Anthropic Claude, Groq, Ollama, Together, Fireworks, Mistral, or Google Gemini — you change one config setting.

## The `get_llm()` factory

```python
from agentic_architectures import get_llm

# Use whatever's in .env / settings
llm = get_llm()

# Override per-call
llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0.4)
llm = get_llm(provider="anthropic", model="claude-haiku-4-5-20251001")
llm = get_llm(provider="ollama", model="llama3.3:70b")
```

Pass the returned LLM into any architecture:

```python
from agentic_architectures.architectures import Reflection
arch = Reflection(llm=llm)
```

## Provider compatibility matrix

| Provider | Extra to install | Env var | Notes |
|---|---|---|---|
| **Nebius** (default) | `[nebius]` | `NEBIUS_API_KEY` | Llama-3.3-70B + Qwen3-Thinking. Cheapest for the included demos. |
| **OpenAI** | `[openai]` | `OPENAI_API_KEY` | All architectures work. Highest quality for reasoning architectures. |
| **Anthropic** | `[anthropic]` | `ANTHROPIC_API_KEY` | Strong on long context. Required for true Computer-Use (nb 34) production. |
| **Groq** | `[groq]` | `GROQ_API_KEY` | Fast inference; good for high-volume Self-Consistency (nb 21). |
| **Ollama** | `[ollama]` | — | Fully local; no API key. Tool calling depends on model — Llama-3.3 instruct works. |
| **Together** | `[together]` | `TOGETHER_API_KEY` | Wide model catalogue. |
| **Fireworks** | `[fireworks]` | `FIREWORKS_API_KEY` | Function-calling first-class. |
| **Mistral** | `[mistral]` | `MISTRAL_API_KEY` | EU-hosted option. |
| **Google** | `[google]` | `GOOGLE_API_KEY` | Gemini-2.x via Generative AI API. |

## Environment configuration

Settings load from `.env` at the repo root. Minimum:

```ini
# .env
LLM_PROVIDER=nebius
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
LLM_TEMPERATURE=0.2

NEBIUS_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=ls-...           # optional, for tracing
LANGSMITH_PROJECT=all-agentic-architectures
```

To switch globally, change `LLM_PROVIDER` + the corresponding key. No code changes needed.

## When you'd override per-architecture

Reasoning architectures (nb 19, 20, 21, 22) default to Qwen3-Thinking for higher quality on multi-step structured reasoning. If you want them on a cheaper model:

```python
from agentic_architectures.architectures import LATS
arch = LATS(llm=get_llm(model="meta-llama/Llama-3.3-70B-Instruct"))
```

Or the opposite — force a single architecture onto a stronger model:

```python
from agentic_architectures.architectures import ChainOfVerification
arch = ChainOfVerification(llm=get_llm(provider="openai", model="gpt-4o"))
```

## Capability matrix

| Feature | Required for | Providers known good |
|---|---|---|
| `with_structured_output(Schema)` | nb 04, 06, 09, 10, 13, 15, 17-35 | All listed above |
| `bind_tools([...])` | nb 02, 03, 05, 11 | All except some Ollama tool-uncapable models |
| `bind(temperature=...)` per-call override | nb 21 Self-Consistency | All |

If an architecture fails on a provider, the failure is usually a missing capability — see the notebook's §10 "Try other providers" cell for graceful-skip patterns.
