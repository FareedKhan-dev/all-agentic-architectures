# Switching LLM providers

Every architecture in this library is provider-agnostic. The same `Reflection` class works against Nebius Llama, OpenAI GPT-4o, Anthropic Claude, Groq, Ollama, Together, Fireworks, Mistral, Google Gemini, or MiniMax — you change one config setting.

## The `get_llm()` factory

```python
from agentic_architectures import get_llm

# Use whatever's in .env / settings
llm = get_llm()

# Override per-call
llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0.4)
llm = get_llm(provider="anthropic", model="claude-haiku-4-5-20251001")
llm = get_llm(provider="minimax", model="MiniMax-M3")
llm = get_llm(provider="minimax", model="MiniMax-M2.7")
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
| **MiniMax** | `[minimax]` | `MINIMAX_API_KEY` | MiniMax-M3 and MiniMax-M2.7 through OpenAI- or Anthropic-compatible APIs. |

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

### MiniMax endpoints and models

`MINIMAX_BASE_URL` selects both the region and protocol. The factory uses the
Anthropic client when the URL ends in `/anthropic`; all other values use the
OpenAI client.

| Region | Protocol | `MINIMAX_BASE_URL` |
|---|---|---|
| Global | OpenAI | `https://api.minimax.io/v1` |
| Global | Anthropic | `https://api.minimax.io/anthropic` |
| China | OpenAI | `https://api.minimaxi.com/v1` |
| China | Anthropic | `https://api.minimaxi.com/anthropic` |

Keep the Anthropic base URL ending at `/anthropic`; the client appends
`/v1/messages` when it sends a request.

| Model | Total context window | Input modalities | Thinking |
|---|---:|---|---|
| `MiniMax-M3` | 1,000,000 tokens | Text, image, video | `adaptive` or `disabled`; the protocol controls the default |
| `MiniMax-M2.7` | 204,800 tokens | Text | Always on |

Pay-as-you-go prices below are in USD per million tokens.

| Model | Input | Output | Cache read | Cache write |
|---|---:|---:|---:|---:|
| `MiniMax-M3` | $0.60 | $2.40 | $0.12 | N/A |
| `MiniMax-M2.7` | $0.30 | $1.20 | $0.06 | $0.375 |

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
