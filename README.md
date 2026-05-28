<div align="center">

<br>

# Agentic Architectures

### Thirty-five production-grade agentic AI patterns. End to end.

A library *and* a living textbook — real LLM outputs, provider-agnostic,
deterministic-picker discipline throughout, and a comparative benchmark
leaderboard that ranks every architecture against every relevant task.

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/FareedKhan-dev/all-agentic-architectures/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white&style=for-the-badge&color=0a0a0a)](https://github.com/FareedKhan-dev/all-agentic-architectures/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/FareedKhan-dev/all-agentic-architectures/docs.yml?branch=main&label=DOCS&logo=materialformkdocs&logoColor=white&style=for-the-badge&color=6366f1)](https://fareedkhan-dev.github.io/all-agentic-architectures/)
[![PyPI](https://img.shields.io/pypi/v/agentic-architectures?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=a855f7)](https://pypi.org/project/agentic-architectures/)
[![License](https://img.shields.io/badge/license-MIT-ec4899?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

<br>

<a href="https://fareedkhan-dev.github.io/all-agentic-architectures/getting-started/quickstart/">
  <img alt="Quickstart" src="https://img.shields.io/badge/Quickstart-→-0a0a0a?style=for-the-badge&labelColor=0a0a0a" />
</a>
<a href="https://fareedkhan-dev.github.io/all-agentic-architectures/">
  <img alt="Documentation" src="https://img.shields.io/badge/Documentation-→-262626?style=for-the-badge&labelColor=262626" />
</a>
<a href="https://fareedkhan-dev.github.io/all-agentic-architectures/architectures/">
  <img alt="Architectures" src="https://img.shields.io/badge/Architectures-→-404040?style=for-the-badge&labelColor=404040" />
</a>
<a href="https://fareedkhan-dev.github.io/all-agentic-architectures/benchmarks/">
  <img alt="Benchmarks" src="https://img.shields.io/badge/Benchmarks-→-525252?style=for-the-badge&labelColor=525252" />
</a>
<a href="https://codespaces.new/FareedKhan-dev/all-agentic-architectures">
  <img alt="Open in Codespaces" src="https://img.shields.io/badge/Open_in_Codespaces-→-737373?style=for-the-badge&labelColor=737373" />
</a>

<br><br>

<table>
<tr>
<td align="center" width="140"><h2><kbd>&nbsp; 35 &nbsp;</kbd></h2><sub>ARCHITECTURES</sub></td>
<td align="center" width="140"><h2><kbd>&nbsp; 283 &nbsp;</kbd></h2><sub>PASSING TESTS</sub></td>
<td align="center" width="140"><h2><kbd>&nbsp; 17 &nbsp;</kbd></h2><sub>BENCHMARK TASKS</sub></td>
<td align="center" width="140"><h2><kbd>&nbsp; 9 &nbsp;</kbd></h2><sub>LLM PROVIDERS</sub></td>
<td align="center" width="140"><h2><kbd>&nbsp; 0 &nbsp;</kbd></h2><sub>MOCKED RUNS</sub></td>
</tr>
</table>

<br>

</div>

---

## Overview

A single Python library that packages every major agentic AI pattern from the literature as a runnable `Architecture` class with a uniform contract. Each pattern ships with a fully executed Jupyter notebook whose theory is written *against* the captured run — not synthetic examples. The library is multi-provider (Nebius, OpenAI, Anthropic, Groq, Ollama, Together, Fireworks, Mistral, Google) and built on top of LangGraph state machines.

The central technical discipline of the repository is the **deterministic-picker pattern** — every LLM-as-Scorer surface has the LLM commit to categorical features (booleans, enums) and lets Python compose the deciding signal. This is the universal escape from the LLM-as-Scorer flat-band pathology, applied in 13 of 35 architectures; 9 more are architecturally immune by design.

---

## Quickstart

```bash
pip install "agentic-architectures[nebius,faiss,tavily]"
```

```python
from agentic_architectures import get_llm
from agentic_architectures.architectures import Reflection

arch = Reflection(llm=get_llm(), max_iterations=2, target_score=8)
result = arch.run("Write a haiku about a glacier.")

print(result.output)
print("score:", result.metadata["final_score"], "/ 10")
```

Same `.run(task)` interface across all 35 architectures. Same `ArchitectureResult` return shape. Swap the class, swap the pattern — your downstream code does not change.

<details>
<summary><b>Set up a virtualenv from a fresh clone</b></summary>

<br>

```bash
git clone https://github.com/FareedKhan-dev/all-agentic-architectures
cd all-agentic-architectures

python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # macOS / Linux

pip install -e ".[dev,test,docs,nebius,faiss,tavily,networkx]"
cp .env.example .env                # then fill in NEBIUS_API_KEY etc.

pytest -q                           # 283 tests pass in ~30s
```

</details>

---

## Architecture families

<table>
<tr>
<td width="33%" valign="top">

#### Reasoning & Reflection
Self-critique loops that drive answer quality up through iteration.
<br><br>
<sub>Reflection · Reflexion · Chain-of-Verification · Self-Discover · Constitutional AI</sub>

</td>
<td width="33%" valign="top">

#### Sampling & Search
Sample many paths or grow a tree with rewards.
<br><br>
<sub>Self-Consistency · Tree of Thoughts · LATS · Mental Loop · Ensemble</sub>

</td>
<td width="33%" valign="top">

#### Retrieval (RAG)
Ground every claim — five retrieval shapes.
<br><br>
<sub>Agentic RAG · Corrective RAG · Self-RAG · Adaptive RAG · GraphRAG</sub>

</td>
</tr>
<tr>
<td width="33%" valign="top">

#### Memory
Learn across calls — pick the storage shape.
<br><br>
<sub>Episodic + Semantic · Graph Memory · MemGPT · Voyager · Agent Workflow Memory</sub>

</td>
<td width="33%" valign="top">

#### Tools & Actions
From one search tool to a real Chromium browser.
<br><br>
<sub>Tool Use · ReAct · Planning · PEV · SWE-Agent · BrowserAgent</sub>

</td>
<td width="33%" valign="top">

#### Multi-Agent
Specialists, debate, multi-perspective research.
<br><br>
<sub>Multi-Agent · Blackboard · Debate · STORM · Meta-Controller</sub>

</td>
</tr>
<tr>
<td width="33%" valign="top">

#### Safety & Routing
Categorical actions through deterministic Python gates.
<br><br>
<sub>Dry-Run · Reflexive Metacognitive · Computer Use</sub>

</td>
<td width="33%" valign="top">

#### Specialty
Patterns with a unique shape.
<br><br>
<sub>RLHF Self-Improvement · Cellular Automata</sub>

</td>
<td width="33%" valign="top">

#### Cross-cutting
Patterns that appear across families.
<br><br>
<sub><a href="https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/deterministic-picker/">Deterministic-picker</a> · <a href="https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/memory/">Memory variants</a></sub>

</td>
</tr>
</table>

---

## The 35 architectures

<details open>
<summary><b>Reasoning &amp; Reflection</b></summary>

| Architecture | Pattern | Reference |
|---|---|---|
| **Reflection** | Generate → critique → refine | Madaan 2023 |
| **Reflexion** | Verbal reflections in episodic memory | Shinn 2023 |
| **Chain-of-Verification (CoVe)** | Verify each baseline claim independently | Dhuliawala 2023 |
| **Self-Discover** | SELECT → ADAPT → IMPLEMENT → SOLVE | Zhou 2024 |
| **Constitutional AI** | Per-rule pass/fail → revise | Bai 2022 |

</details>

<details>
<summary><b>Sampling &amp; Search</b></summary>

| Architecture | Pattern | Reference |
|---|---|---|
| **Self-Consistency** | Sample N paths, majority-vote | Wang 2022 |
| **Tree of Thoughts** | Beam search over thoughts | Yao 2023 |
| **LATS** | MCTS tree with reward backup | Zhou 2024 |
| **Mental Loop** | Simulate → score (deterministic-picker) | this repo |
| **Ensemble** | N voters, weighted aggregation | this repo |

</details>

<details>
<summary><b>Retrieval (RAG)</b></summary>

| Architecture | Pattern | Reference |
|---|---|---|
| **Agentic RAG** | Agent decides when & what to retrieve | LangGraph reference |
| **Corrective RAG (CRAG)** | Grade docs, fall back to web | Yan 2024 |
| **Self-RAG** | Per-doc reflection tokens | Asai 2024 |
| **Adaptive RAG** | Pre-route by query complexity | Jeong 2024 |
| **GraphRAG** | KG + community summaries | Microsoft 2024 |

</details>

<details>
<summary><b>Memory</b></summary>

| Architecture | Stored unit | Reference |
|---|---|---|
| **Episodic + Semantic** | Conversation turns + triples | Park 2023 |
| **Graph Memory** | (subject, predicate, object) triples | this repo |
| **MemGPT** | OS-style context + archival tiers | Packer 2023 |
| **Voyager** | Reusable Python skills (real subprocess) | Wang 2023 |
| **Agent Workflow Memory** | High-level workflow recipes | Wang 2024 |

</details>

<details>
<summary><b>Tools &amp; Actions</b></summary>

| Architecture | Pattern | Reference |
|---|---|---|
| **Tool Use** | Agent with one tool | LangChain reference |
| **ReAct** | Thought → Action → Observation | Yao 2022 |
| **Planning** | Decompose → execute → replan | Wei 2022 |
| **Plan-Execute-Verify (PEV)** | Post-execution verification per step | this repo |
| **SWE-Agent** | Sandboxed file-system agent | Yang 2024 |
| **BrowserAgent** | **Real Playwright** + safety gate | Anthropic Computer-Use 2024 |

</details>

<details>
<summary><b>Multi-Agent</b></summary>

| Architecture | Pattern | Reference |
|---|---|---|
| **Multi-Agent** | Supervisor + specialists | LangGraph reference |
| **Blackboard** | Shared workspace + agents | classical AI |
| **Debate** | N agents × K rounds | Du 2023 |
| **STORM** | Multi-perspective research → article | Shao 2024 |
| **Meta-Controller** | Router over architectures | this repo |

</details>

<details>
<summary><b>Safety, Routing &amp; Specialty</b></summary>

| Architecture | Pattern | Reference |
|---|---|---|
| **Dry-Run** | Propose → simulate → approval gate | this repo |
| **Reflexive Metacognitive** | Self-aware capability routing | this repo |
| **RLHF Self-Improvement** | Multi-dim deterministic scoring + archive | this repo |
| **Cellular Automata** | LLM rules over a grid | this repo |

</details>

---

## Provider compatibility

<table>
<tr><th>Provider</th><th>Install extra</th><th>Notes</th></tr>
<tr><td><b>Nebius</b> &nbsp;<sub><i>(default)</i></sub></td><td><code>[nebius]</code></td><td>Llama-3.3-70B + Qwen3-Thinking; cheapest for the included demos</td></tr>
<tr><td>OpenAI</td><td><code>[openai]</code></td><td>All architectures work; highest quality for reasoning patterns</td></tr>
<tr><td>Anthropic</td><td><code>[anthropic]</code></td><td>Strong on long context; required for production Computer-Use</td></tr>
<tr><td>Groq</td><td><code>[groq]</code></td><td>Fast inference; great for high-volume Self-Consistency</td></tr>
<tr><td>Ollama &nbsp;<sub><i>(local)</i></sub></td><td><code>[ollama]</code></td><td>No API key; tool calling depends on the model</td></tr>
<tr><td>Together</td><td><code>[together]</code></td><td>Wide model catalogue</td></tr>
<tr><td>Fireworks</td><td><code>[fireworks]</code></td><td>Function-calling first-class</td></tr>
<tr><td>Mistral</td><td><code>[mistral]</code></td><td>EU-hosted option</td></tr>
<tr><td>Google</td><td><code>[google]</code></td><td>Gemini 2.x via Generative AI API</td></tr>
</table>

Switch via `LLM_PROVIDER` + the corresponding key in `.env`. No code changes.

---

## Benchmarks

A 17-task suite runs every architecture and scores results. Most recent run, real Nebius Llama-3.3-70B, ~25 min, ~$1.50 in tokens:

<table>
<tr>
<th width="180">Outcome</th>
<th>Architectures</th>
</tr>
<tr>
<td><b>Strong</b> &nbsp;<sub>2/2 or 3/3</sub></td>
<td><sub><code>Reflection</code> &nbsp;<code>SelfConsistency</code> &nbsp;<code>SelfDiscover</code> &nbsp;<code>BrowserAgent</code></sub></td>
</tr>
<tr>
<td><b>Perfect on attempted</b> &nbsp;<sub>1/1</sub></td>
<td><sub>21 more — see leaderboard</sub></td>
</tr>
<tr>
<td><b>Pattern-fit failures</b></td>
<td><sub>LATS on arithmetic (wrong shape) · Debate + Ensemble on Sally trick (group-think) · Reflexion + AWM on raw-fact recall (wrong memory shape)</sub></td>
</tr>
<tr>
<td><b>Overall</b></td>
<td><b>33 / 42 correct</b> &nbsp;<sub>78%</sub></td>
</tr>
</table>

<sub><b>Full leaderboard</b> with per-task answer excerpts: <a href="https://fareedkhan-dev.github.io/all-agentic-architectures/benchmarks/">fareedkhan-dev.github.io/.../benchmarks/</a></sub>

---

## Learning paths

Four curated reading orders, depending on what you're trying to do.

<table>
<tr><th width="180">Path</th><th>For</th><th>Order</th></tr>
<tr>
<td><b>Beginner</b></td>
<td>Mental model</td>
<td><sub>Reflection → Tool Use → ReAct → Planning → Self-Consistency</sub></td>
</tr>
<tr>
<td><b>RAG-focused</b></td>
<td>Production retrieval</td>
<td><sub>Agentic RAG → CRAG → Self-RAG → Adaptive RAG → GraphRAG</sub></td>
</tr>
<tr>
<td><b>Multi-agent</b></td>
<td>Coordination</td>
<td><sub>Multi-Agent → Blackboard → Debate → STORM → Meta-Controller</sub></td>
</tr>
<tr>
<td><b>Safety</b></td>
<td>Guardrails</td>
<td><sub>Dry-Run → Constitutional AI → Reflexive Metacognitive → BrowserAgent (safety gate)</sub></td>
</tr>
</table>

---

## Star history

<div align="center">

<a href="https://www.star-history.com/#FareedKhan-dev/all-agentic-architectures&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FareedKhan-dev/all-agentic-architectures&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FareedKhan-dev/all-agentic-architectures&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FareedKhan-dev/all-agentic-architectures&type=Date" width="720" />
  </picture>
</a>

</div>

---

## Tested

```
pytest -q
283 passed, 37 skipped (env-gated integration), 1 warning in ~30s
```

<table>
<tr><th width="280">Suite</th><th>Coverage</th></tr>
<tr><td><b>Registry sweep</b></td><td>All 35 architectures (metadata + instantiate + build)</td></tr>
<tr><td><b>Pure-Python helpers</b></td><td>Haiku checker, composite scorers, subprocess executor, safety gate, sandbox path</td></tr>
<tr><td><b>Notebook integrity</b></td><td>All 35 notebooks executed, no error outputs, §9 commentary tailored from real captured runs</td></tr>
<tr><td><b>Integration</b> &nbsp;<sub><i>(env-gated)</i></sub></td><td>One real-LLM happy-path per architecture, gated via <code>RUN_INTEGRATION=1</code></td></tr>
</table>

---

## Documentation

| | |
|---|---|
| [**Full site**](https://fareedkhan-dev.github.io/all-agentic-architectures/) | Dark-mode aesthetic, embedded notebooks, auto-generated API reference |
| [Quickstart](https://fareedkhan-dev.github.io/all-agentic-architectures/getting-started/quickstart/) | One-command install, 8-line example |
| [Switching providers](https://fareedkhan-dev.github.io/all-agentic-architectures/getting-started/providers/) | Capability matrix; one env var to swap |
| [Add your own architecture](https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/adding-your-own/) | 5-step contributor recipe |
| [Deterministic-picker pattern](https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/deterministic-picker/) | The central technical pattern, explained once |
| [Memory variants](https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/memory/) | Comparison of all 7 memory shapes |
| [API reference](https://fareedkhan-dev.github.io/all-agentic-architectures/reference/) | mkdocstrings auto-gen from docstrings |
| [Benchmarks](https://fareedkhan-dev.github.io/all-agentic-architectures/benchmarks/) | Full per-task leaderboard with answer excerpts |

---

## Contributing

Contributions welcome. Two paths:

1. **Add a new architecture** — follow the [5-step recipe](https://fareedkhan-dev.github.io/all-agentic-architectures/tutorials/adding-your-own/). The PR template includes a deterministic-picker checklist.
2. **Improve an existing one** — bug fix, prompt tuning, performance, scoring rubric. Open an issue first to discuss scope.

See [**CONTRIBUTING.md**](CONTRIBUTING.md) for the dev setup, code style, and commit-message convention (Conventional Commits — `release-please` auto-generates the CHANGELOG).

---

## Citation

```bibtex
@misc{khan2026agentic,
  title         = {Agentic Architectures: A Library of 35 Production-Grade Agentic AI Patterns},
  author        = {Khan, Fareed},
  year          = {2026},
  howpublished  = {\url{https://github.com/FareedKhan-dev/all-agentic-architectures}},
  note          = {MIT licensed Python library and runnable textbook}
}
```

---

## License

[MIT](LICENSE) — © 2026 Fareed Khan.

<br>

<div align="center">

<sub>
Built on <a href="https://langchain-ai.github.io/langgraph/">LangGraph</a>
&nbsp;·&nbsp;
Docs powered by <a href="https://squidfunk.github.io/mkdocs-material/">Material for MkDocs</a>
&nbsp;·&nbsp;
Default LLM via <a href="https://nebius.com/">Nebius</a>
</sub>

<br><br>

<sub>
<a href="https://github.com/FareedKhan-dev/all-agentic-architectures/stargazers">★ Star</a>
&nbsp;·&nbsp;
<a href="https://github.com/FareedKhan-dev/all-agentic-architectures/fork">Fork</a>
&nbsp;·&nbsp;
<a href="https://github.com/FareedKhan-dev/all-agentic-architectures/issues/new/choose">Open an issue</a>
&nbsp;·&nbsp;
<a href="https://github.com/FareedKhan-dev/all-agentic-architectures/discussions">Discuss</a>
</sub>

</div>
