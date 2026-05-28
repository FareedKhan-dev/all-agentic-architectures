---
title: Agentic Architectures
hide:
  - navigation
  - toc
---

<div class="aa-hero" markdown>
<div class="aa-eyebrow">Library &nbsp;·&nbsp; Textbook &nbsp;·&nbsp; Benchmark</div>

# The complete library of agentic AI patterns.

<p class="aa-lede">
Thirty-five production-grade architectures, built on LangGraph. Real LLM outputs end-to-end, provider-agnostic, deterministic-picker discipline throughout — and a benchmark that ranks them all.
</p>

<div class="aa-cta">
  <a class="md-button md-button--primary" href="getting-started/quickstart/">Start building →</a>
  <a class="md-button" href="architectures/">Browse architectures</a>
  <a class="md-button" href="benchmarks/">See benchmarks</a>
</div>
</div>

<div class="aa-stats" markdown>
<div class="aa-stat"><div class="n">35</div><div class="lbl">Architectures</div></div>
<div class="aa-stat"><div class="n">283</div><div class="lbl">Passing tests</div></div>
<div class="aa-stat"><div class="n">17</div><div class="lbl">Benchmark tasks</div></div>
<div class="aa-stat"><div class="n">9</div><div class="lbl">LLM providers</div></div>
</div>

<div class="aa-pillars" markdown>

<div class="aa-pillar" markdown>
### Living textbook
Every notebook's commentary quotes the actual captured run — not a synthetic example. Theory written *against* observed behavior.
</div>

<div class="aa-pillar" markdown>
### Deterministic picker
Every LLM-as-Scorer surface uses a categorical commitment + Python composition. No flat-band pathology, by design.
</div>

<div class="aa-pillar" markdown>
### Provider agnostic
One `get_llm()` factory speaks Nebius, OpenAI, Anthropic, Groq, Ollama, Together, Fireworks, Mistral, Google.
</div>

<div class="aa-pillar" markdown>
### Comparable
A 17-task benchmark suite runs every architecture and scores results. See which pattern suits which task family.
</div>

</div>

## Eight families. Thirty-five patterns.

<div class="aa-bento">
<div class="aa-bento-card"><span class="cat">Reasoning &amp; Reflection</span><h3>Reflect, verify, revise.</h3><span class="desc">Self-critique loops that drive answer quality up through iteration. Catch hallucinations before they ship.</span><span class="members">Reflection · Reflexion · Chain-of-Verification · Self-Discover · Constitutional AI</span><a class="aa-card-link" href="architectures/01_reflection/" aria-label="Reasoning &amp; Reflection"></a></div>
<div class="aa-bento-card"><span class="cat">Sampling &amp; Search</span><h3>Many paths, one answer.</h3><span class="desc">Sample N reasoning trajectories and pick the modal answer; or grow a tree and search it with rewards.</span><span class="members">Self-Consistency · Tree of Thoughts · LATS · Mental Loop</span><a class="aa-card-link" href="architectures/21_self_consistency/" aria-label="Sampling &amp; Search"></a></div>
<div class="aa-bento-card"><span class="cat">Retrieval (RAG)</span><h3>Ground every claim.</h3><span class="desc">Five distinct retrieval shapes — from agent-decides-when, to corrective grading, to graph community summaries.</span><span class="members">Agentic RAG · Corrective RAG · Self-RAG · Adaptive RAG · GraphRAG</span><a class="aa-card-link" href="architectures/23_agentic_rag/" aria-label="Retrieval"></a></div>
<div class="aa-bento-card"><span class="cat">Memory</span><h3>Learn across calls.</h3><span class="desc">Episodic reflections, archival skills, OS-style tiered context, mined workflows — pick the right shape for your task.</span><span class="members">Episodic + Semantic · MemGPT · Voyager · Agent Workflow Memory</span><a class="aa-card-link" href="architectures/18_reflexion/" aria-label="Memory"></a></div>
<div class="aa-bento-card"><span class="cat">Tools &amp; Actions</span><h3>Act on the world.</h3><span class="desc">From a single search tool to a sandboxed code repo to a real Chromium browser.</span><span class="members">Tool Use · ReAct · Planning · PEV · SWE-Agent · BrowserAgent</span><a class="aa-card-link" href="architectures/02_tool_use/" aria-label="Tools"></a></div>
<div class="aa-bento-card"><span class="cat">Multi-Agent</span><h3>Coordinate many minds.</h3><span class="desc">Specialists, debate, ensembles, multi-perspective research. Beyond the single-agent loop.</span><span class="members">Multi-Agent · Blackboard · Ensemble · Debate · STORM</span><a class="aa-card-link" href="architectures/05_multi_agent/" aria-label="Multi-Agent"></a></div>
<div class="aa-bento-card"><span class="cat">Safety &amp; Routing</span><h3>Block before harm.</h3><span class="desc">Categorical actions through deterministic Python gates. Meta-control over a roster of specialists.</span><span class="members">Dry-Run · Reflexive Metacognitive · Meta-Controller · Computer Use</span><a class="aa-card-link" href="architectures/14_dry_run/" aria-label="Safety"></a></div>
</div>

## Quickstart

```bash
pip install agentic-architectures[nebius,faiss,tavily]
```

```python
from agentic_architectures import get_llm
from agentic_architectures.architectures import Reflection

arch = Reflection(llm=get_llm())
result = arch.run("Write a haiku about the sea.")
print(result.output)
print("score:", result.metadata["final_score"], "/ 10")
```

Same `.run(task)` interface across all 35 architectures. Same `ArchitectureResult` shape. Swap the class, swap the pattern — your downstream code doesn't change.

<div class="aa-cta-card" markdown>
## Start building.
Pick an architecture, paste a snippet, ship.

<div class="aa-cta">
  <a class="md-button md-button--primary" href="getting-started/quickstart/">Get started →</a>
  <a class="md-button" href="https://github.com/FareedKhan-dev/all-agentic-architectures">View on GitHub</a>
</div>
</div>
