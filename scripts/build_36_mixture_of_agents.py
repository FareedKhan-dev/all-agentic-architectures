"""Generate notebooks/36_mixture_of_agents.ipynb — multi-layer MoA."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from notebook_builder import build_notebook, code, md  # noqa: E402


OUT_PATH = Path(__file__).parents[1] / "notebooks" / "36_mixture_of_agents.ipynb"


CELLS = [
    md(
        """# 36 · Mixture of Agents — layered proposer-aggregator collaboration

> **TL;DR.** MoA stacks LLMs in layers. Agents in layer 0 propose independently — each sees only the task. Layer k+1 agents see the task AND all outputs from layer k, producing refined proposals. A final aggregator synthesises. Cost: (num_layers × agents_per_layer + 1) LLM calls.
>
> **Reach for it when** the task benefits from progressive refinement (research synthesis, creative writing, complex reasoning). **Avoid when** a single LLM call would suffice — MoA adds latency and cost for multiplicative LLM calls.

| Property | Value |
|---|---|
| Origin | Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities* (2024) |
| Layers | 2 (default); configurable via `num_layers` |
| Agents / layer | 3 — Decomposer / Synthesizer / Critic |
| Aggregation | Python tally + LLM synthesis |
| Cost | `num_layers × agents_per_layer + 1` LLM calls |
| Distinct from Ensemble | MoA layers *build on each other's context*; Ensemble voters are parallel & independent |

This is **structurally different** from Ensemble (notebook 13). In Ensemble, all K voters see the SAME prompt and answer independently — aggregator merges them. In MoA, layer 1 agents READ layer 0 outputs and refine them — cross-layer context propagation."""
    ),
    md(
        """## 2 · Architecture at a glance

```mermaid
flowchart TB
    T([task]) --> L0[Layer 0<br/><sub>3 agents propose<br/>independently</sub>]
    L0 --> O0a[Decomposer]
    L0 --> O0b[Synthesizer]
    L0 --> O0c[Critic]
    O0a --> L1[Layer 1<br/><sub>3 agents see ALL<br/>layer 0 outputs</sub>]
    O0b --> L1
    O0c --> L1
    L1 --> O1a[Decomposer]
    L1 --> O1b[Synthesizer]
    L1 --> O1c[Critic]
    O1a --> Ag[Aggregator<br/><sub>Python tally +<br/>LLM synthesis</sub>]
    O1b --> Ag
    O1c --> Ag
    Ag --> Z([final answer])

    style L0 fill:#e3f2fd,stroke:#1976d2
    style L1 fill:#e8f5e9,stroke:#388e3c
    style Ag fill:#fff3e0,stroke:#f57c00
```

**Progressive refinement.** Each layer adds context from the previous layer. The final aggregator uses a Python-computed indicator tally (not an LLM-as-Scorer) to weight signals, then an LLM synthesises the final answer."""
    ),
    md(
        """## 3 · Theory

### 3.1 · Why layers?

MoA's core insight: LLMs get better answers when they see OTHER LLMs' answers first. A single LLM answering in isolation has blind spots. But if you show an LLM 3 diverse proposals from a previous layer, it can:
- **Fill gaps** — notice what prior agents missed
- **Resolve contradictions** — surface and address disagreements
- **Strengthen consensus** — reinforce the strongest arguments

### 3.2 · Agent personas

Each layer has 3 agent types (rotated across `agents_per_layer`):

| Persona | Role | Prompt core |
|---|---|---|
| **Decomposer** | Break task into sub-components, build step by step | Systematic, explicit reasoning chain |
| **Synthesizer** | Find patterns across information, weave coherent narrative | Clarity, narrative flow |
| **Critic** | Identify gaps, edge cases, weaknesses in existing proposals | Strengthen by surfacing what's missing |

### 3.3 · Where MoA sits

| Pattern | Layers? | Cross-layer context? | Use when |
|---|---|---|---|
| ReAct (nb 03) | No | n/a | single focused query |
| Ensemble (nb 13) | No — parallel fan-out | No — voters independent | contested questions, need diverse perspectives |
| Self-Consistency (nb 21) | No — N samples same prompt | No | classification / arithmetic where vote tally helps |
| Debate (nb 28) | No — iterative round-based | Yes — sequential critiques | controversial topics |
| **MoA** *(this notebook)* | **Yes — 2+ layers** | **Yes — layer k reads layer k-1** | **complex, open-ended tasks needing progressive refinement** |

### 3.4 · What goes wrong

1. **Context bloat** — each layer feeds ALL prior outputs (O(n²) tokens). At high `num_layers` or `agents_per_layer`, prompts overflow context windows.
2. **Amplification of errors** — if layer 0 agents all hallucinate the same wrong fact, layer 1 amplifies it.
3. **Persona dilution** — with many agents, the Critic persona gets drowned out by Decomposer/Synthesizer consensus.
4. **Cost explosion** — 2 layers × 3 agents + 1 aggregator = 7 LLM calls. Adding a 3rd layer: 10 calls. Price scales multiplicatively."""
    ),
    md("""## 4 · Setup"""),
    code(
        """from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import MixtureOfAgents
from agentic_architectures.ui import print_md, print_header, print_step

enable_langsmith()
print_header(f"Provider: {settings.llm_provider}  ·  Model: {settings.llm_model}")"""
    ),
    md(
        """## 5 · Library walkthrough

Source: [`src/agentic_architectures/architectures/mixture_of_agents.py`](../src/agentic_architectures/architectures/mixture_of_agents.py).

Three node types:

1. **`_layer_generate`** — runs `agents_per_layer` agents on the current layer. Each agent gets a persona (Decomposer / Synthesizer / Critic). Layer >0 agents see ALL prior layer outputs.
2. **`_aggregate`** — Python tallies categorical confidence indicators across the final layer's agents (deterministic picker), then an LLM synthesises the final answer.
3. **`_should_continue`** — the router: if `current_layer < num_layers`, loop back to `_layer_generate`; otherwise, route to `_aggregate`.

The `_should_continue` router creates a `layer → layer → ... → aggregate` loop — same pattern as Reflection and Reflexion but with cross-layer context propagation."""
    ),
    code(
        """from agentic_architectures.architectures.mixture_of_agents import _MoAProposal, MoAState
import json
print('--- _MoAProposal schema ---')
print(json.dumps(_MoAProposal.model_json_schema(), indent=2)[:500] + '...')"""
    ),
    md(
        """## 6 · State

| Field | Type | Set by |
|---|---|---|
| `task` | `str` | caller |
| `layer_outputs` | `list[dict]` (appended per layer) | `_layer_generate` |
| `current_layer` | `int` | `_layer_generate` (incremented) |
| `num_layers` | `int` | caller / default |
| `agents_per_layer` | `int` | caller / default |
| `final_output` | `str` | `_aggregate` |"""
    ),
    md("""## 7 · Build the graph"""),
    code(
        """from IPython.display import Image, display
arch = MixtureOfAgents(num_layers=2, agents_per_layer=3)
graph = arch.build()
display(Image(graph.get_graph().draw_mermaid_png()))"""
    ),
    md(
        """## 8 · Live run — research synthesis

A task where layered refinement genuinely helps: answering a complex, nuanced question that benefits from multiple perspectives and progressive improvement."""
    ),
    code(
        """TASK = (
    "What are the three most important architectural patterns "
    "for building reliable AI agent systems in production, and why? "
    "Include specific trade-offs for each."
)

result = arch.run(TASK)

print_header("Final MoA output")
print_md(result.output)
print()
print(f"Layers executed: {result.state['total_layers_executed']}")
print(f"Total agent calls: {result.state['total_agent_calls']}")"""
    ),
    md("""### 8.1 · Layer-by-layer trace"""),
    code(
        """for t in result.trace:
    if t.get('agent_idx', 0) == 0:
        print_step(f"=== LAYER {t['layer']} ===", "")
    print_step(
        f"Agent {t['agent_idx']+1} [{t['persona']}]",
        f"Key insight: {t['key_insight']}"
    )
    print_md(f"**Answer:** {t['refined_answer'][:300]}...")
    print()"""
    ),
    md(
        """## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_36_commentary.py`.)*"""
    ),
    md(
        """## 10 · Failure modes and extensions

### 10.1 · Where this breaks

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Context overflow** | `num_layers × agents_per_layer` outputs feed into each agent's prompt | Cap at 3 layers, 3 agents; use context-aware truncation |
| **Amplified hallucinations** | Layer 0 hallucinates → Layer 1 amplifies it | Persona diversity (Critic catches some); ground with external tools |
| **Cost explosion** | 7+ LLM calls per task | Reserve for tasks that genuinely need multi-pass refinement |
| **Critic overpowered** | One lonely Critic vs consensus of Decomposer + Synthesizer | Adjust agent ratio or give Critic more weight in final tally |

### 10.2 · Production safety

- **Context budget**: monitor total prompt tokens. MoA with 3 layers × 3 agents × 500-token responses ≈ 4500 tokens of cross-layer context — fine for modern 128K+ models, but watch the quadratic growth.
- **Idempotent by design**: no tool-calling, no side effects — safe for production without sandboxing.
- **Deterministic picker**: the final tally is Python-computed, not LLM-computed — avoids the flat-confidence pathology.

### 10.3 · Three extensions

1. **Heterogeneous models**: use a cheap model (Llama-3.3-8B) for layer 0, a strong model (Claude-Sonnet) for the final layer. Cost ~30% of full-MoA while keeping quality.
2. **Tool-augmented layers**: give layer 0 agents web search tools — ground the entire MoA pipeline in fresh information.
3. **Cross-layer debate**: instead of just reading prior outputs, let layer k agents *critique and rebut* specific layer k-1 proposals.

### 10.4 · What to read next

- [**13 · Ensemble**](./13_ensemble.ipynb) — same-task parallel voters (MoA's simpler cousin).
- [**28 · Multi-Agent Debate**](./28_agent_debate.ipynb) — adversarial back-and-forth.
- [**21 · Self-Consistency**](./21_self_consistency.ipynb) — N samples + majority vote.

### 10.5 · References

1. Wang, J. et al. *Mixture-of-Agents Enhances Large Language Model Capabilities.* 2024. [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
2. Du, Y. et al. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* 2023. [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
3. Wang, X. et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
""",
    ),
]


def main() -> None:
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
