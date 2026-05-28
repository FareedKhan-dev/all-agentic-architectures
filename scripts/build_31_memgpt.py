"""Generate notebooks/31_memgpt.ipynb — OS-style tiered memory."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "31_memgpt.ipynb"

CELLS = [
    md("""# 31 · MemGPT — OS-style virtual memory tiers

> **TL;DR.** Agent has two memory tiers: **context** (bounded, FIFO-evicted) and **archival** (vector-backed, unbounded). Each step: decide action — write to archival, search archival, or answer.

| Property | Value |
|---|---|
| Origin | Packer et al., *MemGPT* (2023). [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) |
| Tiers | Context (RAM analog) + Archival (disk analog) |
| Picker | Categorical action — deterministic-picker |"""),
    md("""## 2 · Architecture

```mermaid
flowchart LR
    A([task]) --> D[DECIDE action]
    D --> E[EXECUTE]
    E -->|loop until answer| D
    E -->|answer| Z([final])

    C[(Context tier<br/>FIFO bounded)]
    AR[(Archival tier<br/>vector-backed)]
    E <-.write/search/page.-> C
    E <-.write/search.-> AR

    style D fill:#fff3e0,stroke:#f57c00
    style C fill:#e3f2fd,stroke:#1976d2
    style AR fill:#fce4ec,stroke:#c2185b
```"""),
    md("""## 3 · Theory

Two key ideas from the paper, preserved here:
1. **Eviction is automatic and lossless** — when context fills, the oldest item is pushed to archival, not discarded.
2. **Agent is the OS** — decides itself when to read/write each tier (no fixed retrieval policy).

Demo: multi-turn — ingest a fact (context gets it + archival), ingest more facts (context evicts the first → it survives in archival), ask about the original fact (agent must `search_archival`)."""),
    md("""## 4 · Setup"""),
    code("""from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import MemGPT
from agentic_architectures.ui import print_md, print_header
enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}")"""),
    md("""## 7 · Build the graph"""),
    code("""from IPython.display import Image, display
arch = MemGPT(llm=llm, context_limit=3, max_iterations=4)
graph = arch.build()
try: display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG unavailable: {e})")
    print(graph.get_graph().draw_mermaid())"""),
    md("""## 8 · Live run — multi-turn with eviction + recall

We feed 4 facts (one more than context_limit=3) then ask about the first fact (now evicted to archival)."""),
    code("""TURNS = [
    "Remember this: My favourite colour is teal.",
    "Remember this: I have a cat named Mochi.",
    "Remember this: I live in Reno, Nevada.",
    "Remember this: I play pickleball on Tuesdays.",  # forces eviction of turn 1
    "What is my favourite colour?",  # must paged-in from archival
]

for i, t in enumerate(TURNS, 1):
    r = arch.run(t)
    print(f"TURN_{i}: {t}")
    print(f"  ACTIONS: {r.metadata['actions_taken']}")
    print(f"  CONTEXT: {r.metadata['context_tier_before']} -> {r.metadata['context_tier_after']}")
    print(f"  ARCHIVAL: {r.metadata['archival_before']} -> {r.metadata['archival_after']}")
    print(f"  ANSWER: {r.output[:200]}")
    print()
print(f"FINAL_CONTEXT: {arch.context_tier}")
print(f"FINAL_ARCHIVAL_COUNT: {arch.archival_count}")"""),
    md("""## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_31_commentary.py`.)*"""),
    md("""## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Forgets to search archival** | Agent answers from context only, misses paged-out fact | Schema rule: if context doesn't contain X, you MUST search before answering |
| **Over-archives** | Every turn writes to archival even trivia | Threshold for what's worth archiving |
| **No eviction** | If context never fills, behaves like plain memory | Lower `context_limit` for the demo |

Extensions: (1) LRU eviction instead of FIFO, (2) tier-aware retrieval (preferring context hits over archival), (3) compression at eviction time (LLM-summarises before paging).

Reference: Packer et al., *MemGPT*. 2023. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)"""),
]

def main():
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")

if __name__ == "__main__": main()
