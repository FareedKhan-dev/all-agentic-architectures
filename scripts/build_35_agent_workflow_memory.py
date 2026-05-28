"""Generate notebooks/35_agent_workflow_memory.ipynb — mine reusable workflows."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from notebook_builder import build_notebook, code, md  # noqa: E402

OUT_PATH = Path(__file__).parents[1] / "notebooks" / "35_agent_workflow_memory.ipynb"

CELLS = [
    md("""# 35 · Agent Workflow Memory (AWM) — mine reusable recipes from past traces

> **TL;DR.** After every solved task, extract a high-level **workflow recipe** (3-6 generalisable steps). Store recipes in a vector-indexed library. Future tasks retrieve the most-similar recipe and follow it.

| Property | Value |
|---|---|
| Origin | Wang et al., *Agent Workflow Memory* (2024). [arXiv:2409.07429](https://arxiv.org/abs/2409.07429) |
| Storage | Vector-indexed `arch.workflows` list |
| Cost | 1 retrieve + 1 answer + 1 extract per task = ~3 calls |
| Sister to | [Voyager (nb 29)](./29_voyager.ipynb) — skills (code) vs workflows (recipes) |"""),
    md("""## 2 · Architecture

```mermaid
flowchart LR
    A([task]) --> R[RETRIEVE most-similar workflow]
    R --> AN[ANSWER<br/><sub>prompt prepends recipe if found</sub>]
    AN --> EX[EXTRACT new workflow<br/><sub>generalised steps</sub>]
    EX --> Z([answer])

    L[(workflow library<br/>vector-indexed)]
    R <-.search.-> L
    EX -.add.-> L

    style R fill:#fff3e0,stroke:#f57c00
    style EX fill:#fce4ec,stroke:#c2185b
    style L fill:#f3e5f5,stroke:#7b1fa2
```"""),
    md("""## 3 · Theory

Voyager (nb 29) stores reusable **code** (skills). AWM stores reusable **strategy** (workflows). Skills are concrete; workflows are abstract. For tasks where a small Python function suffices, use Voyager. For tasks that share *structure* but differ in entities (e.g., "summarise then categorise" applies to news articles AND emails), use AWM.

Demo: 3 sequential tasks of similar structure ("summarise X then categorise it"). Task 1 extracts a workflow; tasks 2 and 3 retrieve and follow it."""),
    md("""## 4 · Setup"""),
    code("""from agentic_architectures import get_llm, enable_langsmith, settings
from agentic_architectures.architectures import AgentWorkflowMemory
from agentic_architectures.ui import print_md, print_header
enable_langsmith()
llm = get_llm(provider="nebius", model="meta-llama/Llama-3.3-70B-Instruct", temperature=0.2)
print_header(f"LLM: {llm.model}")"""),
    md("""## 7 · Build the graph"""),
    code("""from IPython.display import Image, display
arch = AgentWorkflowMemory(llm=llm)
graph = arch.build()
try: display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"(PNG unavailable: {e})")
    print(graph.get_graph().draw_mermaid())"""),
    md("""## 8 · Live run — 3 structurally-similar tasks"""),
    code("""TASKS = [
    "Summarise this in one sentence and then categorise it (news/opinion/research): 'A new study shows that octopuses may have evolved REM-like sleep states, suggesting that complex dreaming is older than previously thought.'",
    "Summarise this in one sentence and then categorise it (news/opinion/research): 'In my view, the new urban-planning rules will choke small businesses; the city council needs to reconsider before the December deadline.'",
    "Summarise this in one sentence and then categorise it (news/opinion/research): 'After fierce debate, Congress passed the infrastructure bill 218-211 on Tuesday, sending it to the President's desk.'",
]

for tag_idx, q in enumerate(TASKS, 1):
    r = arch.run(q)
    print(f"TASK_{tag_idx}: {q[:80]}...")
    print(f"  USED_RETRIEVED: {r.metadata['used_retrieved_workflow']}")
    print(f"  RETRIEVED_TYPE: {r.metadata['retrieved_workflow_type']!r}")
    print(f"  EXTRACTED_TYPE: {r.metadata['extracted_workflow_type']!r}")
    print(f"  LIBRARY: {r.metadata['library_size_before']} -> {r.metadata['library_size_after']}")
    print(f"  ANSWER: {r.output[:200]}")
    print()
print(f"FINAL_LIBRARY_SIZE: {len(arch.workflows)}")
for w in arch.workflows:
    print(f"  - workflow `{w['task_type']}`:")
    for s in w['steps']: print(f"      • {s}")"""),
    md("""## 9 · What we just observed

*(Automatically tailored from the actual captured run by `scripts/tailor_35_commentary.py`.)*"""),
    md("""## 11 · Failure modes & extensions

| Failure | Mitigation |
|---|---|
| **Workflow too specific** | Steps reference task-1 entities; tasks 2-3 don't fit | Strict prompt rule in extraction: "no specific entity names" |
| **Workflow library bloat** | N tasks → N near-duplicate workflows | Cosine-dedup on workflow descriptions |
| **Retrieved workflow misfires** | Wrong workflow retrieved; answer follows bad recipe | Add post-answer verification step |

Extensions: (1) Combine with Voyager skills (workflow specifies which skills to call), (2) hierarchical workflows (workflow contains sub-workflows), (3) workflow scoring by usefulness across tasks.

Reference: Wang et al., *AWM*. 2024. [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)"""),
]

def main():
    out = build_notebook(CELLS, OUT_PATH)
    print(f"wrote: {out}  ({sum(len(c[1]) for c in CELLS)} chars across {len(CELLS)} cells)")

if __name__ == "__main__": main()
